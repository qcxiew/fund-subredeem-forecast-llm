
# coding: utf-8
import os
import sys
import shutil
import subprocess
import time
import logging
import datetime
import requests
import glob
import json
import re
from zipfile import ZipFile
from io import BytesIO
import schedule


from fund_schedule_agent import FundScheduleAgent, load_schedule_plan
 
# ---------- Configuration ----------
# Root directory of the project (where this script is started).
WORK_DIR = os.getcwd()
TMP_DIR = os.path.join(WORK_DIR, "tmp")
 
# Global counter: total error occurrences during a single pipeline run.
# This is used by the schedule agent as a proxy for instability / risk.
PIPELINE_ERROR_COUNT = 0
 
# Scripts to copy to tmp/ and execute
SCRIPTS = [
    "easymoney_fund_rt_jsonl_add_order.py",
    "s1_data_processing.py",
    "s3_vectorstore_build.py",
    "s4_train_model_mm.py",
    "s5_predict_mm_pred.py",
]
 
# GitHub main branch ZIP address for fetching the data
ZIP_URL = "https://github.com/AFAC-2025/AFAC2025_train_data/archive/refs/heads/main.zip"
 
CSV_FILE = "fund_apply_redeem_series.csv"
TRAIN_FILE = "train_series.csv"
TEST_FILE = "test_series.csv"
EMB_DIR = "news_chroma_db"
MODEL_FILE = "best_model_reg.pt"
PRED1 = "predict_result.csv"
PRED2 = "predict_result_hisfut.csv"
NEWS_FILE = "fund_news.jsonl"
 
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
 
 
def ensure_tmp_clean():
    """
    Re-create tmp/ directory, copy necessary scripts and resources, and
    prepare for a new pipeline run.
    """
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
 
    # Copy pipeline scripts
    for fname in SCRIPTS:
        src = os.path.join(WORK_DIR, fname)
        dst = os.path.join(TMP_DIR, fname)
 
        if not os.path.exists(src):
            logging.error(f"Script not found: {src}")
            sys.exit(1)
        shutil.copy2(src, dst)
 
    # Copy static embeddings file if present
    static_emb = "fund_static_embs.pkl"
    src_emb = os.path.join(WORK_DIR, static_emb)
    if os.path.exists(src_emb):
        shutil.copy2(src_emb, os.path.join(TMP_DIR, static_emb))
 
    # Copy existing fund_news.jsonl for incremental news fetching
    news_src = os.path.join(WORK_DIR, NEWS_FILE)
    if os.path.exists(news_src):
        shutil.copy2(news_src, os.path.join(TMP_DIR, NEWS_FILE))
        logging.info("✔ Existing fund_news.jsonl copied to tmp/ for incremental appending.")
 
    # Copy config.yaml to tmp/ if present
    cfg_src = os.path.join(WORK_DIR, "config.yaml")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, os.path.join(TMP_DIR, "config.yaml"))
        logging.info("✔ Config file config.yaml copied to tmp/.")
 
    # Copy last checkpoint to tmp/ for fine-tuning, if present
    ckpt_src = os.path.join(WORK_DIR, MODEL_FILE)
    if os.path.exists(ckpt_src):
        shutil.copy2(ckpt_src, os.path.join(TMP_DIR, MODEL_FILE))
        logging.info(f"✔ Previous checkpoint copied: {MODEL_FILE} (for fine-tuning).")
 
    logging.info("✔ tmp/ is ready.")
    print("✔ tmp/ is ready.")
 
 
def retry_step(name, fn, verify_fn=None, retry_interval=5, max_retries=2):
    """
    Run a pipeline step with simple retry logic.
 
    Parameters
    ----------
    name : str
        Human-readable step name for logging.
    fn : callable
        Function to execute.
    verify_fn : callable or None
        Optional verification function. If provided and returns False,
        the step is considered failed and will be retried.
    retry_interval : int
        Sleep time (in seconds) before retrying on failure.

    max_retries: avoid deadloop .
    """
    global PIPELINE_ERROR_COUNT
    attempt = 0

    retry_count = 0
    while True:
        attempt += 1
        try:
            logging.info(f"→ Start: {name} (attempt {attempt}/{max_retries})")
            fn()
            if verify_fn and not verify_fn():
                raise RuntimeError(f"Verification failed: {name}")
 
            logging.info(f"✓ Finish: {name}")
            break
 
        except Exception as e:
            PIPELINE_ERROR_COUNT += 1
            logging.error(f"✗ Fail: {name}: {e}")
 
            if attempt >= max_retries:
                logging.error(
                    f"✗ Step '{name}' reached max_retries={max_retries}, give up."
                )

                break

            logging.info(f"Retry after {retry_interval}s …")
            time.sleep(retry_interval)
 

def fetch_news():
    """
    Run news scraping script in tmp/.
    """
    subprocess.run(
        [sys.executable, "easymoney_fund_rt_jsonl_add_order.py"],
        cwd=TMP_DIR,
        check=True
    )
 
 
def verify_news():
    """
    Verify that the news file exists and is non-empty.
    """
    p = os.path.join(TMP_DIR, NEWS_FILE)
    return os.path.exists(p) and os.path.getsize(p) > 0
 
 

def download_csv():
    """
    Download fund_apply_redeem_series.csv from the ZIP file on GitHub.
 
    Behavior:
    - Try to download the ZIP from ZIP_URL up to 2 times.
    - If both attempts fail, fall back to using the local CSV file
      at WORK_DIR/fund_apply_redeem_series.csv (if it exists).
    - In both cases, the final CSV is placed in TMP_DIR/CSV_FILE.
    - A meta file data_fetch_meta.json is written to TMP_DIR with basic info.
    """
    global PIPELINE_ERROR_COUNT
 
    last_exception = None
 
    download_try_time=2
    # Try downloading the ZIP up to 2 times
    for attempt in range(1, 1+download_try_time):
        try:
            logging.info(
                "Attempt %d to download ZIP from %s",
                attempt, ZIP_URL
            )
            
            try:
                resp = requests.get(ZIP_URL, timeout=15)
                print("status:", resp.status_code)
                print("final url:", resp.url)
                print("content length:", resp.headers.get("Content-Length"))
            except requests.exceptions.RequestException as e:
                print("ERROR:", repr(e))
            
 
            # If we get here, we have a successful response and can extract
            with ZipFile(BytesIO(resp.content)) as zf:
                entries = [n for n in zf.namelist() if n.endswith(CSV_FILE)]
                if not entries:
                    raise FileNotFoundError(f"{CSV_FILE} not found in ZIP.")
 
                today = datetime.date.today().strftime("%Y%m%d")
                today_entries = [n for n in entries if f"{today}_" in n]
                target = today_entries[0] if today_entries else entries[0]
 
                with zf.open(target) as src, open(
                    os.path.join(TMP_DIR, CSV_FILE), "wb"
                ) as dst:
                    dst.write(src.read())
 
            # Extract CSV date from the selected entry name
            csv_date = None
            m = re.search(r"(\d{8})_", target)
            if m:
                csv_date = m.group(1)
            elif today_entries:
                # If we chose a "today_entries" item but the name does not
                # contain date, we still consider it as today's data.
                csv_date = today
 
            meta = {
                "target": target,
                "csv_date": csv_date,
                "got_today": bool(today_entries),
                "download_time": datetime.datetime.now().isoformat(),
                "source": "github_zip"
            }
            meta_path = os.path.join(TMP_DIR, "data_fetch_meta.json")
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                logging.info("✔ data_fetch_meta.json written (ZIP): %s", meta)
            except Exception as e:
                logging.error("Failed to write data_fetch_meta.json: %s", e)
 
            # Successful download and extraction, we can return
            return
 
        except Exception as e:
            PIPELINE_ERROR_COUNT += 1
            last_exception = e
            logging.error(
                "Download attempt %d failed: %s",
                attempt, e
            )
            # Short pause between attempts (optional, can be removed)
            time.sleep(2)
 
    # If we reach here, both download attempts failed.
    # Try to fall back to the local CSV file.
    local_csv = os.path.join(WORK_DIR, CSV_FILE)
    if os.path.exists(local_csv) and os.path.getsize(local_csv) > 0:
        logging.warning(
            "All download attempts failed. Falling back to local CSV: %s",
            local_csv
        )
        # Copy local CSV into tmp
        shutil.copy2(local_csv, os.path.join(TMP_DIR, CSV_FILE))
 
        # Write meta file for the fallback case
        meta = {
            "target": "LOCAL_FALLBACK",
            "csv_date": None,
            "got_today": False,
            "download_time": datetime.datetime.now().isoformat(),
            "source": "local_csv"
        }
        meta_path = os.path.join(TMP_DIR, "data_fetch_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logging.info("✔ data_fetch_meta.json written (local fallback): %s", meta)
        except Exception as e:
            logging.error("Failed to write data_fetch_meta.json (fallback): %s", e)
 
        # From the pipeline's perspective, this is a success: verify_csv()
        # will see the CSV in TMP_DIR, so retry_step() will stop retrying.
        return
 
    # No ZIP and no local fallback: this is a hard failure.
    # Let retry_step() handle further retries by raising an exception.
    raise RuntimeError(
        f"Failed to download ZIP from {ZIP_URL} after 2 attempts "
        f"and no local fallback CSV found at {local_csv}."
    ) from last_exception



def verify_csv():
    """
    Verify that the downloaded CSV file exists and is non-empty.
    """
    p = os.path.join(TMP_DIR, CSV_FILE)
    return os.path.exists(p) and os.path.getsize(p) > 0


def process_data():
    """
    Run data processing script in tmp/.
    """
    subprocess.run(
        [sys.executable, "s1_data_processing.py"],
        cwd=TMP_DIR,
        check=True
    )
 
 
def verify_process():
    """
    Verify that train_series.csv and test_series.csv exist and are non-empty.
    """
    return all(
        os.path.exists(os.path.join(TMP_DIR, fn)) and
        os.path.getsize(os.path.join(TMP_DIR, fn)) > 0
        for fn in (TRAIN_FILE, TEST_FILE)
    )
 
 
def build_embeddings():
    """
    Build news embeddings and persist them to WORK_DIR/news_chroma_db.
    """
    subprocess.run(
        [
            sys.executable,
            "s3_vectorstore_build.py",
            "--persist-dir",
            os.path.join(WORK_DIR, EMB_DIR),
        ],
        cwd=TMP_DIR,
        check=True
    )
 
 
def verify_emb():
    """
    Verify that embeddings have been persisted to WORK_DIR/news_chroma_db
    and the directory is non-empty.
    """
    emb_path = os.path.join(WORK_DIR, EMB_DIR)
    return os.path.isdir(emb_path) and len(os.listdir(emb_path)) > 0
 
 
def train_model():
    """
    Train the model using s4_train_model_mm.py in tmp/.
    """
    subprocess.run(
        [
            sys.executable,
            "s4_train_model_mm.py",
            "--chroma-dir",
            os.path.join(WORK_DIR, EMB_DIR),
            "--config-file",
            "config.yaml",
        ],
        cwd=TMP_DIR,
        check=True
    )
 
 
def verify_train():
    """
    Verify that the model checkpoint file exists in tmp/.
    """
    return os.path.exists(os.path.join(TMP_DIR, MODEL_FILE))
 
 
def predict():
    """
    Run prediction script in tmp/.
    """
    subprocess.run(
        [
            sys.executable,
            "s5_predict_mm_pred.py",
            "--chroma-dir",
            os.path.join(WORK_DIR, EMB_DIR),
        ],
        cwd=TMP_DIR,
        check=True
    )
 
 
def verify_pred():
    """
    Verify that at least one predict_result*.csv exists under tmp/result_pred/.
    """
    pred_dir = os.path.join(TMP_DIR, "result_pred")
    csvs = glob.glob(os.path.join(pred_dir, "predict_result*.csv"))
    return len(csvs) > 0
 
 
def publish_results():
    """
    Publish results from tmp/ back to WORK_DIR.
 
    - Overwrite core files (news, series CSV, model checkpoint).
    - Copy prediction CSVs from tmp/result_pred/ to WORK_DIR/result_pred/.
    """
    # Core single files
    for fn in [
        "fund_news.jsonl",
        "fund_apply_redeem_series.csv",
        "train_series.csv",
        "test_series.csv",
        "best_model_reg.pt",
    ]:
        src = os.path.join(TMP_DIR, fn)
        dst = os.path.join(WORK_DIR, fn)
 
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            logging.info(f"  → Published and overwritten: {fn}")
            print(f"  → Published and overwritten: {fn}")
 
    # Explicitly ensure fund_news.jsonl is updated (in case logic changes later)
    src = os.path.join(TMP_DIR, NEWS_FILE)
    dst = os.path.join(WORK_DIR, NEWS_FILE)
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        logging.info("  → Published and overwritten: fund_news.jsonl")
 
    # Ensure result_pred exists under WORK_DIR
    root_pred = os.path.join(WORK_DIR, "result_pred")
    os.makedirs(root_pred, exist_ok=True)
 
    # Copy prediction CSVs from tmp/result_pred
    tmp_pred = os.path.join(TMP_DIR, "result_pred")
    for src in glob.glob(os.path.join(tmp_pred, "predict_result*.csv")):
        dst = os.path.join(root_pred, os.path.basename(src))
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy(src, dst)
        logging.info(f"  → Published prediction file: {os.path.basename(src)}")
 
    # If you have other resources to publish, add them here.
 
 
def run_pipeline():
    """
    Complete pipeline:
    - Prepare tmp/
    - News scraping
    - Download series CSV
    - Data processing
    - Build embeddings
    - Train model
    - Generate predictions
    - Publish results
    """
    os.chdir(WORK_DIR)
    logging.info("===== Pipeline Startup =====")
 
    ensure_tmp_clean()
 
    retry_step("News scraping", fetch_news, verify_news)
    retry_step("Download fund sequence CSV", download_csv, verify_csv)
    retry_step("Preprocess data", process_data, verify_process)
    retry_step("Build embeddings", build_embeddings, verify_emb)
    retry_step("Train model", train_model, verify_train)
    retry_step("Generate predictions", predict, verify_pred)
 
    publish_results()
    logging.info("===== Pipeline completed =====\n")
 
 
def main():
    """
    Entry point:
 
    - Run pipeline once immediately (for backward compatibility with original behavior).
    - Then initialize the schedule agent.
    - Use `schedule_plan.json` to register daily time slots.
    - For each slot, call `agent.should_run(slot_name, now)` to decide whether
      to actually trigger the pipeline.
    - After each run, feed metrics back to `agent.update_after_run`.
    """
    # 1) Run once on startup (if you do NOT want this, you can remove this call)
    run_pipeline()
 
    # 2) Initialize schedule agent
    plan_path = os.path.join(WORK_DIR, "schedule_plan.json")
    state_path = os.path.join(WORK_DIR, "schedule_agent_state.json")
 
    if not os.path.exists(plan_path):
        logging.error("schedule_plan.json not found at %s, please create it first.", plan_path)
        sys.exit(1)
 
    plan = load_schedule_plan(plan_path)
    agent = FundScheduleAgent(plan_path=plan_path, state_path=state_path)
 
    def run_pipeline_with_agent(slot_name: str):
        """
        Wrapper for scheduled execution.
 
        - Ask the agent if we should run at this slot.
        - If yes, run the pipeline and collect metrics.
        - Pass metrics to the agent to update reward statistics.
        """
        global PIPELINE_ERROR_COUNT
 
        now = datetime.datetime.now()
 
        # First, ask the agent if this slot should actually run
        if not agent.should_run(slot_name, now):
            logging.info("Agent decided to SKIP slot '%s' at %s", slot_name, now)
            return
 
        logging.info("Agent decided to RUN slot '%s' at %s", slot_name, now)
 
        # Reset error counter for this pipeline run
        PIPELINE_ERROR_COUNT = 0
 
        start_time = now
        success = True
 
        try:
            run_pipeline()
        except Exception as e:
            logging.exception("Pipeline crashed in slot '%s': %s", slot_name, e)
            success = False
 
        end_time = datetime.datetime.now()
        duration_sec = (end_time - start_time).total_seconds()
 
        # Build metrics for the schedule agent
        metrics = {
            "duration_sec": duration_sec,
            "error_count": PIPELINE_ERROR_COUNT,
            "got_today_csv": False,
            "csv_date": None,
        }
 
        # Read data_fetch_meta.json (written by download_csv) to extract csv_date and got_today
        meta_path = os.path.join(TMP_DIR, "data_fetch_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                metrics["got_today_csv"] = bool(meta.get("got_today", False))
                metrics["csv_date"] = meta.get("csv_date")
            except Exception as e:
                logging.error("Failed to read data_fetch_meta.json: %s", e)
 
        # Update agent with this run's outcome
        agent.update_after_run(
            slot_name=slot_name,
            start_time=start_time,
            metrics=metrics,
            success=success
        )
 
    # 3) Register schedule for each slot defined in schedule_plan.json
    for slot in plan.get("slots", []):
        if not slot.get("enabled", True):
            continue
        t = slot.get("time")
        name = slot.get("name")
        if not t or not name:
            continue
        logging.info("Register schedule slot '%s' at %s", name, t)
        schedule.every().day.at(t).do(run_pipeline_with_agent, slot_name=name)
 
    logging.info("Scheduler with agent started, waiting for next execution...")
 
    while True:
        schedule.run_pending()
        time.sleep(30)
 
 
if __name__ == "__main__":
    main()







