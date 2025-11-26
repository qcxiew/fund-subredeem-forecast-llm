
import json
import hashlib
import datetime
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import Chroma  # Deprecation warning is harmless but you can later switch to langchain_community
 
import argparse
import os
 
# ---- CLI and configuration ----
parser = argparse.ArgumentParser(
    description="Build or update the news Chroma vectorstore"
)
parser.add_argument(
    "--persist-dir",
    default="news_chroma_db",
    help="Chroma persist directory (default: news_chroma_db)"
)
args = parser.parse_args()
PERSIST_DIR = args.persist_dir
 
MODEL_NAME   = "Qwen/Qwen3-Embedding-4B"
# MODEL_NAME = "/home/xxx/modelSrc/modelscope/Qwen3/Qwen3_06B_embed/Qwen3-Embedding-0.6B"
BATCH_SIZE = 4
CHUNK_SIZE = 512
STRIDE = 128
 
print(f"s3_vectorstore_build CHUNK_SIZE= {CHUNK_SIZE} ; STRIDE= {STRIDE} -----33---")
 
# ---- model initialization ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    trust_remote_code=True
)
 
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).eval().to(device)
 
 
def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               stride: int = STRIDE) -> list[str]:
    """
    Split a long text into overlapping chunks by token count.
 
    - chunk_size: maximum number of tokens per chunk
    - stride:     overlap between consecutive chunks
    """
    ids = tokenizer.encode(text)
    chunks, start = [], 0
    total = len(ids)
    while start < total:
        end = min(start + chunk_size, total)
        sub = ids[start:end]
        chunks.append(tokenizer.decode(sub, skip_special_tokens=True))
        start += max(1, chunk_size - stride)
    return chunks
 
 
def get_embs(texts: list[str]) -> np.ndarray:
    """
    Compute embeddings for a batch of texts.
    Uses last-token pooling + L2 normalization.
    """
    all_embs = []
 
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        batch_dict = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CHUNK_SIZE,
            return_tensors="pt"
        ).to(device)
 
        with torch.no_grad():
            outputs = model(**batch_dict)
            hs = outputs.last_hidden_state  # (B, L, D)
 
        # Last non-padding token for each sample
        idx = batch_dict["attention_mask"].sum(1) - 1      # (B,)
        bidx = torch.arange(hs.size(0), device=device)
        emb = hs[bidx, idx]                                # (B, D)
        emb = F.normalize(emb, p=2, dim=1)                 # L2 normalization
        all_embs.append(emb.cpu().numpy())
 
        torch.cuda.empty_cache()
        time.sleep(0.1)  # small sleep for smoother GPU usage
 
    return np.concatenate(all_embs, axis=0)
 
 
def _dedupe_buffer(ids: list[str],
                   texts: list[str],
                   metas: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """
    Remove duplicate IDs within a single buffer, keeping the first occurrence.
 
    Chroma's `get(ids=...)` requires the list of IDs to be unique.
    This function enforces uniqueness in the batch before we call `get`.
    """
    seen = set()
    new_ids, new_texts, new_metas = [], [], []
    for i, uid in enumerate(ids):
        if uid in seen:
            continue
        seen.add(uid)
        new_ids.append(uid)
        new_texts.append(texts[i])
        new_metas.append(metas[i])
    return new_ids, new_texts, new_metas


def build_news_vectorstore(input_jsonl: str = "fund_news.jsonl"):
    """
    Incrementally build / update the Chroma vectorstore for news.
 
    Steps:
    1) Read all existing IDs from Chroma into a Python set.
    2) Stream the input JSONL file line by line.
    3) For each record, create text chunks and generate deterministic IDs (uid).
    4) Skip IDs already in the collection or already in the current buffer.
    5) For each buffer:
       - Deduplicate IDs inside the buffer.
       - Call coll.get(ids=...) to double-check presence.
       - Insert only truly new IDs with embeddings.
    6) Repeat until EOF, then flush any remaining buffer.
    """
 
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=None)
    coll = db._collection
    existing_ids: set[str] = set()
    batch = 10000
    offset = 0
 
    # 1) Load all existing IDs in pages
    while True:
        res = coll.get(offset=offset, limit=batch)
        ids = res["ids"]
        if not ids:
            break
        existing_ids.update(ids)
        offset += batch
 
    print(f"[start] Collection has {len(existing_ids)} embeddings")
 
    total_new = 0
    buf_texts: list[str] = []
    buf_ids: list[str] = []
    buf_metas: list[dict] = []
    buf_ids_set: set[str] = set()  # tracks IDs within current buffer
 
    # 2) Stream the JSONL file
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
 
            raw_ts = rec.get("datetime", rec.get("date", 0))
            if isinstance(raw_ts, (int, float)):
                # Milliseconds timestamp
                date_str = datetime.datetime.fromtimestamp(raw_ts / 1000).date().isoformat()
            else:
                date_str = str(raw_ts).split("T")[0]
 
            title = rec.get("title", "").strip()
            content = rec.get("content", "").strip()
            text = f"{title}\n\n{content}"
            chunks = chunk_text(text)
            url = rec.get("url", "").strip()
            title_md5 = hashlib.md5(title.encode("utf-8")).hexdigest()
 
            for idx, chunk in enumerate(chunks):
                # raw_id should be deterministic for identical (url, title, chunk index, chunk)
                raw_id = f"{url}|{title_md5}|{idx}|{chunk}"
                uid = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
 
                # Skip if already in collection or already in current buffer
                if uid in existing_ids or uid in buf_ids_set:
                    continue
 
                buf_texts.append(chunk)
                buf_ids.append(uid)
                buf_metas.append({
                    "id": uid,
                    "date": date_str,
                    "title": title,
                    "datetime": raw_ts
                })
                buf_ids_set.add(uid)
 
                # When buffer reaches BATCH_SIZE, flush it
                if len(buf_ids) >= BATCH_SIZE:
                    # Deduplicate inside the buffer (safety)
                    tmp_ids, tmp_texts, tmp_metas = _dedupe_buffer(
                        buf_ids, buf_texts, buf_metas
                    )
                    if tmp_ids:
                        # Double-check with Chroma in case of race / concurrent updates
                        resp = coll.get(ids=tmp_ids)
                        present = set(resp["ids"])
 
                        new_ids = [u for u in tmp_ids if u not in present]
                        new_texts = [
                            tmp_texts[i] for i, u in enumerate(tmp_ids)
                            if u not in present
                        ]
                        new_metas = [
                            tmp_metas[i] for i, u in enumerate(tmp_ids)
                            if u not in present
                        ]
 
                        if new_ids:
                            embs = get_embs(new_texts).astype("float16")
                            coll.add(
                                ids=new_ids,
                                documents=new_texts,
                                metadatas=new_metas,
                                embeddings=embs.tolist()
                            )
                            db.persist()
                            total_new += len(new_ids)
                            existing_ids.update(new_ids)
                            print(
                                f"[Inserted] This batch adds {len(new_ids)} new entries, "
                                f"total_new={total_new}."
                            )
 
                    # Clear buffer
                    buf_texts.clear()
                    buf_ids.clear()
                    buf_metas.clear()
                    buf_ids_set.clear()
 
    # 3) Flush any remaining records smaller than one batch
    if buf_ids:
        tmp_ids, tmp_texts, tmp_metas = _dedupe_buffer(buf_ids, buf_texts, buf_metas)
        if tmp_ids:
            resp = coll.get(ids=tmp_ids)
            present = set(resp["ids"])
 
            new_ids = [u for u in tmp_ids if u not in present]
            new_texts = [
                tmp_texts[i] for i, u in enumerate(tmp_ids)
                if u not in present
            ]
            new_metas = [
                tmp_metas[i] for i, u in enumerate(tmp_ids)
                if u not in present
            ]
 
            if new_ids:
                embs = get_embs(new_texts).astype("float16")
                coll.add(
                    ids=new_ids,
                    documents=new_texts,
                    metadatas=new_metas,
                    embeddings=embs.tolist()
                )
                db.persist()
                total_new += len(new_ids)
                existing_ids.update(new_ids)
                print(
                    f"[Inserted] Final batch adds {len(new_ids)} new entries, "
                    f"total_new={total_new}."
                )
 
    total_after = coll.count()
    print(
        f"[Completed] Collection size changed from "
        f"{len(existing_ids) - total_new} to {total_after} (embeddings)."
    )
 
 
if __name__ == "__main__":
    build_news_vectorstore("fund_news.jsonl")


    

