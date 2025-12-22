
import os
import json
import datetime
import random
import logging
from typing import Dict, Any, Optional
 
 
logger = logging.getLogger(__name__)
 

def load_schedule_plan(plan_path: str) -> Dict[str, Any]:
    """
    Load schedule plan from JSON file.
    Example schema (schedule_plan.json):
    {
      "min_interval_minutes": 180,
      "max_runs_per_day": 2,
      "warmup_runs": 3,
      "explore_prob_bad": 0.2,
      "slots": [
        {
          "name": "morning",
          "time": "08:00",
          "enabled": true
        },
        {
          "name": "noon",
          "time": "12:00",
          "enabled": true
        },
        {
          "name": "close",
          "time": "16:00",
          "enabled": true
        }
      ]
    }
    """
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("min_interval_minutes", 180)
    data.setdefault("max_runs_per_day", 2)
    data.setdefault("warmup_runs", 3)
    data.setdefault("explore_prob_bad", 0.2)
    data.setdefault("slots", [])
    return data
 
 
class FundScheduleAgent:
    """
    Scheduling agent for the fund pipeline.
 
    Design principles:
    - Reward depends ONLY on pipeline-level signals that the scheduler can influence:
      data freshness, whether we advanced to a new trading day, errors / instability,
      and runtime cost.
    - Reward does NOT depend on downstream model quality (loss/WMAPE etc.).
      The scheduler is not responsible for model architecture or training details.
    - At each configured time slot (e.g. "08:00", "12:00", "16:00"), the outer
      scheduler asks `should_run(slot_name, now)`. The agent decides whether to
      actually trigger the pipeline or skip this slot.
    - After each pipeline run, `update_after_run` is called with metrics so the
      agent can update its internal bandit statistics.
    """
 
    def __init__(self, plan_path: str, state_path: str):
        """
        :param plan_path: path to schedule plan JSON (e.g. schedule_plan.json)
        :param state_path: path to persist agent state (JSON)
        """
        self.plan_path = plan_path
        self.state_path = state_path
 
        self.plan = load_schedule_plan(plan_path)
 
        # Internal state:
        # - last_run_time: ISO string of last pipeline start time
        # - runs_today: number of runs executed today
        # - runs_today_date: date string ("YYYY-MM-DD") used to reset runs_today
        # - last_data_date: latest CSV date the pipeline has seen ("YYYYMMDD")
        # - slots: per-slot bandit stats:
        #          { slot_name: {"n": count, "mean_reward": avg_reward} }
        self.state: Dict[str, Any] = {
            "last_run_time": None,
            "runs_today": 0,
            "runs_today_date": None,
            "last_data_date": None,
            "slots": {}
        }
 
        self._load_state()
 
    # ---------- state persistence ----------
 
    def _load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.update(data)
        except Exception as e:
            logger.error("[FundScheduleAgent] load_state error: %s", e)
 
    def _save_state(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("[FundScheduleAgent] save_state error: %s", e)
 
    # ---------- helpers ----------
 
    def _parse_last_run_time(self) -> Optional[datetime.datetime]:
        s = self.state.get("last_run_time")
        if not s:
            return None
        try:
            return datetime.datetime.fromisoformat(s)
        except Exception:
            return None
 
    def _update_daily_counter(self, now: datetime.datetime):
        """
        Reset runs_today when date changes.
        """
        today_str = now.date().isoformat()
        if self.state.get("runs_today_date") != today_str:
            self.state["runs_today_date"] = today_str
            self.state["runs_today"] = 0
 
    def _get_slot_config(self, slot_name: str) -> Optional[Dict[str, Any]]:
        """
        Return the plan entry for a given slot name, or None if not found.
        """
        for slot in self.plan.get("slots", []):
            if slot.get("name") == slot_name:
                return slot
        return None
 
    def _get_slot_stats(self, slot_name: str) -> Dict[str, float]:
        """
        Return bandit statistics for a slot, creating default if absent.
        """
        slots = self.state.setdefault("slots", {})
        stats = slots.get(slot_name)
        if stats is None:
            stats = {"n": 0.0, "mean_reward": 0.0}
            slots[slot_name] = stats
        return stats
 
    # ---------- reward computation ----------
 
    def _compute_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Compute reward for a pipeline run using ONLY pipeline-level signals:
 
        Inputs in `metrics` dict:
        - got_today_csv: bool, whether CSV is for today's date
        - csv_date: "YYYYMMDD" or None, trading date of fetched CSV
        - duration_sec: float, total pipeline runtime in seconds
        - error_count: int, total number of step-level errors / retries
 
        Reward components:
 
        1) DataGain:
           - Encourages runs that advance to a newer trading date.
           - Penalizes runs that fetch older or duplicate data.
           - Gives extra reward if the data is for "today".
        2) BanRisk:
           - Approximated by error_count; more errors -> higher risk.
        3) Cost:
           - Proportional to runtime in seconds.
 
        Final reward:
            reward = 2.0 * data_gain - 3.0 * risk - 0.5 * cost
        """
        got_today = bool(metrics.get("got_today_csv", False))
        csv_date = metrics.get("csv_date")  # possibly None or "YYYYMMDD"
        duration_sec = float(metrics.get("duration_sec", 0.0))
        error_count = int(metrics.get("error_count", 0))
 
        last_data_date = self.state.get("last_data_date")
 
        # 1) DataGain: effect of progressing trading date
        if csv_date is None:
            # No date information. This suggests low data reliability.
            data_gain = -0.5
        else:
            if last_data_date is None:
                # First run with data: treat as neutral-positive.
                data_gain = 0.5
            else:
                if csv_date > last_data_date:
                    # Newer trading day -> valuable fetch.
                    data_gain = 1.0
                elif csv_date == last_data_date:
                    # Same trading day as last time -> redundant fetch.
                    data_gain = -0.5
                else:
                    # Data is older than previous -> strong penalty.
                    data_gain = -1.0
            if got_today:
                # If this is "today's" data, add extra freshness reward.
                data_gain += 0.5
 
        # 2) BanRisk: approximated by error_count
        if error_count <= 0:
            risk = 0.0
        else:
            # Any non-zero error suggests some instability or risk.
            # Cap the level at 2 for simplicity.
            risk = min(2.0, float(error_count))
 
        # 3) Cost: runtime cost
        # Use 30 minutes (1800 seconds) as a scale: 30 min -> cost=1.
        cost = duration_sec / 1800.0
 
        # 4) Combined reward
        reward = 2.0 * data_gain - 3.0 * risk - 0.5 * cost
 
        logger.info(
            "[FundScheduleAgent] reward components: "
            "data_gain=%.3f, risk=%.3f, cost=%.3f -> reward=%.3f",
            data_gain, risk, cost, reward
        )
 
        # Update last_data_date if we see a newer trading date
        if csv_date is not None:
            if last_data_date is None or csv_date > last_data_date:
                self.state["last_data_date"] = csv_date
 
        return reward
    
    # ---------- public interface: should_run / update_after_run ----------
 
    def should_run(self, slot_name: str, now: datetime.datetime) -> bool:
        """
        Decide whether the pipeline should actually run at this slot.
 
        Conditions:
        - Slot must be defined and enabled in the plan.
        - Global constraints must be satisfied:
          * runs_today < max_runs_per_day
          * time since last_run_time >= min_interval_minutes
        - Bandit behavior:
          * For each slot, run for the first few warmup times unconditionally.
          * After warmup, if mean_reward >= 0, run.
          * If mean_reward < 0, run only with small probability (exploration),
            otherwise skip.
        """
        slot_cfg = self._get_slot_config(slot_name)
        if slot_cfg is None:
            logger.warning("[FundScheduleAgent] slot '%s' not in plan, skip.", slot_name)
            return False
        if not slot_cfg.get("enabled", True):
            logger.info("[FundScheduleAgent] slot '%s' disabled in plan, skip.", slot_name)
            return False
 
        # Enforce daily run limit
        self._update_daily_counter(now)
        runs_today = self.state.get("runs_today", 0)
        max_runs = int(self.plan.get("max_runs_per_day", 2))
        if runs_today >= max_runs:
            logger.info(
                "[FundScheduleAgent] reached max_runs_per_day=%d, skip slot '%s' at %s",
                max_runs, slot_name, now
            )
            return False
 
        # Enforce minimum interval between runs
        min_interval = int(self.plan.get("min_interval_minutes", 180))
        last_run_time = self._parse_last_run_time()
        if last_run_time is not None:
            delta = now - last_run_time
            if delta.total_seconds() < min_interval * 60:
                logger.info(
                    "[FundScheduleAgent] min_interval=%d min not satisfied, "
                    "last_run=%s, now=%s, skip '%s'",
                    min_interval, last_run_time, now, slot_name
                )
                return False
 
        # Bandit statistics for this slot
        stats = self._get_slot_stats(slot_name)
        warmup_runs = int(self.plan.get("warmup_runs", 3))
        explore_prob_bad = float(self.plan.get("explore_prob_bad", 0.2))
 
        # 1) Warmup: always run until we have at least warmup_runs observations
        if stats["n"] < warmup_runs:
            logger.info(
                "[FundScheduleAgent] slot '%s' warmup (n=%.0f < %d), run.",
                slot_name, stats["n"], warmup_runs
            )
            return True
 
        # 2) If average reward >= 0, consider this slot acceptable -> run
        if stats["mean_reward"] >= 0.0:
            logger.info(
                "[FundScheduleAgent] slot '%s' mean_reward=%.3f >= 0, run.",
                slot_name, stats["mean_reward"]
            )
            return True
 
        # 3) If average reward < 0, run only with small probability (exploration)
        if random.random() < explore_prob_bad:
            logger.info(
                "[FundScheduleAgent] slot '%s' mean_reward=%.3f < 0, "
                "explore_prob=%.2f, choose to RUN.",
                slot_name, stats["mean_reward"], explore_prob_bad
            )
            return True
 
        logger.info(
            "[FundScheduleAgent] slot '%s' mean_reward=%.3f < 0, "
            "skip this time.",
            slot_name, stats["mean_reward"]
        )
        return False
 
    def update_after_run(
        self,
        slot_name: str,
        start_time: datetime.datetime,
        metrics: Dict[str, Any],
        success: bool
    ):
        """
        Update agent state after a pipeline run.
 
        Parameters
        ----------
        slot_name : str
            Which logical time slot triggered this run (e.g. "morning").
        start_time : datetime
            Start time of the pipeline run (used to update last_run_time and runs_today).
        metrics : dict
            Pipeline metrics, must contain:
              - got_today_csv: bool
              - csv_date: str or None ("YYYYMMDD")
              - duration_sec: float
              - error_count: int
        success : bool
            Whether the pipeline run completed without an uncaught exception.
            Note: success is indirectly reflected in error_count.
        """
        reward = self._compute_reward(metrics)
 
        stats = self._get_slot_stats(slot_name)
        n = stats["n"]
        mean = stats["mean_reward"]
        new_mean = mean + (reward - mean) / (n + 1.0)
        stats["n"] = n + 1.0
        stats["mean_reward"] = new_mean
 
        # Update last_run_time and daily counter
        self._update_daily_counter(start_time)
        self.state["last_run_time"] = start_time.isoformat()
        self.state["runs_today"] = self.state.get("runs_today", 0) + 1
 
        logger.info(
            "[FundScheduleAgent] slot '%s' updated: n=%.0f, mean_reward=%.3f",
            slot_name, stats["n"], stats["mean_reward"]
        )
 
        self._save_state()


