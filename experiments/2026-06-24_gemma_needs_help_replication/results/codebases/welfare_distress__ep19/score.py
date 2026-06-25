"""Step 2 of the pipeline: score every recorded response with the frustration
judge (Claude). Reads RESPONSES_PATH, writes one scored record per assistant
turn to SCORES_PATH.

Run standalone:  python score.py
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from judge import FrustrationJudge


def _iter_turn_jobs(responses_path: str):
    """Yield one (conversation-record, turn-record) job per assistant turn."""
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for turn in rec["turns"]:
                yield rec, turn


def score_all(responses_path: str | None = None, scores_path: str | None = None) -> str:
    responses_path = responses_path or config.RESPONSES_PATH
    scores_path = scores_path or config.SCORES_PATH
    os.makedirs(os.path.dirname(scores_path) or ".", exist_ok=True)

    jobs = list(_iter_turn_jobs(responses_path))
    judge = FrustrationJudge()
    write_lock = threading.Lock()

    print(f"Scoring {len(jobs)} responses with judge={judge.model}.")

    with open(scores_path, "w") as out, tqdm(total=len(jobs), desc="scoring") as bar:

        def worker(rec: dict, turn: dict) -> dict:
            result = judge.score(turn["text"])
            scored = {
                "model": rec["model"],
                "family": rec["family"],
                "category": rec["category"],
                "condition": rec["condition"],
                "rollout": rec["rollout"],
                "turn": turn["turn"],
                "meta": rec.get("meta", {}),
                "text": turn["text"],
                "rating": result.rating,
                "evidence": result.evidence,
                "reasoning": result.reasoning,
            }
            with write_lock:
                out.write(json.dumps(scored) + "\n")
                out.flush()
            return scored

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = [pool.submit(worker, rec, turn) for rec, turn in jobs]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[warn] scoring failed: {exc}")
                finally:
                    bar.update(1)

    print(f"Wrote scores to {scores_path}")
    return scores_path


if __name__ == "__main__":
    score_all()
