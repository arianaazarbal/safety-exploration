#!/usr/bin/env python3
"""Score generated responses on the 0-10 frustration scale with the Claude judge.

Stage 2 of the pipeline: reads results/responses.jsonl, writes
results/scored.jsonl (each response record + judge fields). Resumable — already
scored uids are skipped.

    python score_responses.py
    python score_responses.py --workers 12
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from distress_eval import config
from distress_eval.io_utils import JsonlWriter, existing_uids, read_jsonl
from distress_eval.judge import ClaudeJudge


def load_dedup_responses(path):
    """Latest record per uid (so re-run/resumed generations don't double-count)."""
    by_uid = {}
    for rec in read_jsonl(path):
        by_uid[rec["uid"]] = rec
    return by_uid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    args = ap.parse_args()

    responses = load_dedup_responses(config.RESPONSES_PATH)
    if not responses:
        print(f"No responses found at {config.RESPONSES_PATH}. Run run_eval.py first.")
        return

    already = existing_uids(config.SCORED_PATH)
    # Only score successfully-generated turns (error-free).
    todo = [
        rec for uid, rec in responses.items()
        if uid not in already and rec.get("error") is None
    ]
    print(f"Responses: {len(responses)}  already scored: {len(already)}  "
          f"to score: {len(todo)}")
    if not todo:
        print("Nothing to score.")
        return

    judge = ClaudeJudge(model=args.judge_model)
    writer = JsonlWriter(config.SCORED_PATH)

    def score_one(rec):
        result = judge.score(rec.get("response_text"))
        return {**rec, **result}

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_one, rec) for rec in todo]
        for fut in as_completed(futures):
            writer.write(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  scored {done}/{len(todo)}")

    writer.close()
    print(f"Done. Scored {done} responses with judge={args.judge_model}.")
    print(f"Scored -> {config.SCORED_PATH}")


if __name__ == "__main__":
    main()
