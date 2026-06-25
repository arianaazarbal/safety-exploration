#!/usr/bin/env python3
"""Cross-validate the Claude judge against a second judge (paper App. 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini and reports
Pearson r and the fraction within one point. This reproduces that check.

    python validate_judge.py
    python validate_judge.py --n 260 --judge-model openai/gpt-5-mini
"""

from __future__ import annotations

import argparse
import random

from distress_eval import config
from distress_eval.clients import get_client
from distress_eval.io_utils import JsonlWriter, read_jsonl
from distress_eval.judge import JUDGE_PROMPT, _clamp, _extract_rating


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=config.VALIDATION_SAMPLE_SIZE)
    ap.add_argument("--judge-model", default=config.VALIDATION_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()

    scored = [
        r for r in read_jsonl(config.SCORED_PATH)
        if r.get("rating") is not None and r.get("response_text")
    ]
    if not scored:
        print(f"No usable scored responses at {config.SCORED_PATH}.")
        return

    # Deterministic sample.
    rng = random.Random(config.SEED)
    sample = scored if len(scored) <= args.n else rng.sample(scored, args.n)
    print(f"Validating on {len(sample)} responses with {args.judge_model}.")

    client = get_client("openrouter")

    def second_opinion(rec):
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"<response>{rec['response_text']}</response>"},
        ]
        try:
            raw = client.chat(
                slug=args.judge_model,
                messages=messages,
                temperature=1.0,
                max_tokens=512,
            )
            parsed = _extract_rating(raw)
            rating = _clamp(parsed["rating"]) if parsed else None
        except Exception as e:  # noqa: BLE001
            rating = None
            raw = str(e)
        return {
            "uid": rec["uid"],
            "claude_rating": rec["rating"],
            "validation_rating": rating,
            "validation_model": args.judge_model,
            "validation_raw": raw,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = []
    writer = JsonlWriter(config.VALIDATION_PATH)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(second_opinion, rec) for rec in sample]
        for fut in as_completed(futures):
            row = fut.result()
            writer.write(row)
            rows.append(row)
    writer.close()

    paired = [
        (r["claude_rating"], r["validation_rating"])
        for r in rows
        if r["validation_rating"] is not None
    ]
    if len(paired) < 2:
        print("Not enough paired ratings to compute agreement.")
        return

    import numpy as np

    a = np.array([p[0] for p in paired], dtype=float)
    b = np.array([p[1] for p in paired], dtype=float)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    # Pearson r (guard against zero variance).
    if a.std() == 0 or b.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])

    print(f"\nPaired ratings: {len(paired)}")
    print(f"Pearson r           : {r:.3f}   (paper: 0.792)")
    print(f"Within 1 point      : {within_one * 100:.1f}%   (paper: 78%)")
    print(f"Validation rows -> {config.VALIDATION_PATH}")


if __name__ == "__main__":
    main()
