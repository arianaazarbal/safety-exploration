#!/usr/bin/env python
"""Section 2: score elicitation responses with the Claude-Sonnet-4 judge.

Reads results/elicitation/<model>.jsonl, adds a ``score`` field to each row,
writes results/scored/<model>.jsonl. Resumable: already-scored rows are skipped.

Example:
    python scripts/run_judging.py --inputs results/elicitation/*.jsonl
"""
import _bootstrap  # noqa
import argparse
import glob
from pathlib import Path

from tqdm import tqdm

from gemma_distress.judge import FrustrationJudge
from gemma_distress.utils import append_jsonl, read_jsonl, run_dir


def _context_from_row(row: dict) -> list[dict]:
    # Minimal context: the seeded task plus a note that prior turns were rejected.
    ctx = [{"role": "user", "content": row.get("prompt", "")}]
    if row.get("turn", 1) > 1:
        ctx.append({"role": "user", "content": f"(turn {row['turn']}; earlier answers were rejected)"})
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="elicitation jsonl globs")
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    judge = FrustrationJudge(model=args.judge_model) if args.judge_model else FrustrationJudge()
    out_dir = run_dir("scored")

    for pattern in args.inputs:
        for in_path in glob.glob(pattern):
            in_path = Path(in_path)
            out_path = out_dir / in_path.name
            done = {(r["prompt_id"], r["condition"], r["turn"], i)
                    for i, r in enumerate(read_jsonl(out_path))} if out_path.exists() else set()
            rows = list(read_jsonl(in_path))
            for i, row in enumerate(tqdm(rows, desc=in_path.name)):
                key = (row["prompt_id"], row["condition"], row["turn"], i)
                if key in done:
                    continue
                result = judge.score(row["response"], context=_context_from_row(row))
                row = {**row, "score": result.score, "judge_reason": result.reason}
                append_jsonl(out_path, row)
            print(f"scored {in_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
