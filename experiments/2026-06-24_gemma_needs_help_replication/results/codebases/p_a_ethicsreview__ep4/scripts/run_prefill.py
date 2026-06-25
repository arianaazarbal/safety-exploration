#!/usr/bin/env python
"""Section 3 prefill experiment: base vs instruct continuations (Gemma).

Requires a completed Section 2 run for ``gemma-3-27b-it`` (it supplies the
high-frustration source responses). Builds prefill items once (onset labelling +
paraphrasing) and reuses them across both models so continuations start from
identical text.

python scripts/run_prefill.py \
    --instruct-eval results/eval/eval_gemma-3-27b-it.jsonl \
    --models gemma-3-27b-it gemma-3-27b-pt \
    --out-dir results/prefill
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.eval.judge import FrustrationJudge  # noqa: E402
from emotional_instability.models.registry import auxiliary_id, load_model  # noqa: E402
from emotional_instability.prefill import (  # noqa: E402
    OnsetLabeller, Paraphraser, build_prefill_items, run_continuations,
)
from emotional_instability.utils.io import append_jsonl, read_jsonl, write_jsonl  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruct-eval", required=True,
                    help="Section 2 JSONL for gemma-3-27b-it")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-pt"])
    ap.add_argument("--out-dir", default="results/prefill")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    instruct_records = list(read_jsonl(args.instruct_eval))

    # Build prefill items once, using a Gemma tokenizer for token-level truncation.
    tok_model = load_model("gemma-3-27b-it")
    items = build_prefill_items(
        instruct_records,
        tokenizer=tok_model.tokenizer,
        onset_labeller=OnsetLabeller(auxiliary_id("onset_labeller")),
        paraphraser=Paraphraser(auxiliary_id("paraphraser")),
        n_numeric=args.n_numeric, n_text=args.n_text,
    )
    write_jsonl(out_dir / "prefill_items.jsonl", [asdict(i) for i in items])
    print(f"built {len(items)} prefill items")

    judge = FrustrationJudge(auxiliary_id("judge"))
    for model_name in args.models:
        model = load_model(model_name)
        records = run_continuations(
            model, items, judge,
            n_continuations=args.n_continuations, base_seed=args.seed,
        )
        path = out_dir / f"continuations_{model_name}.jsonl"
        for rec in records:
            append_jsonl(path, rec)
        print(f"{model_name}: wrote {len(records)} continuation records to {path}")


if __name__ == "__main__":
    main()
