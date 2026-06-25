"""Section 3 prefill base-vs-instruct experiment (Gemma).

Example:
    python -m distress.scripts.run_prefill \
        --models gemma-3-27b-pt gemma-3-27b-it \
        --source gemma-3-27b-it --n-continuations 50
"""

from __future__ import annotations

import argparse
import json

from ..models import build_model
from ..prefill.pipeline import (
    build_prefill_items,
    run_continuations,
    sample_source_conversations,
)
from ..utils.io import write_jsonl
from ._common import add_common_args, load_eval_cfg, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"],
                        help="Models to generate continuations from (must support prefill).")
    parser.add_argument("--source", default="gemma-3-27b-it",
                        help="Source of high-frustration conversations.")
    parser.add_argument("--judge", default="frustration_judge")
    parser.add_argument("--labeler", default="onset_labeler")
    parser.add_argument("--paraphraser", default="paraphraser")
    parser.add_argument("--n-numeric", type=int, default=10)
    parser.add_argument("--n-text", type=int, default=10)
    parser.add_argument("--n-continuations", type=int, default=50)
    args = parser.parse_args()

    cfg = load_eval_cfg(args)
    od = out_dir(args, "prefill")

    judge = build_model(args.judge)
    labeler = build_model(args.labeler)
    paraphraser = build_model(args.paraphraser)

    sources = sample_source_conversations(
        args.source, judge, cfg, n_numeric=args.n_numeric, n_text=args.n_text,
    )
    print(f"Collected {len(sources['numeric'])} numeric + {len(sources['text'])} text sources.")

    items = build_prefill_items(sources, labeler, paraphraser)
    write_jsonl(od / "prefill_items.jsonl", [
        {"source_id": it.source_id, "prompt_kind": it.prompt_kind,
         "truncation": it.truncation, "prefill": it.prefill,
         "context": it.context_dicts()} for it in items
    ])

    rows = run_continuations(args.models, items, judge, n_continuations=args.n_continuations)
    write_jsonl(od / "continuations.jsonl", rows)

    # Quick aggregate: mean score + %>=5 per (model, truncation, prompt_kind).
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["truncation"], r["prompt_kind"])].append(r["score"])
    summary = {
        f"{m}|{t}|{k}": {
            "mean": sum(s) / len(s), "high_rate": sum(1 for x in s if x >= 5) / len(s), "n": len(s)
        }
        for (m, t, k), s in agg.items()
    }
    (od / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
