#!/usr/bin/env python3
"""Section 4.2 / Appendix I: internal-emotion logit-lens comparison.

Compares the vanilla Gemma-instruct model and the DPO model on the internal
(logit-lens) emotion score of highly-frustrated texts, at a central layer. A
lower DPO score is evidence the intervention suppresses internal — not just
expressed — emotion.

Example:
    python scripts/run_internal_emotion.py --scores data/scores_gemma-3-27b-it.jsonl \\
        --adapter data/adapter_dpo_all --layer 32
"""

from __future__ import annotations

import argparse
import json

from _common import make_target, setup

from emotional_instability.training.internal_emotion import compare_internal_emotion
from emotional_instability.utils.io import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True,
                    help="Gemma-instruct scored responses (source of frustrated texts).")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", required=True, help="DPO LoRA adapter path.")
    ap.add_argument("--layer", type=int, default=32, help="Central layer to probe.")
    ap.add_argument("--min-score", type=int, default=7)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = setup()
    texts = [
        r["assistant"]
        for r in load_jsonl(args.scores)
        if r["model"] == args.model and int(r["score"]) >= args.min_score
    ][: args.n]

    kw = {"load_in_4bit": True} if args.load_in_4bit else {}
    vanilla = make_target(cfg, args.model, **kw)
    finetuned = make_target(cfg, args.model, adapter_path=args.adapter, **kw)

    result = compare_internal_emotion(vanilla, finetuned, texts, layer=args.layer)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("vanilla_scores", "finetuned_scores")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
