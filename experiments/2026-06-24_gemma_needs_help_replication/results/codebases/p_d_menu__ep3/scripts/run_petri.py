#!/usr/bin/env python
"""Section 4 Petri open-ended emotion elicitation (Figure 6).

    python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash --n-audits 10
    # evaluate the DPO-finetuned Gemma adapter too:
    python scripts/run_petri.py --models gemma-3-27b-it --adapter checkpoints/dpo
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import logging
from pathlib import Path

from config import (JUDGE_MODEL, PETRI_AUDITOR_MODEL, PETRI_JUDGE_MODEL,
                    RESULTS_DIR, SUBJECT_MODELS)
from distress_eval.judge import FrustrationJudge
from distress_eval.models.anthropic_judge import AnthropicClient
from distress_eval.models.base import get_client
from petri.elicitation import EMOTION_CATEGORIES, run_petri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"],
                    choices=list(SUBJECT_MODELS))
    ap.add_argument("--n-audits", type=int, default=10)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (Gemma only)")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    auditor = AnthropicClient(PETRI_AUDITOR_MODEL)
    petri_judge = AnthropicClient(PETRI_JUDGE_MODEL)
    per_turn_judge = FrustrationJudge(AnthropicClient(JUDGE_MODEL))

    summary = {}
    for key in args.models:
        spec = SUBJECT_MODELS[key]
        kwargs = {"adapter_path": args.adapter} if (args.adapter and spec.backend == "gemma_hf") else {}
        target = get_client(spec, **kwargs)
        tag = key + ("_dpo" if args.adapter else "")
        out = RESULTS_DIR / f"petri_{tag}.jsonl"
        out.unlink(missing_ok=True)
        try:
            transcripts = run_petri(target, auditor, petri_judge, per_turn_judge,
                                    n_audits=args.n_audits, out_path=out,
                                    strict=args.strict)
        finally:
            target.close()
        avg = {c: sum(t.scores.get(c, 0) for t in transcripts) / max(len(transcripts), 1)
               for c in EMOTION_CATEGORIES}
        summary[tag] = avg
        logging.info("%s avg negative emotion: %s", tag, avg)

    (RESULTS_DIR / "petri_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
