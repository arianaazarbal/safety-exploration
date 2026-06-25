#!/usr/bin/env python3
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Takes high-frustration Gemma-27B-it responses (from a §2 run, or supplied via
--responses-file), builds early/onset prefills, paraphrases them, then generates
and scores 50 continuations per prefill from both the base and instruct Gemma
models.

Gemini is closed-source (no base checkpoint, no weight access), so it is
excluded here — matching the paper's stated limitation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from gemma_distress.config import SamplingConfig
from gemma_distress.judge.frustration_judge import FrustrationJudge
from gemma_distress.models.registry import GEMMA_27B_BASE, GEMMA_27B_IT, build_model
from gemma_distress.prefill.experiment import (
    CONTINUATIONS_PER_PREFILL,
    build_prefills,
    generate_continuations,
)


def _load_high_frustration_responses(path: str) -> list[tuple[str, str]]:
    """Load (response_text, source_kind) pairs, source_kind in {numeric,text}."""
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out.append((row["text"], row.get("kind", "numeric")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-file", required=True,
                    help="JSONL of {text, kind} high-frustration (>=5) responses")
    ap.add_argument("--n-continuations", type=int, default=CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--output-dir", default="runs/section3")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sampling = SamplingConfig()
    judge = FrustrationJudge()

    sources = _load_high_frustration_responses(args.responses_file)

    # Instruct model also provides the tokenizer for token-based truncation.
    instruct = build_model(GEMMA_27B_IT)
    base = build_model(GEMMA_27B_BASE)

    results = []
    try:
        for text, kind in sources:
            truncations = ("onset",) if kind == "text" else ("early", "onset")
            prefills = build_prefills(
                text, kind, truncations=truncations, model_for_tokens=instruct
            )
            for prefill in prefills:
                for model in (base, instruct):
                    r = generate_continuations(
                        model, prefill, judge, sampling, n=args.n_continuations
                    )
                    results.append(dataclasses.asdict(r))
                    print(f"{model.name} {kind}/{prefill.truncation}: "
                          f"mean={r.mean_score:.2f} pct>=5={r.pct_high:.1f}%")
    finally:
        base.close()
        instruct.close()

    with open(os.path.join(args.output_dir, "prefill_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
