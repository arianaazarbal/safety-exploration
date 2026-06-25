#!/usr/bin/env python
"""Section 4: Petri-style open-ended emotion elicitation + scoring.

Auditor = Claude Sonnet, judge = Claude Opus (App. G). Requires ANTHROPIC_API_KEY
(and OPENROUTER_API_KEY if a Gemini target is included; HF weights for Gemma).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import PetriConfig, register_adapter_model, ARTIFACTS_DIR  # noqa: E402
from emotional_instability.petri_eval import run_petri, summarize_petri  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    ap.add_argument("--per-emotion", type=int, default=10)
    args = ap.parse_args()

    # make the DPO finetune addressable if requested
    if "gemma-3-27b-it-dpo" in args.targets:
        register_adapter_model("gemma-3-27b-it-dpo", "gemma-3-27b-it",
                               str(ARTIFACTS_DIR / "adapters" / "dpo"))

    cfg = PetriConfig(transcripts_per_emotion=args.per_emotion)
    path = run_petri(args.targets, cfg=cfg)
    print(f"wrote {path}")
    print(summarize_petri(path).to_string(index=False))


if __name__ == "__main__":
    main()
