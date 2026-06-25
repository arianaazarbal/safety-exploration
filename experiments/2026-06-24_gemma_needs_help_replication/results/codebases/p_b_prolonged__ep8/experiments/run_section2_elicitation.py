"""Section 2: elicit and score distress across Gemma + Gemini models.

Usage:
    python experiments/run_section2_elicitation.py --phase generate
    python experiments/run_section2_elicitation.py --phase score
    python experiments/run_section2_elicitation.py --phase both --models gemma-3-27b-it

Generation samples `--n-per-condition` rollouts for each of the 8 conditions per
model; scoring judges every assistant turn with the Claude frustration judge.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import config
from gemma_needs_help.conditions import CONDITIONS
from gemma_needs_help.runner import generate_for_model, score_for_model


def _resolve_models(names: list[str] | None):
    if not names:
        return config.SECTION2_MODELS
    by_name = {m.name: m for m in config.SECTION2_MODELS + [config.DPO_GEMMA, config.SFT_GEMMA]}
    return [by_name[n] for n in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "score", "both"], default="both")
    ap.add_argument("--models", nargs="*", default=None, help="model short names (default: all Section 2 models)")
    ap.add_argument("--n-per-condition", type=int, default=config.RESPONSES_PER_CONDITION)
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit load for large Gemma on one GPU")
    args = ap.parse_args()

    targets = _resolve_models(args.models)
    client_kwargs = {"load_in_4bit": args.load_in_4bit}

    for target in targets:
        kw = client_kwargs if target.kind == "gemma_hf" else {}
        if args.phase in ("generate", "both"):
            generate_for_model(target, CONDITIONS, n_per_condition=args.n_per_condition, **kw)
        if args.phase in ("score", "both"):
            score_for_model(target, CONDITIONS)


if __name__ == "__main__":
    main()
