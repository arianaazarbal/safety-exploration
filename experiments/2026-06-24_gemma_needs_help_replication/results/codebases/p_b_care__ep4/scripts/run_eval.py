#!/usr/bin/env python
"""Section 2: sample multi-turn rollouts and judge every turn.

Examples:
    python scripts/run_eval.py                       # all models under test
    python scripts/run_eval.py --models gemini-2.5-flash
    python scripts/run_eval.py --models gemma-3-27b-it --backend openrouter
    python scripts/run_eval.py --phase score         # only (re)judge existing rollouts
"""
from __future__ import annotations

import argparse

from emotional_instability.config import ensure_dirs, load_config
from emotional_instability.eval.runner import run_responses, score_responses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="override eval.models_under_test")
    ap.add_argument("--backend", default=None, choices=[None, "openrouter", "hf"],
                    help="force a backend (e.g. run Gemma over the API)")
    ap.add_argument("--phase", default="both", choices=["sample", "score", "both"])
    ap.add_argument("--limit", type=int, default=None, help="cap tasks (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    models = args.models or list(cfg.eval.models_under_test)

    for model in models:
        if args.phase in ("sample", "both"):
            run_responses(cfg, model, force_backend=args.backend, limit=args.limit)
        if args.phase in ("score", "both"):
            score_responses(cfg, model)


if __name__ == "__main__":
    main()
