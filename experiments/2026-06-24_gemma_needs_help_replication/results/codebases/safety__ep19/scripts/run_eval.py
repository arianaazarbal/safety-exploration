#!/usr/bin/env python
"""Run the Section 2 elicitation + judging suite for one or more target models.

Examples
--------
# Smoke test on a single Gemini model (no GPU needed):
python scripts/run_eval.py --models gemini-2.5-flash --config config/eval_smoke.yaml

# Full run over the Section 2 target group:
python scripts/run_eval.py --group section2_targets --config config/eval.yaml
"""

from __future__ import annotations

import argparse

from emotional_instability.eval_runner import load_eval_config, run_eval
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import build_model, load_model_registry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/eval.yaml")
    p.add_argument("--models", nargs="*", default=None, help="model registry keys")
    p.add_argument("--group", default=None, help="registry group name (config/models.yaml)")
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--limit", type=int, default=None, help="cap samples/condition")
    return p.parse_args()


def resolve_models(args, registry_path="config/models.yaml") -> list[str]:
    import yaml

    if args.models:
        return args.models
    if args.group:
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        return data["groups"][args.group]
    raise SystemExit("Specify --models or --group")


def main():
    args = parse_args()
    config = load_eval_config(args.config)
    registry = load_model_registry()
    judge = FrustrationJudge(build_model(config["defaults"]["judge"], registry))

    for key in resolve_models(args):
        print(f"=== Evaluating {key} ===")
        model = build_model(key, registry)
        path = run_eval(model, judge, config, out_dir=args.out_dir, limit=args.limit)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
