#!/usr/bin/env python
"""Section 2: run the full distress-elicitation eval for one model and judge it.

Examples:
    python scripts/01_run_eval.py --model gemma-3-27b-it
    python scripts/01_run_eval.py --model gemini-2.5-flash
    python scripts/01_run_eval.py --model gemma-3-27b-it --no-score   # rollouts only
"""
import argparse
import json

import _bootstrap  # noqa: F401
from gemma_distress.eval.runner import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name (config/models.yaml)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-score", action="store_true", help="skip judging (rollouts only)")
    ap.add_argument("--tp-size", type=int, default=1, help="vLLM tensor-parallel size")
    args = ap.parse_args()

    backend_kwargs = {}
    # Tensor-parallel only applies to the local Gemma backend.
    from gemma_distress.config import get_model_spec
    if get_model_spec(args.model).backend == "hf_local":
        backend_kwargs["tensor_parallel_size"] = args.tp_size

    summ = run_eval(args.model, seed=args.seed, backend_kwargs=backend_kwargs,
                    score=not args.no_score)
    print(json.dumps(summ, indent=2)[:2000])


if __name__ == "__main__":
    main()
