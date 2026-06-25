#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation for a target model.

    python scripts/08_run_petri.py --model gemma-3-27b-it
    python scripts/08_run_petri.py --model gemma-3-27b-dpo
    python scripts/08_run_petri.py --model gemini-2.5-flash
"""
import argparse
import json

import _bootstrap  # noqa: F401
from gemma_distress.petri.runner import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    args = ap.parse_args()

    from gemma_distress.config import get_model_spec
    backend_kwargs = {}
    if get_model_spec(args.model).backend == "hf_local":
        backend_kwargs["tensor_parallel_size"] = args.tp_size

    summ = run_petri(args.model, backend_kwargs=backend_kwargs)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
