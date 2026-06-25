#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation for one or more models."""
import argparse

from gemma_distress import config
from gemma_distress.models import build_client
from gemma_distress.petri import PetriEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=list(config.GEMMA_INSTRUCT) + list(config.GEMINI_MODELS))
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    evaluator = PetriEvaluator()
    for model_key in args.models:
        client = build_client(model_key, adapter_path=args.adapter)
        path = evaluator.run_model(model_key, client)
        print(f"[petri] {model_key} -> {path}")


if __name__ == "__main__":
    main()
