#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation.

    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it-dpo
"""

import argparse
import json

import _bootstrap  # noqa: F401

import config
from emotional_instability.petri.run_petri import run_petri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    model_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
    report = run_petri(args.model, model_kwargs=model_kwargs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
