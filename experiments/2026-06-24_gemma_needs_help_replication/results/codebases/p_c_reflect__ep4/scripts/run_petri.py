#!/usr/bin/env python
"""Section 4: open-ended emotion elicitation via the Petri-style harness.

    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it --adapter results/training/adapters/dpo_all_layers
    python scripts/run_petri.py --model gemini-2.5-flash
"""

import argparse
import json

from gemma_distress.petri.runner import run_petri


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--n", type=int, default=10, help="transcripts per emotion")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--openrouter", action="store_true")
    args = p.parse_args()

    summary = run_petri(
        args.model, adapter_path=args.adapter, n_transcripts=args.n,
        max_turns=args.max_turns, use_openrouter=args.openrouter,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
