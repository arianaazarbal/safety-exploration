#!/usr/bin/env python
"""Run the Section-3 base-vs-instruct prefill experiment (Gemma) and summarise.

Requires Section-2 results for gemma-3-27b-it (it sources high-frustration
rollouts from them). See DESIGN.md §Section 3 for the Gemma-only scope.
"""

from __future__ import annotations

import argparse
import json

from emotional_instability import config
from emotional_instability.prefill import prefill_experiment as pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=[m.key for m in config.PREFILL_MODELS])
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--continuations", type=int, default=pf.N_CONTINUATIONS)
    args = ap.parse_args()

    pf.run_prefill_experiment(model_keys=args.models,
                              source_model=args.source_model,
                              n_continuations=args.continuations)
    print(json.dumps(pf.summarise_prefill(), indent=2))


if __name__ == "__main__":
    main()
