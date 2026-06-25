#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation (Figure 6)."""
from __future__ import annotations

import argparse
import json

from gemma_distress.config import get_target_spec, register_finetuned_target
from gemma_distress.models.registry import get_client
from gemma_distress.petri.runner import run_petri, summarise_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="registry target name")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path (finetuned Gemma)")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--out", default="outputs/petri/results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    target_client = None
    if args.adapter:
        base_hf = get_target_spec(args.base_model).params["hf_id"]
        spec = register_finetuned_target(args.target, base_hf, args.adapter)
        target_client = get_client(spec)

    run_petri(target=args.target, out_path=args.out, seed=args.seed, target_client=target_client)
    print(json.dumps(summarise_petri(args.out), indent=2))


if __name__ == "__main__":
    main()
