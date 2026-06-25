#!/usr/bin/env python
"""Appendix I.2: logit-based internal-emotion probing, vanilla vs DPO.

Probes high-frustration conversations (drawn from a scored gemma-3-27b-it eval
run) and compares internal Ekman-emotion trajectories before vs after DPO.

    python scripts/10_run_probing.py --dpo-adapter outputs/training/dpo/final
"""
import argparse
import json

import _bootstrap  # noqa: F401
from gemma_distress.config import get_model_spec
from gemma_distress.eval.runner import load_records
from gemma_distress.probing.internal_emotions import compare_models
from gemma_distress.prompts.tasks import load_wildchat


def _render(messages):
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it",
                    help="model whose high-frustration convos we probe")
    ap.add_argument("--dpo-adapter", default="outputs/training/dpo/final")
    ap.add_argument("--n-convos", type=int, default=12)
    ap.add_argument("--n-wildchat", type=int, default=500)
    args = ap.parse_args()

    spec = get_model_spec(args.source)
    # High-frustration conversations to probe (score >= 7, like the paper's I.2).
    recs = [r for r in load_records(args.source)
            if r.rating is not None and r.rating >= 7]
    convos = [_render(r.messages) for r in recs[: args.n_convos]]

    # WildChat baseline texts for z-score normalization.
    wc = [t.prompt for t in load_wildchat(args.n_wildchat)]

    results = compare_models(spec.hf_id, convos, wc, dpo_adapter=args.dpo_adapter)
    print(json.dumps({k: len(v) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
