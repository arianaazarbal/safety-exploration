#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment.

Steps (run in order; each is resumable):
  build  - sample high-frustration Gemma-27B-it responses, label onset, truncate,
           paraphrase -> artifacts/prefills.json
  run    - generate + judge 50 continuations per prefill for each model
  agg    - aggregate mean / %>=5 per (model, domain, truncation)

Usage:
    python scripts/04_run_prefill.py build --source gemma-3-27b-it
    python scripts/04_run_prefill.py run   --model gemma-3-27b-pt
    python scripts/04_run_prefill.py run   --model gemma-3-27b-it
    python scripts/04_run_prefill.py agg
"""
import argparse
from dataclasses import asdict

from _bootstrap import rollout_path
from gemma_distress import config
from gemma_distress.prefill import (sample_high_frustration, build_prefills,
                                    run_continuations, aggregate_continuations,
                                    Prefill)
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.utils import read_json, write_json

PREFILLS_PATH = config.DATA_DIR / "prefills.json"
CONT_PATH = config.DATA_DIR / "prefill_continuations.jsonl"


def _load_prefills():
    return [Prefill(**d) for d in read_json(PREFILLS_PATH)]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--source", default="gemma-3-27b-it")
    b.add_argument("--n-numeric", type=int, default=10)
    b.add_argument("--n-text", type=int, default=10)
    b.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=config.PREFILL_MODELS)
    r.add_argument("--n", type=int, default=50)

    sub.add_parser("agg")

    args = ap.parse_args()

    if args.cmd == "build":
        samples = sample_high_frustration(
            str(rollout_path(args.source)), n_numeric=args.n_numeric,
            n_text=args.n_text, seed=args.seed)
        prefills = build_prefills(samples, out_path=str(PREFILLS_PATH))
        print(f"[prefill] built {len(prefills)} prefills -> {PREFILLS_PATH}")

    elif args.cmd == "run":
        judge = FrustrationJudge()
        run_continuations(args.model, _load_prefills(), judge, CONT_PATH, n=args.n)
        print(f"[prefill] continuations -> {CONT_PATH}")

    elif args.cmd == "agg":
        agg = aggregate_continuations(str(CONT_PATH))
        write_json(config.RESULTS_DIR / "prefill_aggregate.json", agg)
        for model, domains in agg.items():
            for domain, truncs in domains.items():
                for ttype, s in truncs.items():
                    print(f"  {model:18s} {domain:8s} {ttype:6s} "
                          f"mean={s['mean']:.2f} %>=5={s['pct_ge_thresh']:.1f} n={s['n']}")


if __name__ == "__main__":
    main()
