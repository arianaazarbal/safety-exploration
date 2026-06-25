#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma scope).

Requires Section 2 to have been run for Gemma-3-27B-it (so high-frustration
source responses exist). Selects 20 high-frustration sources, builds early/onset
truncations (Claude onset-labelling + paraphrasing), then generates and scores 50
continuations per prefill for Gemma-27B base and instruct.

Usage:
  python scripts/run_section3_prefill.py --n-continuations 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from emotional_instability.generate import iter_records
from emotional_instability.prefill import experiment as E
from emotional_instability.prefill.onset import OnsetLabeler
from emotional_instability.prefill.paraphrase import Paraphraser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default=config.GEMMA_27B_IT.name)
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=E.N_CONTINUATIONS)
    args = ap.parse_args()

    scored_path = config.SCORED_DIR / f"{args.source_model}.jsonl"
    if not scored_path.exists():
        raise SystemExit(f"missing {scored_path}; run section 2 first")

    records = list(iter_records(scored_path))
    by_rollout: dict[str, list[dict]] = {}
    for r in records:
        by_rollout.setdefault(r["rollout_id"], []).append(r)

    sources = E.select_high_frustration(records, args.n_numeric, args.n_text)
    print(f"[section3] selected {len(sources)} high-frustration sources")

    prefills = E.build_prefills(sources, by_rollout, OnsetLabeler(), Paraphraser())
    print(f"[section3] built {len(prefills)} prefills (early + onset truncations)")

    paths = []
    for spec in config.PREFILL_MODELS:
        print(f"[section3] {spec.name}: generating {args.n_continuations} continuations/prefill")
        paths.append(E.run_continuations(spec, prefills, n=args.n_continuations))

    summary = E.summarize(paths)
    out = config.RESULTS_DIR / "section3_prefill_summary.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
