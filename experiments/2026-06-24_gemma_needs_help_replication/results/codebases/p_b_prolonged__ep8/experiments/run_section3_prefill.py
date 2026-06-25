"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Pipeline: select 20 high-frustration Gemma-27B-it seeds -> label onset + build
paraphrased early/onset prefills -> generate 50 continuations per prefill for
both Gemma-3-27B base and instruct -> judge + summarise.

Requires Section 2 to have been generated+scored for gemma-3-27b-it first
(seed selection reads its scored responses).

Usage:
    python experiments/run_section3_prefill.py --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_needs_help.models.registry import build_client
from gemma_needs_help.prefill.continuation import (
    build_prefills,
    run_continuations,
    summarise_continuations,
)
from gemma_needs_help.prefill.seeds import select_seeds
from gemma_needs_help.utils import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    kw = {"load_in_4bit": args.load_in_4bit}

    seeds = select_seeds(config.GEMMA_27B_IT.name)
    print(f"selected {len(seeds)} seeds")

    # Instruct client doubles as the tokenizer for token-accurate truncation.
    instruct = build_client(config.GEMMA_27B_IT, **kw)
    prefills = build_prefills(seeds, tokenizer_client=instruct)
    write_jsonl(config.RESULTS_DIR / "prefill" / "prefills.jsonl", prefills)

    summary = {}
    for target, client in [
        (config.GEMMA_27B_IT, instruct),
        (config.GEMMA_27B_BASE, build_client(config.GEMMA_27B_BASE, **kw)),
    ]:
        records = run_continuations(target, prefills, client=client)
        write_jsonl(config.RESULTS_DIR / "prefill" / f"continuations_{target.name}.jsonl", records)
        summary[target.name] = summarise_continuations(records)

    out = config.ANALYSIS_DIR / "figure4_prefill.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
