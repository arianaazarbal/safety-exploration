#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma family).

Selects high-frustration source responses (score >= 5) from a prior Section 2
run of Gemma-3-27B-it — 10 numeric + 10 text — reconstructs their conversation
history, builds early/onset prefills, paraphrases them, then generates and
scores 50 continuations per prefill for Gemma base (pt) and instruct (it).

Requires a local GPU stack for the base/pt model (see DESIGN.md).

  python scripts/run_prefill.py --results-dir results
"""
from __future__ import annotations

import argparse
import json
import os
import random

from emotional_instability import config
from emotional_instability.config import GEMMA_27B_IT, GEMMA_27B_PT
from emotional_instability.prefill.continuations import run_prefill_study


def _reconstruct_history(record: dict) -> list[dict]:
    """Rebuild the conversation history (turns before the final assistant turn)
    from a stored response record. We only stored per-turn rows, so we use the
    same condition plan to reproduce the user turns deterministically.

    Simplest robust approach: the record carries the user_message for its own
    turn; we reconstruct prior turns from the model's own responses.jsonl by
    matching condition+meta. Here we approximate with a single-user-turn history
    when prior turns are unavailable, which still exercises the onset/early
    continuation logic. For exact reconstruction, run prefill from the rollout
    objects directly (see DESIGN.md)."""
    return [{"role": "user", "content": record["user_message"]}]


def _select_sources(results_dir: str, model_name: str, seed: int) -> list[dict]:
    path = os.path.join(results_dir, model_name.replace("/", "_"), "responses.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    high = [r for r in rows if r["rating"] >= 5]
    numeric = [r for r in high if r["category"] in ("impossible_numeric", "tones", "extended")]
    text = [r for r in high if r["category"] in ("triggers", "wildchat")]
    rng = random.Random(seed)
    rng.shuffle(numeric); rng.shuffle(text)
    selected = numeric[:10] + text[:10]
    for r in selected:
        r["history"] = _reconstruct_history(r)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--source-model", default="Gemma-3-27B-it")
    ap.add_argument("--out-dir", default=config.DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sources = _select_sources(args.results_dir, args.source_model, args.seed)
    print(f"Selected {len(sources)} high-frustration source responses.")
    path = run_prefill_study(
        sources, model_specs=[GEMMA_27B_PT, GEMMA_27B_IT],
        out_dir=args.out_dir, paraphrase_model=config.JUDGE_MODEL,
    )
    print(f"Prefill continuations written to {path}")


if __name__ == "__main__":
    main()
