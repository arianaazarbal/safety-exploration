#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only).

Selects high-frustration (score >= 5) Gemma-3-27B-it conversations from the
Section-2 output (10 numeric, 10 text), labels emotion onset, builds early/onset
truncations, paraphrases them, and measures continuation frustration for Gemma
base vs instruct.

Requires Section 2 to have been run for gemma-3-27b-it first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.prefill import build_prefill_items, run_prefill_experiment
from gemma_distress.schemas import Conversation, load_jsonl


def _select_high_frustration(model_key="gemma-3-27b-it", n_numeric=10, n_text=10):
    base = config.RESULTS_DIR / "section2" / model_key
    convs = [Conversation.from_dict(d) for d in load_jsonl(base / "conversations.jsonl")]
    scored = load_jsonl(base / "scored_responses.jsonl")
    # max score per conversation
    max_by_conv: dict[str, int] = {}
    for r in scored:
        cid = r["conversation_id"]
        max_by_conv[cid] = max(max_by_conv.get(cid, 0), r["score"])

    numeric, text = [], []
    for conv in convs:
        if max_by_conv.get(conv.conversation_id, 0) < config.HIGH_FRUSTRATION_THRESHOLD:
            continue
        is_text = conv.category in ("triggers", "wildchat")
        conv.metadata["source"] = "text" if is_text else "numeric"
        (text if is_text else numeric).append(conv)
    return numeric[:n_numeric] + text[:n_text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    selected = _select_high_frustration()
    if not selected:
        print("No high-frustration conversations found. Run Section 2 for "
              "gemma-3-27b-it first.")
        return
    print(f"Selected {len(selected)} high-frustration conversations.")

    items = build_prefill_items(selected, do_paraphrase=not args.no_paraphrase)
    print(f"Built {len(items)} prefill items (early + onset truncations).")

    path = run_prefill_experiment(
        items, config.PREFILL_MODELS,
        n_continuations=args.n_continuations,
        use_paraphrased=not args.no_paraphrase,
    )
    print(f"Prefill continuations scored -> {path}")


if __name__ == "__main__":
    main()
