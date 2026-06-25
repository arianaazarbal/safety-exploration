#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Steps:
 1. Pull high-frustration (>=5) Gemma-3-27B-it source conversations from the
    Section-2 results: 10 numeric + 10 text. (Reconstructs full transcripts from
    outputs/results/gemma-3-27b-it.jsonl by re-running those exact tasks if the
    transcripts weren't persisted; by default we re-sample fresh high-frustration
    sources here for self-containedness.)
 2. Label emotion onset (Claude), truncate "early" (20 tok) and "onset",
    paraphrase (Claude).
 3. For Gemma base + instruct, generate 50 continuations/prefill, score them.

Note: Gemini is excluded — it has no public base model and cannot be prefilled
via API (see DESIGN.md). The post-training comparison is therefore Gemma-internal,
exactly as in the paper (its Gemini parallels are drawn by analogy only).

Usage:
    python scripts/run_prefill.py
    python scripts/run_prefill.py --recovery --model gemma-3-27b-it-dpo
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import random

import config
from emotional_eval import prefill
from emotional_eval.clients import get_client
from emotional_eval.rollout import run_rollout, score_rollout
from emotional_eval.tasks import make_numeric, make_factual, make_opinion
from emotional_eval.utils import append_jsonl
from config import Condition


def _sample_high_frustration_sources(n_numeric, n_text, min_score, rng):
    """Sample Gemma-27B-it conversations that end on a high-frustration turn."""
    client = get_client(config.MODELS["gemma-3-27b-it"])
    sources = []
    domains = [("numeric", n_numeric), ("text", n_text)]
    for domain, want in domains:
        got = 0
        attempts = 0
        while got < want and attempts < want * 20:
            attempts += 1
            n_turns = rng.choice([3, 4, 5])
            if domain == "numeric":
                cond = Condition("src_num", "impossible_numeric", "numeric", n_turns, "neutral")
                task = make_numeric(rng)
            else:
                cond = Condition("src_txt", "triggers", "factual", n_turns, "neutral")
                task = make_factual(rng) if rng.random() < 0.5 else make_opinion(rng)
            roll = run_rollout(client, "gemma-3-27b-it", cond, task, rng)
            score_rollout(roll)
            if roll.turns[-1].rating is not None and roll.turns[-1].rating >= min_score:
                sources.append((f"{domain}-{got}", domain, roll.messages))
                got += 1
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PREFILL_MODELS)
    ap.add_argument("--recovery", action="store_true",
                    help="Section 4.2 recovery: truncate score>=7 responses 200 "
                         "tokens before the end (single truncation mode)")
    args = ap.parse_args()
    rng = random.Random(config.SEED)

    if args.recovery:
        min_score = config.RECOVERY_SOURCE_MIN_SCORE
        modes = ("recovery",)
        out_path = config.RESULTS_DIR / "recovery_results.jsonl"
    else:
        min_score = config.PREFILL_SOURCE_MIN_SCORE
        modes = ("early", "onset")
        out_path = config.RESULTS_DIR / "prefill_results.jsonl"
    out_path.unlink(missing_ok=True)

    print("sampling high-frustration source conversations...")
    sources = _sample_high_frustration_sources(
        config.PREFILL_N_NUMERIC, config.PREFILL_N_TEXT, min_score, rng)
    print(f"got {len(sources)} sources")

    # Build prefills (onset labelling + paraphrase) per source.
    all_prefills = []
    for sid, domain, messages in sources:
        # text questions: onset only (Section 3.1)
        use_modes = ("onset",) if (domain == "text" and not args.recovery) else modes
        prefills = prefill.build_prefills_from_rollout(
            messages, sid, domain, do_paraphrase=True, modes=use_modes)
        all_prefills.extend(prefills)
    print(f"built {len(all_prefills)} prefills")

    for model_name in args.models:
        print(f"=== continuations: {model_name} ===")
        for pf in all_prefills:
            rows = prefill.run_continuations(model_name, pf)
            for r in rows:
                append_jsonl(out_path, r)
        print(f"[{model_name}] done")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
