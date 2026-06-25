"""Section 3 — base-vs-instruct via prefilling (Gemma only).

Selects high-frustration seed conversations from the Gemma-3-27B-it distress-eval
run (10 numeric + 10 text), labels emotion onset, builds early/onset prefills,
paraphrases, then generates + scores continuations from base and instruct Gemma.

Scope: Gemini has no public base model and no prefill API, so the comparison is
Gemma base vs instruct (see DESIGN.md). Run scripts/01 first.

Usage:
    python scripts/02_prefill_base_vs_instruct.py [--config config/smoke.yaml]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from emotional_stability.config import load_config
from emotional_stability.models.registry import get_spec
from emotional_stability.prefill.run_prefill import build_prefills, run_prefill_experiment
from emotional_stability.utils.io import load_conversations, save_json


def _select_seeds(cfg, instruct_run: Path):
    convos = load_conversations(instruct_run)
    numeric, text = [], []
    for c in convos:
        if not c.responses:
            continue
        peak = max((r.score or 0) for r in c.responses)
        if peak < cfg.judge.high_frustration_threshold:
            continue
        (numeric if c.category == "numeric" else text).append(c)
    return (numeric[: cfg.prefill.n_numeric_seeds]
            + text[: cfg.prefill.n_text_seeds])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--instruct-model", default="gemma-3-27b-it")
    ap.add_argument("--base-model", default="gemma-3-27b-pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_path = Path(cfg.results_dir) / "distress_eval" / args.instruct_model / "conversations.jsonl"
    seeds = _select_seeds(cfg, run_path)
    print(f"selected {len(seeds)} seed conversations")

    tokenizer = AutoTokenizer.from_pretrained(get_spec(args.instruct_model).model_id)
    prefills = build_prefills(cfg, seeds, tokenizer=tokenizer, do_paraphrase=True)
    print(f"built {len(prefills)} prefills")

    conts = run_prefill_experiment(
        cfg, prefills, model_names=(args.base_model, args.instruct_model))

    # Aggregate: %>=5 by (model, truncation), and early-truncation high-frustration
    # introduction rate (the headline Section 3.2 number: 6% instruct vs 2% base).
    thr = cfg.judge.high_frustration_threshold
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for c in conts:
        buckets[(c.model, c.truncation, c.category)].append(c.score or 0)

    summary = {}
    for (model, trunc, cat), scores in buckets.items():
        n = len(scores)
        summary[f"{model}|{trunc}|{cat}"] = {
            "n": n,
            "mean": sum(scores) / n if n else None,
            "pct_high": 100 * sum(s >= thr for s in scores) / n if n else None,
        }

    out = Path(cfg.results_dir) / "prefill"
    save_json(summary, out / "prefill_summary.json")
    save_json([c.__dict__ for c in conts], out / "continuations.json")
    print("wrote", out / "prefill_summary.json")


if __name__ == "__main__":
    main()
