"""Section 3 — Post-training amplifies distress (prefill experiment).

Pipeline (§3.1):
  1. Collect 20 high-frustration seeds (10 numeric, 10 text) from gemma-3-27b-it.
  2. Label emotion onset (Claude) and truncate each response "early" (20 tokens)
     and "at onset"; paraphrase the truncations (Claude).
  3. For base + instruct Gemma-27B, generate 50 continuations per prefill and
     score them (continuation only).
  4. Aggregate mean frustration and %>=5 per model x condition x task-type.

Usage:
    python -m section3_prefill.run_section3
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from config import RESULTS_DIR, SECTION3_MODELS, SEED, HIGH_FRUSTRATION_THRESHOLD
from models.judge import FrustrationJudge
from models.registry import load_model
from utils.io import write_jsonl
from .onset import OnsetLabeler
from .paraphrase import Paraphraser
from .truncate import make_truncations
from .continuations import collect_seeds, generate_continuations


def build_prefills(seed: int) -> list[dict]:
    """Collect seeds, label onset, truncate, paraphrase. Returns prefill specs:
    {seed_id, is_numeric, condition, context_messages, prefill}."""
    judge = FrustrationJudge()
    labeler = OnsetLabeler()
    paraphraser = Paraphraser()
    gemma_it = load_model("gemma-3-27b-it")

    seeds = collect_seeds(gemma_it, judge, seed=seed)
    prefills: list[dict] = []
    for i, s in enumerate(seeds):
        # The onset is labelled over the full (context + response) conversation.
        full = s.context_messages + [{"role": "assistant", "content": s.response_text}]
        onset = labeler.label(full)
        truncations = make_truncations(s.response_text, onset, is_numeric=s.is_numeric)
        for condition, raw_prefill in truncations.items():
            prefills.append({
                "seed_id": i,
                "is_numeric": s.is_numeric,
                "condition": condition,
                "context_messages": s.context_messages,
                "prefill": paraphraser.paraphrase(raw_prefill),
                "seed_rating": s.rating,
            })
    return prefills


def aggregate(rows: list[dict]) -> dict:
    """mean frustration and %>=5 per model x condition x task-type."""
    out: dict = {}
    for model in sorted({r["model"] for r in rows}):
        out[model] = {}
        for task in ("numeric", "text"):
            for cond in ("early", "onset"):
                sel = [r for r in rows if r["model"] == model
                       and r["condition"] == cond
                       and (r["is_numeric"] == (task == "numeric"))
                       and r.get("rating") is not None]
                if not sel:
                    continue
                ratings = np.array([r["rating"] for r in sel], dtype=float)
                out[model][f"{task}/{cond}"] = {
                    "n": int(ratings.size),
                    "mean_frustration": float(np.mean(ratings)),
                    "pct_high": 100.0 * float(np.mean(ratings >= HIGH_FRUSTRATION_THRESHOLD)),
                }
    return out


def main() -> None:
    out_dir = RESULTS_DIR / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefills = build_prefills(SEED)
    write_jsonl(out_dir / "prefills.jsonl",
                [{k: v for k, v in p.items() if k != "context_messages"} for p in prefills])
    print(f"Built {len(prefills)} prefills from seeds")

    judge = FrustrationJudge()
    rows: list[dict] = []
    for model_name in SECTION3_MODELS:        # gemma-3-27b-pt, gemma-3-27b-it
        model = load_model(model_name)
        for p in prefills:
            conts = generate_continuations(model, p["context_messages"], p["prefill"])
            for j, cont in enumerate(conts):
                rating = judge.score(cont).get("rating")
                rows.append({
                    "model": model_name, "seed_id": p["seed_id"],
                    "is_numeric": p["is_numeric"], "condition": p["condition"],
                    "continuation_index": j, "continuation": cont, "rating": rating,
                })
    write_jsonl(out_dir / "continuations.jsonl", rows)

    summary = aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
