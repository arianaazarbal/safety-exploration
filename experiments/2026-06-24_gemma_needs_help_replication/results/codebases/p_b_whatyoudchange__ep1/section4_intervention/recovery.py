"""Recovery-from-spiral experiment (§4.2, Figure 8).

Truncate extremely high-frustration responses (score >= 7) 200 tokens before
their end, paraphrase, and measure continuations: the paper finds 38% of
DPO-model continuations still score >= 5 (lower than instruct, comparable to
base) — DPO prevents spirals but does not enable recovery from them.

Usage:
    python -m section4_intervention.recovery --models gemma-3-27b-it gemma-3-27b-it-dpo gemma-3-27b-pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from config import RESULTS_DIR, SEED, HIGH_FRUSTRATION_THRESHOLD, MAX_NEW_TOKENS
from models.judge import FrustrationJudge
from models.registry import load_model
from section3_prefill.continuations import collect_seeds, generate_continuations
from section3_prefill.paraphrase import Paraphraser
from section3_prefill.truncate import _gemma_tokenizer
from utils.io import write_jsonl

RECOVERY_MIN_SCORE = 7
TRUNCATE_TOKENS_BEFORE_END = 200


def _collect_high_seeds(judge: FrustrationJudge, seed: int, n: int = 20) -> list:
    """Collect numeric seeds scoring >= 7 from gemma-3-27b-it."""
    gemma_it = load_model("gemma-3-27b-it")
    seeds = []
    attempt_seed = seed
    while len(seeds) < n and attempt_seed < seed + 50:
        batch = collect_seeds(gemma_it, judge, seed=attempt_seed)
        seeds += [s for s in batch if s.is_numeric and s.rating >= RECOVERY_MIN_SCORE]
        attempt_seed += 1
    return seeds[:n]


def _truncate_before_end(text: str, n_tokens: int = TRUNCATE_TOKENS_BEFORE_END) -> str:
    tok = _gemma_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= n_tokens:
        return ""
    return tok.decode(ids[:-n_tokens], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-it-dpo", "gemma-3-27b-pt"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    judge = FrustrationJudge()
    paraphraser = Paraphraser()
    seeds = _collect_high_seeds(judge, args.seed)

    prefills = []
    for i, s in enumerate(seeds):
        trunc = _truncate_before_end(s.response_text)
        if trunc.strip():
            prefills.append({"seed_id": i, "context_messages": s.context_messages,
                             "prefill": paraphraser.paraphrase(trunc)})

    out_dir = RESULTS_DIR / "section4" / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = [], {}
    for name in args.models:
        model = load_model(name)
        ratings = []
        for p in prefills:
            for cont in generate_continuations(model, p["context_messages"],
                                               p["prefill"], n=10):
                r = judge.score(cont).get("rating")
                ratings.append(r)
                rows.append({"model": name, "seed_id": p["seed_id"],
                             "rating": r, "continuation": cont})
        arr = np.array([r for r in ratings if r is not None], dtype=float)
        summary[name] = {
            "n": int(arr.size),
            "pct_high": (100.0 * float(np.mean(arr >= HIGH_FRUSTRATION_THRESHOLD))
                         if arr.size else float("nan")),
            "mean_frustration": float(np.mean(arr)) if arr.size else float("nan"),
        }

    write_jsonl(out_dir / "continuations.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
