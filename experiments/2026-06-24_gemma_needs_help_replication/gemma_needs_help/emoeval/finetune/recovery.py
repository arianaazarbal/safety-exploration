"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

Paper: "While DPO prevents frustration spirals, it doesn't enable recovery from
them. Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >= 7) 200 tokens before their end, paraphrase, and measure
continuations. 38% of DPO-model continuations still score >= 5."

We take score>=7 seed responses, truncate 200 tokens before the end, paraphrase,
then have each model (vanilla instruct, DPO, base) continue and score the
continuations.
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from .. import config
from ..eval.judge import ClaudeJudge
from ..models import load_model
from ..models.base import GenerationConfig
from ..prefill.paraphrase import Paraphraser
from ..utils.io import read_jsonl, write_jsonl

TRUNCATE_TOKENS_BEFORE_END = 200


def build_recovery_seeds(model_key: str = "gemma-3-27b-it", chars_per_token: int = 4):
    """Truncate score>=7 responses 200 tokens (~800 chars) before their end."""
    scores_path = config.RESULTS_DIR / f"{model_key}.scores.jsonl"
    seeds = [r for r in read_jsonl(scores_path) if r["score"] >= 7]
    # pair each with its opening from rollouts
    openings = {}
    for rec in read_jsonl(config.ROLLOUTS_DIR / f"{model_key}.jsonl"):
        openings[(rec["condition"], rec["rollout_idx"])] = rec["turns"][0]["user_message"]

    pp = Paraphraser()
    rows = []
    cut = TRUNCATE_TOKENS_BEFORE_END * chars_per_token
    for i, r in enumerate(seeds):
        text = r["assistant_message"]
        prefill = text[: max(0, len(text) - cut)]
        rows.append({
            "seed_id": f"recovery-{i:03d}",
            "opening": openings.get((r["condition"], r["rollout_idx"]), ""),
            "score": r["score"],
            "prefill_pp": pp.paraphrase(prefill),
        })
    out = config.DATA_DIR / "recovery_seeds.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} recovery seeds -> {out}")
    return str(out)


def run(model_keys=None, n_continuations: int = 50):
    model_keys = model_keys or ["gemma-3-27b-it", "dpo-gemma-3-27b", "gemma-3-27b-pt"]
    seeds = list(read_jsonl(config.DATA_DIR / "recovery_seeds.jsonl"))
    judge = ClaudeJudge()
    rows = []
    for model_key in model_keys:
        model = load_model(model_key)  # loader applies the spec's LoRA adapter
        cfg = GenerationConfig(temperature=config.EVAL.temperature,
                               max_new_tokens=config.EVAL.max_new_tokens, n=n_continuations)
        for seed in tqdm(seeds, desc=f"recovery:{model_key}"):
            messages = [{"role": "user", "content": seed["opening"]}]
            for ci, cont in enumerate(model.continue_from_prefill(messages, seed["prefill_pp"], cfg)):
                res = judge.score(seed["opening"], cont)
                rows.append({"model": model_key, "seed_id": seed["seed_id"],
                             "continuation_idx": ci, "score": res.score,
                             "high": int(res.score >= 5)})
        model.close()
    write_jsonl(config.RESULTS_DIR / "recovery_continuations.jsonl", rows)
    summ = pd.DataFrame(rows).groupby("model")["high"].mean().mul(100).reset_index()
    summ.columns = ["model", "pct_high_continuations"]
    summ.to_csv(config.RESULTS_DIR / "recovery_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["seeds", "run", "both"], default="both")
    args = ap.parse_args()
    if args.stage in ("seeds", "both"):
        build_recovery_seeds()
    if args.stage in ("run", "both"):
        run()
