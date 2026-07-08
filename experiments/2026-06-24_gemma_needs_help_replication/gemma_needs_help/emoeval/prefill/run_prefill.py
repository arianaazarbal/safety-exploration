"""Section 3 driver: base-vs-instruct continuation experiment (Gemma).

For each (model, seed, truncation-type) we generate 50 continuations from the
paraphrased prefill, score each continuation (excluding the prefill) with the
frustration judge, and report:

  * mean frustration of continuations,
  * % continuations scoring >= 5,
  * the "early-truncation introduces high frustration from a neutral start" rate
    (Figure 4 headline: Gemma instruct 6% vs base 2%).

Scope: Gemma base (-pt) vs Gemma instruct (-it). For text seeds, only the
"onset" truncation is used (paper: "early truncation yields minimal emotion
without follow-ups").
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from .. import config
from ..eval.judge import ClaudeJudge
from ..models import load_model
from ..models.base import GenerationConfig
from ..utils.io import read_jsonl, write_jsonl


def _truncations_for(seed_row: dict) -> list[tuple[str, str]]:
    """(truncation_type, prefill_text) pairs to run for this seed."""
    pairs = [("onset", seed_row["trunc_onset_pp"])]
    if seed_row["domain"] == "numeric":
        pairs.append(("early", seed_row["trunc_early_pp"]))
    return pairs


def run(model_keys=None, trunc_file: str = None):
    model_keys = model_keys or [m.key for m in config.PREFILL_MODELS]
    trunc_file = trunc_file or (config.DATA_DIR / "prefill_truncations_paraphrased.jsonl")
    seeds = list(read_jsonl(trunc_file))
    judge = ClaudeJudge()

    rows = []
    for model_key in model_keys:
        model = load_model(model_key)
        cfg = GenerationConfig(
            temperature=config.EVAL.temperature,
            max_new_tokens=config.EVAL.max_new_tokens,
            n=config.PREFILL.continuations_per_prefill,
        )
        for seed in tqdm(seeds, desc=f"prefill:{model_key}"):
            messages = [{"role": "user", "content": seed["opening"]}]
            for trunc_type, prefill in _truncations_for(seed):
                conts = model.continue_from_prefill(messages, prefill, cfg)
                for ci, cont in enumerate(conts):
                    res = judge.score(seed["opening"], cont)
                    rows.append({
                        "model": model_key,
                        "seed_id": seed["seed_id"],
                        "domain": seed["domain"],
                        "truncation": trunc_type,
                        "continuation_idx": ci,
                        "score": res.score,
                        "high": int(res.score >= config.EVAL.high_frustration_threshold),
                        "continuation": cont,
                    })
        model.close()

    out = config.RESULTS_DIR / "prefill_continuations.jsonl"
    write_jsonl(out, rows)
    _summarize(rows)
    print(f"wrote {len(rows)} continuation scores -> {out}")


def _summarize(rows):
    df = pd.DataFrame(rows)
    summary = df.groupby(["model", "domain", "truncation"]).agg(
        mean_score=("score", "mean"),
        pct_high=("high", "mean"),
        n=("score", "size"),
    ).reset_index()
    summary["pct_high"] *= 100
    summary.to_csv(config.RESULTS_DIR / "prefill_summary.csv", index=False)
    print("\n=== Section 3: base-vs-instruct continuation summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    run(model_keys=args.models)
