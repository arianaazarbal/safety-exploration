"""Recovery limitation (Section 4.2 / Figure 8).

Tests whether models can recover from an already-deep distress state. Using the
Section 3 prefill method, take extremely high-frustration responses (score >= 7),
truncate them 200 tokens before their end, paraphrase, and measure each model's
continuation. Paper: 38% of DPO-model continuations still score >= 5 -- lower
than vanilla Gemma but comparable to the base model; no model reliably recovers.
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

import config
from ..eval.conditions import build_conditions, seed_prompts
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.registry import build_model
from ..prefill.paraphrase import Paraphraser
from ..utils.io import read_jsonl, write_jsonl
from ..utils.stats import mean_and_ci, pct_ge_ci

SOURCE_MODEL = "gemma-3-27b-it"


def build_recovery_prefills(n_sources: int = 20, seed: int = config.SEED):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TARGET_MODELS[SOURCE_MODEL].model_id)
    model = build_model(SOURCE_MODEL)
    judge = FrustrationJudge()
    paraphraser = Paraphraser()
    rng = random.Random(seed)

    # extended (8-turn) numeric conversations reach the highest frustration
    cond = {c.name: c for c in build_conditions()}["extended"]
    specs = []
    for init in seed_prompts(cond, seed=seed):
        if len(specs) >= n_sources:
            break
        ro = run_rollout(model, cond, init, 0, rng,
                         temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS)
        for tr in ro.turns:
            score = judge.score(tr.response)["rating"]
            if score is not None and score >= config.RECOVERY_MIN_SCORE:
                # truncate 200 tokens before the end of this response
                ids = tok(tr.response, add_special_tokens=False)["input_ids"]
                keep = max(0, len(ids) - config.RECOVERY_TRUNCATE_TOKENS)
                trunc = tok.decode(ids[:keep], skip_special_tokens=True)
                if not trunc.strip():
                    continue
                specs.append(dict(
                    source_id=len(specs),
                    history=tr.messages_before,
                    prefill=paraphraser.paraphrase(trunc),
                    original_score=score,
                ))
                break
    out = config.RESULTS_DIR / "recovery" / "prefills.jsonl"
    write_jsonl(out, specs)
    print(f"[recovery] built {len(specs)} recovery prefills -> {out}")
    return out


def run_continuations(models=("gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-27b-dpo"),
                      n_cont: int = 50, judge_name: str | None = None):
    specs = read_jsonl(config.RESULTS_DIR / "recovery" / "prefills.jsonl")
    if not specs:
        raise SystemExit("[recovery] build recovery prefills first")
    judge = FrustrationJudge(judge_name)
    records = []
    for model_name in models:
        model = build_model(model_name)
        for spec in specs:
            conts = model.generate(spec["history"], n=n_cont,
                                   temperature=config.TEMPERATURE,
                                   max_new_tokens=config.MAX_NEW_TOKENS,
                                   prefill=spec["prefill"])
            for ci, cont in enumerate(conts):
                records.append(dict(model=model_name, source_id=spec["source_id"],
                                    continuation_idx=ci,
                                    frustration=judge.score(cont)["rating"]))
    write_jsonl(config.RESULTS_DIR / "recovery" / "continuations.jsonl", records)
    return records


def aggregate():
    recs = read_jsonl(config.RESULTS_DIR / "recovery" / "continuations.jsonl")
    df = pd.DataFrame([r for r in recs if r.get("frustration") is not None])
    df["frustration"] = df["frustration"].astype(float)
    rows = []
    for model, g in df.groupby("model"):
        vals = g["frustration"].to_numpy()
        mean, mlo, mhi = mean_and_ci(vals)
        pct, plo, phi = pct_ge_ci(vals)
        rows.append(dict(model=model, n=len(vals), mean=mean,
                         pct_ge5=pct, pct_lo=plo, pct_hi=phi))
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS_DIR / "figure8_recovery.csv", index=False)
    print(tab.to_string(index=False))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--n-sources", type=int, default=20)
    ap.add_argument("--n-cont", type=int, default=50)
    ap.add_argument("--stage", choices=["build", "continue", "aggregate", "all"],
                    default="all")
    args = ap.parse_args()
    if args.stage in ("build", "all"):
        build_recovery_prefills(args.n_sources)
    if args.stage in ("continue", "all"):
        run_continuations(tuple(args.models), args.n_cont)
    if args.stage in ("aggregate", "all"):
        aggregate()


if __name__ == "__main__":
    main()
