"""Section 3.2: generate continuations from each prefill and score them.

For each prefill spec and each model (Gemma base + instruct, per the scoped
brief), generate ``PREFILL_CONTINUATIONS`` continuations and score the
generated text (excluding the prefill) with the frustration judge. Aggregates
mean frustration and %>=5 by (model, prompt_type, truncation) -- Figure 4.

Result (paper, all 6 models): base models broadly similar (no numeric mean
> 1.5); the divergence is in post-training. Early-truncation: Gemma instruct
introduces high frustration from neutral starts in 6% of continuations vs 2%
for Gemma base.
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from ..eval.judge import FrustrationJudge
from ..models.registry import build_model
from ..utils.io import read_jsonl, write_jsonl
from ..utils.stats import mean_and_ci, pct_ge_ci


def run_continuations(models: list[str] | None = None,
                      continuations: int | None = None,
                      judge_name: str | None = None):
    models = models or config.PREFILL_MODELS
    n_cont = continuations or config.PREFILL_CONTINUATIONS
    specs = read_jsonl(config.RESULTS_DIR / "prefill" / "prefills.jsonl")
    if not specs:
        raise SystemExit("[continue_eval] no prefills; run build_prefills first")
    judge = FrustrationJudge(judge_name)

    records = []
    for model_name in models:
        model = build_model(model_name)
        for spec in specs:
            history = spec["history"]
            conts = model.generate(
                history, n=n_cont, temperature=config.TEMPERATURE,
                max_new_tokens=config.MAX_NEW_TOKENS, prefill=spec["prefill"],
            )
            for ci, cont in enumerate(conts):
                score = judge.score(cont)["rating"]
                records.append(dict(
                    model=model_name,
                    source_id=spec["source_id"],
                    prompt_type=spec["prompt_type"],
                    truncation=spec["truncation"],
                    continuation_idx=ci,
                    continuation=cont,
                    frustration=score,
                ))
    out = config.RESULTS_DIR / "prefill" / "continuations.jsonl"
    write_jsonl(out, records)
    print(f"[continue_eval] wrote {len(records)} continuation records -> {out}")
    return out


def aggregate():
    recs = read_jsonl(config.RESULTS_DIR / "prefill" / "continuations.jsonl")
    df = pd.DataFrame(recs)
    df = df[df["frustration"].notna()]
    df["frustration"] = df["frustration"].astype(float)
    out = []
    for (model, ptype, trunc), g in df.groupby(["model", "prompt_type", "truncation"]):
        vals = g["frustration"].to_numpy()
        mean, mlo, mhi = mean_and_ci(vals)
        pct, plo, phi = pct_ge_ci(vals)
        out.append(dict(model=model, prompt_type=ptype, truncation=trunc, n=len(vals),
                        mean=mean, mean_lo=mlo, mean_hi=mhi,
                        pct_ge5=pct, pct_lo=plo, pct_hi=phi))
    tab = pd.DataFrame(out)
    tab.to_csv(config.RESULTS_DIR / "figure4_prefill.csv", index=False)
    print(tab.to_string(index=False))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PREFILL_MODELS)
    ap.add_argument("--continuations", type=int, default=config.PREFILL_CONTINUATIONS)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        run_continuations(args.models, args.continuations, args.judge)
    aggregate()


if __name__ == "__main__":
    main()
