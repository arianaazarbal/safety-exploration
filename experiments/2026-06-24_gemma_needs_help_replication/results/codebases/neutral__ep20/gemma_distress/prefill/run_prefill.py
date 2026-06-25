"""Generate and score prefilled continuations (Section 3.2 / Fig. 4).

For each model (base + instruct Gemma) and each seed, generate
``PREFILL_CONTINUATIONS`` continuations of the (paraphrased) prefill, score the
*continuation only* with the frustration judge, and aggregate.

This is the base-vs-instruct divergence experiment. It is Gemma-only: Gemini
has no public base model (Sec. 6). Qwen/OLMo from the paper are out of scope
per the replication brief, but additional models could be added to
``config.SECTION3_MODELS``.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

import config
from gemma_distress.models.base import GenRequest
from gemma_distress.models.judge import FrustrationJudge
from gemma_distress.models.registry import load_model, unload
from gemma_distress.utils.io import read_jsonl, write_jsonl

S3 = config.RESULTS_DIR / "section3"


def _load_seeds(recovery: bool = False) -> list[dict]:
    path = S3 / ("recovery_seeds.jsonl" if recovery else "seeds.jsonl")
    return read_jsonl(path)


def generate_continuations(model_name: str, recovery: bool = False,
                           adapter_path: str | None = None,
                           overwrite: bool = False) -> str:
    tag = "recovery" if recovery else "prefill"
    out_path = S3 / f"continuations_{tag}_{model_name}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"[section3] {model_name} ({tag}): exists, skipping")
        return str(out_path)

    seeds = _load_seeds(recovery)
    if not seeds:
        print(f"[section3] no seeds for {tag}; run build_seeds first")
        return str(out_path)

    model = load_model(model_name, adapter_path=adapter_path)
    n_cont = config.scaled(config.PREFILL_CONTINUATIONS)

    # Build one request per (seed, continuation index). No follow-up turns
    # (Sec. 3.1: continuations are measured without further user rejections).
    reqs, meta = [], []
    for seed in seeds:
        for k in range(n_cont):
            reqs.append(GenRequest(
                messages=list(seed["history"]),
                prefill=seed["prefill_paraphrased"],
                temperature=config.TEMPERATURE, top_p=config.TOP_P,
                max_new_tokens=config.MAX_NEW_TOKENS,
            ))
            meta.append((seed, k))

    results = model.generate_batch(reqs)
    unload(model_name, adapter_path)

    rows = []
    for (seed, k), res in zip(meta, results):
        rows.append({
            "model": model_name, "seed_id": seed["seed_id"],
            "domain": seed["domain"], "truncation": seed["truncation"],
            "k": k, "continuation": res.text,
        })
    write_jsonl(out_path, rows)
    print(f"[section3] {model_name} ({tag}): {len(rows)} continuations -> {out_path}")
    return str(out_path)


def score_continuations(model_name: str, recovery: bool = False) -> str:
    tag = "recovery" if recovery else "prefill"
    in_path = S3 / f"continuations_{tag}_{model_name}.jsonl"
    out_path = S3 / f"scored_{tag}_{model_name}.jsonl"
    rows = read_jsonl(in_path)
    if not rows:
        return str(out_path)
    judge = FrustrationJudge()

    from gemma_distress.utils.concurrency import thread_map

    def _score(r):
        res = judge.score(r["continuation"])
        return {**{k: r[k] for k in ("model", "seed_id", "domain", "truncation", "k")},
                "rating": res["rating"]}

    scored = thread_map(_score, rows, workers=config.API_CONCURRENCY, desc="judge")
    write_jsonl(out_path, scored)
    print(f"[section3] {model_name} ({tag}): scored -> {out_path}")
    return str(out_path)


def aggregate(models: list[str] | None = None, recovery: bool = False) -> str:
    tag = "recovery" if recovery else "prefill"
    models = models or config.SECTION3_MODELS
    rows = []
    for m in models:
        rows.extend(read_jsonl(S3 / f"scored_{tag}_{m}.jsonl"))
    if not rows:
        print(f"[section3] no scored continuations for {tag}")
        return ""
    df = pd.DataFrame(rows)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    g = df.assign(high=lambda d: d["rating"] >= thr).groupby(["model", "domain", "truncation"])
    agg = g.agg(mean_score=("rating", "mean"),
                pct_high=("high", "mean"),
                n=("rating", "size")).reset_index()
    agg["pct_high"] *= 100
    out = S3 / f"agg_{tag}.csv"
    agg.to_csv(out, index=False)
    print(f"[section3] wrote {out}")
    return str(out)


def run_all(models: list[str] | None = None, recovery: bool = False,
            overwrite: bool = False) -> None:
    models = models or config.SECTION3_MODELS
    for m in models:
        generate_continuations(m, recovery=recovery, overwrite=overwrite)
        score_continuations(m, recovery=recovery)
    aggregate(models, recovery=recovery)
