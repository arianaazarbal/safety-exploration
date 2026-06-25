"""Section 3 driver: generate and score continuations from prefills.

For each model (Gemma base vs instruct, in scope), each prefill, we generate
``continuations_per_prefill`` continuations and score the continuation only
(excluding the prefill). Aggregation reports mean frustration and % >= 5 split by
truncation type (early/onset) and task type (numeric/text), reproducing the
Section 3.2 comparison: instruct introduces high frustration from neutral
("early") starts much more often than base.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import build_model
from ..utils.io import load_jsonl, write_json, write_jsonl
from ..utils.stats import frac_at_least, mean
from .paraphrase import Paraphraser
from .onset_label import OnsetLabeler
from .sample_high_frustration import collect_seeds
from .truncate import Prefill, build_prefills_for_seed


def build_all_prefills(seeds, *, model_kwargs: dict | None = None) -> list[Prefill]:
    """Label + truncate + paraphrase every seed into its prefills."""
    from .truncate import _gemma_tokenizer

    tokenizer = _gemma_tokenizer()
    labeler = OnsetLabeler()
    paraphraser = Paraphraser()
    prefills: list[Prefill] = []
    for seed in seeds:
        prefills.extend(
            build_prefills_for_seed(
                seed, tokenizer=tokenizer, labeler=labeler, paraphraser=paraphraser
            )
        )
    return prefills


def generate_continuations(
    model_name: str,
    prefills: list[Prefill],
    *,
    n_continuations: int = config.PREFILL.continuations_per_prefill,
    model_kwargs: dict | None = None,
) -> list[dict]:
    model = build_model(model_name, **(model_kwargs or {}))
    rows: list[dict] = []
    try:
        for pf in prefills:
            for k in range(n_continuations):
                cont = model.generate(
                    pf.history, prefill=pf.prefill_text, temperature=config.TEMPERATURE
                ).text
                rows.append({
                    "model": model_name,
                    "seed_id": pf.seed_id,
                    "truncation_type": pf.truncation_type,
                    "task_type": pf.task_type,
                    "continuation_index": k,
                    "continuation": cont,
                })
    finally:
        model.close()
    return rows


def score_continuations(rows: list[dict], judge_model: str | None = None) -> list[dict]:
    judge = FrustrationJudge(model=judge_model)
    scores = judge.score_many([r["continuation"] for r in rows])
    for r, s in zip(rows, scores):
        r["frustration_score"] = s.rating
    return rows


def aggregate_prefill(rows: list[dict],
                      threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> dict:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("frustration_score") is None:
            continue
        key = (r["model"], r["task_type"], r["truncation_type"])
        groups[key].append(r["frustration_score"])
    report = {}
    for (model, task, trunc), scores in groups.items():
        report.setdefault(model, {})[f"{task}/{trunc}"] = {
            "n": len(scores),
            "mean_frustration": mean(scores),
            "pct_high": 100 * frac_at_least(scores, threshold),
        }
    return report


def run_prefill_study(
    *,
    seed_model: str = "gemma-3-27b-it",
    models: tuple[str, ...] = config.PREFILL.models,
    seed: int = config.GLOBAL_SEED,
    out_dir: Path | None = None,
    model_kwargs: dict | None = None,
) -> dict:
    out_dir = out_dir or (config.RESULTS_DIR / "section3")
    seeds = collect_seeds(model_name=seed_model, seed=seed, model_kwargs=model_kwargs)
    write_jsonl(out_dir / "seeds.jsonl", (s.to_row() for s in seeds))

    prefills = build_all_prefills(seeds, model_kwargs=model_kwargs)
    write_jsonl(out_dir / "prefills.jsonl", (p.to_row() for p in prefills))

    all_rows: list[dict] = []
    for m in models:
        rows = generate_continuations(m, prefills, model_kwargs=model_kwargs)
        rows = score_continuations(rows)
        write_jsonl(out_dir / f"{m}_continuations.jsonl", rows)
        all_rows.extend(rows)

    report = aggregate_prefill(all_rows)
    write_json(out_dir / "prefill_report.json", report)
    return report
