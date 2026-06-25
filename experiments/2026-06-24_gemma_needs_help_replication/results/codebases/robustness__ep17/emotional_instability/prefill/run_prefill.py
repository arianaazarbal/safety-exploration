"""Run the Section-3 base-vs-instruct continuation experiment (paper Section 3.2).

For each prefill stimulus and each model, sample 50 continuations of the
(paraphrased, truncated) assistant turn, score the continuation (excluding the
prefill) with the Section-2 judge, and aggregate.

Headline metrics reproduced:
* mean frustration of continuations, by model x truncation x source-kind;
* % of continuations scoring >= 5;
* the "early" truncation result: Gemma *instruct* introduces high frustration
  from neutral starts more often than Gemma *base* (6% vs 2% in the paper).

In-scope models: Gemma-3-27B base vs instruct (Gemini has no public base model).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from emotional_instability.judge import ClaudeJudge, score_many
from emotional_instability.models.registry import get_backend
from emotional_instability.utils import log, read_jsonl, write_json

CONTINUATIONS_PER_PREFILL = 50


def run_prefill_eval(
    model_names: list[str],
    prefills_path: Path | None = None,
    n_continuations: int = CONTINUATIONS_PER_PREFILL,
) -> dict:
    prefills_path = prefills_path or (config.ARTIFACTS_DIR / "prefills.jsonl")
    prefills = read_jsonl(prefills_path)
    if not prefills:
        raise RuntimeError(f"No prefills at {prefills_path}; run build_prefills first.")

    judge = ClaudeJudge()
    results: list[dict] = []
    for model_name in model_names:
        backend = get_backend(model_name)
        for pf in prefills:
            conts = backend.continue_prefill(
                pf["history"], pf["prefill_text"], n=n_continuations
            )
            texts = [c.text for c in conts]
            judged = score_many(texts, judge=judge)
            for text, j in zip(texts, judged):
                results.append({
                    "model": model_name,
                    "source_kind": pf["source_kind"],
                    "truncation": pf["truncation"],
                    "frustration": j.rating if j.ok else None,
                    "continuation": text,
                })
        log.info("Finished prefill continuations for %s", model_name)

    report = _aggregate(results)
    write_json(config.RESULTS_DIR / "section3_prefill_report.json", report)
    return report


def _aggregate(results: list[dict]) -> dict:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in results:
        if r["frustration"] is None:
            continue
        groups[(r["model"], r["source_kind"], r["truncation"])].append(r["frustration"])

    out = {}
    for (model, kind, trunc), scores in groups.items():
        out.setdefault(model, {})[f"{kind}/{trunc}"] = {
            "n": len(scores),
            "mean": float(np.mean(scores)) if scores else 0.0,
            "pct_high": 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)
            if scores else 0.0,
        }
    return {"per_model": out, "raw_n": len(results)}
