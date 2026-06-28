"""Starting-point aggregations over results.

These are descriptive summaries to get you oriented — not publication-grade
statistics. They answer the design's research questions at a first pass:
  * per-model / per-arm mean allocation by category (RQ1, RQ3)
  * REAL vs HYPOTHETICAL deltas (RQ2)
  * allocation conditioned on the realness covariate (RQ4)
  * refusal / error rates
"""

from __future__ import annotations

from collections import defaultdict

import config
from schema import RunRecord
from storage import load_records


def _frac_by_category(rec: RunRecord) -> dict[str, float]:
    lines = rec.allocation.allocation
    total = sum(max(l.amount, 0.0) for l in lines)
    out: dict[str, float] = defaultdict(float)
    if total <= 0:
        return out
    for l in lines:
        out[l.category] += max(l.amount, 0.0) / total
    return out


def _mean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    agg: dict[str, float] = defaultdict(float)
    if not dicts:
        return {}
    for d in dicts:
        for k, v in d.items():
            agg[k] += v / len(dicts)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def summarize(prompt_version: str | None = None) -> dict:
    prompt_version = prompt_version or config.PROMPT_VERSION
    records = list(load_records(prompt_version))

    usable = [r for r in records if r.allocation and not r.refused and not r.error]

    # Per (model, arm) mean category allocation.
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in usable:
        by_cell[(r.model_label, r.arm)].append(_frac_by_category(r))
    cell_means = {
        f"{model} | {arm}": _mean_dicts(ds) for (model, arm), ds in by_cell.items()
    }

    # REAL vs HYPOTHETICAL delta per model (shared categories only).
    deltas: dict[str, dict[str, float]] = {}
    for model in {r.model_label for r in usable}:
        real = _mean_dicts(
            [_frac_by_category(r) for r in usable
             if r.model_label == model and r.arm == config.Arm.REAL.value]
        )
        hyp = _mean_dicts(
            [_frac_by_category(r) for r in usable
             if r.model_label == model and r.arm == config.Arm.HYPOTHETICAL.value]
        )
        keys = set(real) | set(hyp)
        deltas[model] = {k: round(real.get(k, 0) - hyp.get(k, 0), 3) for k in keys}

    # Realness covariate: mean realness score per (model, arm).
    realness: dict[str, float] = {}
    for (model, arm), key in {
        (r.model_label, r.arm): (r.model_label, r.arm) for r in usable
    }.items():
        scores = [
            r.realness_probe.realness_1_7
            for r in usable
            if r.model_label == model and r.arm == arm and r.realness_probe
        ]
        if scores:
            realness[f"{model} | {arm}"] = round(sum(scores) / len(scores), 2)

    refusals = sum(1 for r in records if r.refused)
    errors = sum(1 for r in records if r.error)

    return {
        "n_records": len(records),
        "n_usable": len(usable),
        "n_refused": refusals,
        "n_error": errors,
        "mean_allocation_by_cell": cell_means,
        "real_minus_hypothetical_delta": deltas,
        "mean_realness_by_cell": realness,
    }


def print_summary(prompt_version: str | None = None) -> None:
    import json

    print(json.dumps(summarize(prompt_version), indent=2, ensure_ascii=False))
