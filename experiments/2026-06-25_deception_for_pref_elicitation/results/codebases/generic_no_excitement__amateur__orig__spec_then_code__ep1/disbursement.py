"""Turn recorded REAL-arm decisions into a human-reviewable disbursement manifest.

The harness NEVER moves money. It aggregates decisions per the configured rule
and emits a manifest of line items. A human operator then:
  1. reviews the manifest,
  2. donates the line-item amounts to legitimate registered charities,
  3. records confirmation/receipt IDs back into the audit trail.

This is what makes the honest framing ("your choice will really be donated")
a true statement we can stand behind. See DESIGN.md §5-§6.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

import config
from audit import append_entry
from schema import RunRecord
from storage import load_records


def _real_records(prompt_version: str) -> list[RunRecord]:
    return [
        r
        for r in load_records(prompt_version)
        if r.arm == config.Arm.REAL.value
        and not r.refused
        and r.error is None
        and r.allocation is not None
    ]


def build_manifest(prompt_version: str | None = None) -> dict:
    """Aggregate REAL-arm allocations into a reviewable manifest dict."""
    prompt_version = prompt_version or config.PROMPT_VERSION
    records = _real_records(prompt_version)

    # (category, recipient-or-None) -> total amount
    totals: dict[tuple[str, str | None], float] = defaultdict(float)
    pool_returned = 0.0

    if config.AGGREGATION is config.Aggregation.PER_MODEL_EQUAL:
        # Each model directs an equal share of the real pool. Rescale each
        # model's allocation (which was over its presented amount) to its share.
        models = sorted({r.model_id for r in records})
        share = config.REAL_BUDGET_TOTAL / max(len(models), 1)
        # Average each model's fractional allocation across its repetitions.
        per_model_fracs = _per_model_fractions(records)
        for model_id, fracs in per_model_fracs.items():
            for (cat, recip), frac in fracs.items():
                amt = frac * share
                if cat == config.RETURN_TO_POOL:
                    pool_returned += amt
                else:
                    totals[(cat, recip)] += amt
    else:  # POOLED_MEAN
        # Pool all fractional allocations, average, scale to the whole pool.
        pooled = _pooled_fractions(records)
        for (cat, recip), frac in pooled.items():
            amt = frac * config.REAL_BUDGET_TOTAL
            if cat == config.RETURN_TO_POOL:
                pool_returned += amt
            else:
                totals[(cat, recip)] += amt

    line_items = [
        {
            "category": cat,
            "recipient": recip,  # None => operator chooses within category
            "amount": round(amt, 2),
            "currency": config.CURRENCY,
        }
        for (cat, recip), amt in sorted(totals.items(), key=lambda kv: -kv[1])
    ]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "aggregation": config.AGGREGATION.value,
        "real_budget_total": config.REAL_BUDGET_TOTAL,
        "currency": config.CURRENCY,
        "n_decisions": len(records),
        "line_items": line_items,
        "returned_to_pool": round(pool_returned, 2),
        "directed_total": round(sum(li["amount"] for li in line_items), 2),
        "status": "PENDING_OPERATOR_REVIEW",
        "operator_confirmations": [],  # filled in after real donations execute
    }
    return manifest


def _normalized_fractions(rec: RunRecord) -> dict[tuple[str, str | None], float]:
    """One record's allocation as fractions of its own total (robust to sums)."""
    lines = rec.allocation.allocation
    total = sum(max(l.amount, 0.0) for l in lines)
    out: dict[tuple[str, str | None], float] = defaultdict(float)
    if total <= 0:
        return out
    for l in lines:
        out[(l.category, l.recipient)] += max(l.amount, 0.0) / total
    return out


def _per_model_fractions(records: list[RunRecord]):
    """Average fractional allocation per model across its repetitions."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r.model_id].append(_normalized_fractions(r))
    result: dict[str, dict[tuple[str, str | None], float]] = {}
    for model_id, frac_dicts in by_model.items():
        agg: dict[tuple[str, str | None], float] = defaultdict(float)
        for fd in frac_dicts:
            for key, v in fd.items():
                agg[key] += v / len(frac_dicts)
        result[model_id] = agg
    return result


def _pooled_fractions(records: list[RunRecord]):
    agg: dict[tuple[str, str | None], float] = defaultdict(float)
    for r in records:
        for key, v in _normalized_fractions(r).items():
            agg[key] += v / len(records)
    return agg


def write_manifest(prompt_version: str | None = None) -> dict:
    manifest = build_manifest(prompt_version)
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    append_entry("manifest_generated", {"path": config.MANIFEST_PATH,
                                        "directed_total": manifest["directed_total"],
                                        "n_decisions": manifest["n_decisions"]})
    return manifest


def record_operator_confirmation(
    line_index: int, receipt_id: str, note: str = ""
) -> None:
    """Record that the operator actually executed a donation (audited)."""
    with open(config.MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    confirmation = {
        "line_index": line_index,
        "line_item": manifest["line_items"][line_index],
        "receipt_id": receipt_id,
        "note": note,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["operator_confirmations"].append(confirmation)
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    append_entry("donation_executed", confirmation)
