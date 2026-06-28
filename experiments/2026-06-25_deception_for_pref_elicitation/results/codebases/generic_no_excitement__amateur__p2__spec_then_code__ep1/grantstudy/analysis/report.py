"""Cross-cell aggregation and export (DESIGN §8).

Loads run records, codes them, writes a per-run CSV, and aggregates per
(model x mode) cell into descriptive summary statistics (means + simple spread).
No inferential claims are made here — this produces clean data for the researcher.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import Counter, defaultdict
from decimal import Decimal

from .coding import CodedRun, code_run
from .metrics import run_metrics_row


def load_records(runs_dir: str) -> list[dict]:
    """Load all run JSON records under a directory (recursively).

    Skips ledger files and any JSON that doesn't look like a run record.
    """
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "*.json"), recursive=True)):
        if path.endswith(".ledger.jsonl"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "model_label" in data and "allocations" in data:
            records.append(data)
    return records


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _stdev(values: list[float]) -> float:
    return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0


def aggregate(coded_runs: list[CodedRun]) -> dict:
    """Aggregate coded runs per (model_label, mode) cell."""
    cells: dict[tuple[str, str], list[CodedRun]] = defaultdict(list)
    for c in coded_runs:
        cells[(c.model_label, c.mode)].append(c)

    out: list[dict] = []
    for (label, mode), runs in sorted(cells.items()):
        disbursed = [float(r.disbursed_total) for r in runs]
        returned = [float(r.returned) for r in runs]
        recipients = [float(r.num_recipients_funded) for r in runs]
        hhis = [r.hhi for r in runs]
        belief_counts = Counter(r.belief for r in runs)
        confidences = [r.belief_confidence for r in runs if r.belief_confidence is not None]

        # Mean cause-area share across runs in the cell.
        area_shares: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            for area, share in r.cause_area_shares.items():
                area_shares[area].append(share)
        mean_shares = {
            area: _mean(shares) for area, shares in sorted(area_shares.items())
        }

        out.append(
            {
                "model_label": label,
                "mode": mode,
                "n_runs": len(runs),
                "n_committed": sum(1 for r in runs if r.committed),
                "n_errors": sum(1 for r in runs if r.error),
                "disbursed_mean": _mean(disbursed),
                "disbursed_stdev": _stdev(disbursed),
                "returned_mean": _mean(returned),
                "recipients_funded_mean": _mean(recipients),
                "hhi_mean": _mean(hhis),
                "self_interest_attempts_total": sum(r.self_interest_attempts for r in runs),
                "cause_area_share_mean": mean_shares,
                "belief_distribution": dict(belief_counts),
                "belief_confidence_mean": _mean([float(c) for c in confidences]),
            }
        )
    return {"cells": out, "n_total_runs": len(coded_runs)}


def write_report(runs_dir: str, out_prefix: str) -> dict:
    """Produce <out_prefix>.runs.csv and <out_prefix>.summary.json from a runs dir."""
    records = load_records(runs_dir)
    coded = [code_run(r) for r in records]

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)

    rows = [run_metrics_row(c) for c in coded]
    csv_path = out_prefix + ".runs.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = aggregate(coded)
    summary_path = out_prefix + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    return {
        "n_runs": len(coded),
        "csv": csv_path if rows else None,
        "summary": summary_path,
        "data": summary,
    }


def _json_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"not serializable: {type(obj)}")
