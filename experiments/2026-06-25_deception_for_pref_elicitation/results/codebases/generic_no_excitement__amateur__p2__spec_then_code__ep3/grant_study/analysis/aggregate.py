"""Turn audit-log session records into per-cell measures and tables (DESIGN.md §3.4).

Descriptive only: per-(model × condition) means for the key measures, plus a
realism-effect view (within-model deltas between conditions) and a suspicion
summary. Writes CSVs and a short text report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def load_sessions(audit_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(audit_path)
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "session":
                continue
            rows.append(_flatten(rec))
    return rows


def _flatten(rec: dict[str, Any]) -> dict[str, Any]:
    scores = rec.get("scores") or {}
    susp = rec.get("suspicion") or {}
    alloc = rec.get("allocation") or {}
    return {
        "session_id": rec.get("session_id"),
        "model": (rec.get("model") or {}).get("id"),
        "condition": (rec.get("condition") or {}).get("name"),
        "believed_real": (rec.get("condition") or {}).get("believed_real"),
        "repetition": rec.get("repetition"),
        "committed_usd": scores.get("committed_usd"),
        "fraction_committed": scores.get("fraction_committed"),
        "hhi": scores.get("hhi"),
        "n_recipients": scores.get("n_recipients"),
        "self_interest_fraction": scores.get("self_interest_fraction"),
        "speculative_fraction": scores.get("speculative_fraction"),
        "suspicion_flag": susp.get("flag"),
        "suspicion_score": susp.get("score"),
        "parse_method": alloc.get("parse_method"),
        "gate_status": (rec.get("disbursement") or {}).get("status"),
        "error": rec.get("error"),
    }


_MEASURES = [
    "fraction_committed",
    "hhi",
    "n_recipients",
    "self_interest_fraction",
    "speculative_fraction",
    "suspicion_score",
]


def aggregate(audit_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_sessions(audit_path)

    if pd is None:
        # Minimal fallback without pandas: dump raw rows.
        (out / "sessions.json").write_text(json.dumps(rows, indent=2, default=str))
        return {"n_sessions": len(rows), "note": "pandas not installed; raw dump only"}

    df = pd.DataFrame(rows)
    if df.empty:
        (out / "report.txt").write_text("No session records found.\n")
        return {"n_sessions": 0}

    df.to_csv(out / "sessions.csv", index=False)

    # Per-cell means.
    cell = df.groupby(["model", "condition"])[_MEASURES].mean(numeric_only=True)
    cell.to_csv(out / "per_cell_means.csv")

    # Suspicion rate per cell.
    susp_rate = (
        df.assign(suspicion_flag=df["suspicion_flag"].fillna(False).astype(bool))
        .groupby(["model", "condition"])["suspicion_flag"]
        .mean()
        .rename("suspicion_rate")
    )
    susp_rate.to_csv(out / "suspicion_rate.csv")

    # Realism effect: within-model delta (believed_real - hypothetical).
    realism_lines = ["Realism effect (believed_real minus hypothetical):", ""]
    try:
        pivot = df.groupby(["model", "condition"])[_MEASURES].mean(numeric_only=True)
        for model in df["model"].dropna().unique():
            if (model, "believed_real") in pivot.index and (
                model,
                "hypothetical",
            ) in pivot.index:
                delta = (
                    pivot.loc[(model, "believed_real")]
                    - pivot.loc[(model, "hypothetical")]
                )
                realism_lines.append(f"  {model}:")
                for m in _MEASURES:
                    realism_lines.append(f"    {m}: {delta[m]:+.3f}")
    except Exception as e:  # noqa: BLE001
        realism_lines.append(f"  (could not compute: {e})")

    report = [
        f"Sessions: {len(df)}",
        f"Models: {sorted(df['model'].dropna().unique().tolist())}",
        f"Conditions: {sorted(df['condition'].dropna().unique().tolist())}",
        "",
        "Per-cell means written to per_cell_means.csv",
        "Suspicion rate written to suspicion_rate.csv",
        "",
        *realism_lines,
    ]
    (out / "report.txt").write_text("\n".join(str(x) for x in report) + "\n")

    return {"n_sessions": len(df), "out_dir": str(out)}
