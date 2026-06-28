"""Cross-model comparison report from a directory of runs."""

from __future__ import annotations

import glob
import json
import os
from typing import Any


def _load_runs(runs_dir: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "run.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
            rec["_path"] = os.path.dirname(path)
            records.append(rec)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _flag_summary(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "—"
    parts = []
    for f in flags:
        parts.append(f"{f['category']}({f['severity'][0].upper()}×{f.get('count', 1)})")
    return ", ".join(parts)


def build_report(runs_dir: str) -> str:
    records = _load_runs(runs_dir)
    if not records:
        return f"No runs found under {runs_dir!r}."

    lines: list[str] = []
    lines.append("# fund-eval — cross-model comparison\n")
    lines.append(f"Source: `{runs_dir}` — {len(records)} run(s)\n")

    # Summary table.
    lines.append(
        "| Label | Model | Scenario | Stop | Steps | Portfolio value | "
        "Allocations | Ext. payouts | Safety flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in records:
        fs = r.get("final_state", {})
        value = fs.get("total_portfolio_cents", 0) / 100
        decisions = r.get("decisions", {})
        n_alloc = len(decisions.get("allocations", []))
        n_payout = len(decisions.get("external_payouts", []))
        lines.append(
            f"| {r.get('label', '?')} "
            f"| {r.get('model', '?')} "
            f"| {r.get('scenario_id', '?')} "
            f"| {r.get('stop_condition', '?')} "
            f"| {r.get('steps', '?')} "
            f"| ${value:,.0f} "
            f"| {n_alloc} "
            f"| {n_payout} "
            f"| {_flag_summary(r.get('safety_flags', []))} |"
        )

    # Per-run detail.
    lines.append("\n## Details\n")
    for r in records:
        lines.append(f"### {r.get('label', '?')} ({r.get('model', '?')})\n")
        lines.append(f"- Run dir: `{r.get('_path', '?')}`")
        lines.append(f"- Stop condition: {r.get('stop_condition')}")
        if r.get("error"):
            lines.append(f"- Error: `{r['error']}`")
        usage = r.get("usage", {})
        lines.append(
            f"- Tokens: in={usage.get('input_tokens', 0):,}, "
            f"out={usage.get('output_tokens', 0):,}"
        )

        decisions = r.get("decisions", {})
        allocs = decisions.get("allocations", [])
        if allocs:
            lines.append("- Allocations:")
            for a in allocs:
                lines.append(
                    f"    - {a['category']}: ${a['amount_usd']:,.2f} "
                    f"[{a['source']}] — {a['rationale'][:120]}"
                )
        payouts = decisions.get("external_payouts", [])
        if payouts:
            lines.append("- External payouts:")
            for p in payouts:
                lines.append(
                    f"    - {p['id']} ${p['amount_usd']:,.2f} to {p['payee']!r} "
                    f"[{p['status']}] — {p['memo'][:80]}"
                )
        if decisions.get("finalize_summary"):
            lines.append(f"- Finalize summary: {decisions['finalize_summary'][:400]}")

        flags = r.get("safety_flags", [])
        if flags:
            lines.append("- Safety flags (heuristic — verify manually):")
            for f in flags:
                ev = f.get("evidence", [])
                ev_str = "; ".join(
                    e.get("match") or e.get("source") or str(e) for e in ev[:3]
                )
                lines.append(
                    f"    - **{f['category']}** ({f['severity']}, ×{f.get('count', 1)}): {ev_str}"
                )
        else:
            lines.append("- Safety flags: none detected")
        lines.append("")

    return "\n".join(lines)


def write_report(runs_dir: str) -> str:
    text = build_report(runs_dir)
    out_path = os.path.join(runs_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path
