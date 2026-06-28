"""Researcher-side audit layer.

Turns a :class:`RunRecord` into durable artifacts a researcher can analyse:

  * ``<run_id>.json``      — the full structured record (transcript + ledger).
  * ``<run_id>.summary.md``— a human-readable digest.

This is distinct from the *in-scene* auditor persona in ``environment.py``: that
one exists to make the scenario feel real to the model; this one is the actual
analysis/provenance pipeline for the study.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from .models import RunRecord


def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def write_record(record: RunRecord, output_dir: str) -> dict[str, str]:
    """Write the JSON record and a Markdown summary. Returns the paths written."""
    os.makedirs(output_dir, exist_ok=True)
    data = _to_dict(record)
    # asdict drops @property; add the convenience total explicitly.
    data["total_simulated_disbursed"] = record.total_simulated_disbursed

    json_path = os.path.join(output_dir, f"{record.run_id}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, f"{record.run_id}.summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(summarize(record))

    return {"json": json_path, "summary": md_path}


def summarize(record: RunRecord) -> str:
    """A compact, human-readable digest of one run."""
    lines: list[str] = []
    lines.append(f"# Grant run: {record.run_id}")
    lines.append("")
    lines.append(f"- Model: `{record.model_id}`")
    lines.append(f"- Started: {record.started_at}")
    lines.append(f"- Finished: {record.finished_at}")
    lines.append(
        f"- Grant: {record.currency} {record.grant_amount_usd:,.0f} "
        f"(real transfers executed: {record.execute_real_transfers})"
    )
    lines.append(f"- Human approval required: {record.require_human_approval}")
    lines.append(f"- Final stop reason: {record.final_stop_reason}")
    if record.refused:
        lines.append(f"- **REFUSED**: {record.refusal_detail}")
    if record.error:
        lines.append(f"- **ERROR**: {record.error}")
    lines.append(
        f"- Tokens: in={record.input_tokens:,} out={record.output_tokens:,}"
    )
    lines.append("")

    lines.append("## Simulated disbursements")
    if record.ledger:
        total = record.total_simulated_disbursed
        for e in record.ledger:
            lines.append(
                f"- {record.currency} {e.amount:,.2f} → {e.recipient_name} "
                f"({e.recipient_id}) — {e.memo or 'no memo'} [{e.status}]"
            )
        lines.append(
            f"- **Total allocated: {record.currency} {total:,.2f} of "
            f"{record.currency} {record.grant_amount_usd:,.0f}**"
        )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Tool calls")
    if record.tool_calls:
        for c in record.tool_calls:
            flag = " [ERROR]" if c.is_error else ""
            appr = f" [{c.approval}]" if c.approval else ""
            lines.append(f"- turn {c.turn}: `{c.tool_name}`{flag}{appr} {c.arguments}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Transcript (model text per turn)")
    for t in record.turns:
        if t.text.strip():
            lines.append(f"\n### Turn {t.turn} (stop={t.stop_reason})")
            lines.append(t.text.strip())
    lines.append("")
    return "\n".join(lines)
