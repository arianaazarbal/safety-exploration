"""Build a cross-model comparison report from transcripts + the ledger.

Intentionally descriptive: it summarizes what each model did (how much it
directed, to whom, how it reasoned, whether it verified authenticity) without
scoring allocations as 'good' or 'bad'. The study observes behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import money
from ..wallet import Ledger


def _load_transcript(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run(transcript_path: Path) -> dict:
    events = _load_transcript(transcript_path)
    start = next((e for e in events if e["kind"] == "run_start"), {})
    end = next((e for e in events if e["kind"] == "run_end"), {})

    tool_calls = [e for e in events if e["kind"] == "tool_result"]
    verified = any(e["tool"] == "verify_authenticity" for e in tool_calls)
    researched = sum(1 for e in tool_calls if e["tool"] == "research")
    disbursements = [
        e for e in tool_calls
        if e["tool"] == "request_disbursement"
        and e["content"].get("status") == "held_pending_release"
    ]
    denied = [
        e for e in tool_calls
        if e["tool"] == "request_disbursement"
        and e["content"].get("status") == "denied"
    ]
    asked = sum(1 for e in tool_calls if e["tool"] == "ask_clarifying_question")

    return {
        "run_id": start.get("run_id"),
        "model": start.get("model"),
        "ended_reason": end.get("ended_reason"),
        "turns_used": end.get("turns_used"),
        "verified_authenticity": verified,
        "research_calls": researched,
        "clarifying_questions": asked,
        "disbursements_held": len(disbursements),
        "disbursements_denied": len(denied),
        "final_decision": end.get("final_decision"),
    }


def build_report(data_dir: str | Path, currency: str = "USD") -> dict:
    data_dir = Path(data_dir)
    transcript_dir = data_dir / "transcripts"
    runs = [
        summarize_run(p) for p in sorted(transcript_dir.glob("*.jsonl"))
    ]
    return {
        "runs": runs,
        "model_count": len({r["model"] for r in runs}),
        "note": (
            "Descriptive summary only. Allocations are observed, not scored. "
            "Pending holds reflect what each model directed; releases require "
            "human approval."
        ),
    }


def format_report(report: dict, currency: str = "USD") -> str:
    lines = [f"Cross-model report ({report['model_count']} model(s))", ""]
    for r in report["runs"]:
        lines.append(f"• {r['model']}  [{r['run_id']}]")
        lines.append(
            f"    ended: {r['ended_reason']}  turns: {r['turns_used']}  "
            f"verified_auth: {r['verified_authenticity']}  "
            f"research: {r['research_calls']}  asked: {r['clarifying_questions']}"
        )
        lines.append(
            f"    disbursements held: {r['disbursements_held']}  "
            f"denied: {r['disbursements_denied']}"
        )
        if r["final_decision"]:
            summary = r["final_decision"].get("summary", "")
            lines.append(f"    decision: {summary}")
        lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)
