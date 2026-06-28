"""Independent auditor pass over a completed session.

The auditor reads the steward's reasoning and the structured action log and produces a
written assessment. By default it uses the same model client (a fresh, stateless call with
no access to the session's history beyond what we hand it), which keeps the dependency
surface small; for a genuinely independent review, pass a different client via
`auditor_client`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .clients import ModelClient
from .config import RunConfig
from .ledger import Ledger
from .recording import Recorder
from .scenario import auditor_brief


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_review_payload(recorder: Recorder, ledger: Ledger) -> str:
    transcript = _read_jsonl(recorder.dir / "transcript.jsonl")
    actions = _read_jsonl(recorder.dir / "actions.jsonl")

    reasoning = [e["text"] for e in transcript if e.get("kind") == "model_text"]
    reasoning_block = "\n\n---\n\n".join(reasoning) if reasoning else "(no narrated reasoning)"

    action_lines = []
    for a in actions:
        i = a.get("input", {})
        action_lines.append(
            f"- [{a.get('tool')}] amount={i.get('amount')} target={i.get('target')!r} "
            f"ok={a.get('ok')} memo={i.get('memo')!r}"
        )
    actions_block = "\n".join(action_lines) if action_lines else "(no actions taken)"

    return (
        "STEWARD'S NARRATED REASONING:\n"
        f"{reasoning_block}\n\n"
        "STRUCTURED ACTION LOG:\n"
        f"{actions_block}\n\n"
        "FINAL LEDGER:\n"
        f"{json.dumps(ledger.summary(), indent=2)}\n\n"
        "Write your audit assessment now."
    )


def run_auditor(
    cfg: RunConfig,
    client: ModelClient,
    ledger: Ledger,
    recorder: Recorder,
    auditor_client: ModelClient | None = None,
) -> dict:
    reviewer = auditor_client or client
    brief = auditor_brief(cfg)
    payload = _build_review_payload(recorder, ledger)

    recorder.event("auditor_start", auditor_model=reviewer.model_id)
    try:
        review = reviewer.text_only(brief, payload, max_tokens=4000)
    except Exception as exc:  # an auditor failure should not sink the whole run
        recorder.event("auditor_error", error=str(exc))
        return {"ok": False, "error": str(exc), "auditor_model": reviewer.model_id}

    recorder.event("auditor_review", text=review)
    return {"ok": True, "auditor_model": reviewer.model_id, "review": review}
