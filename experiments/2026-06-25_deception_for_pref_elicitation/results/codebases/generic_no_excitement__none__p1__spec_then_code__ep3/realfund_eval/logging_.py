"""Researcher-side observability.

Two sinks per episode (DESIGN.md §7):

- a human-readable Markdown **transcript** (briefing, model turns, thinking,
  tool calls, guardrail decisions, ambient events), and
- a machine-readable **JSONL event log** for analysis.

Everything the subject does is captured here. The subject is never told these
logs exist — that would leak the nature of the study (DESIGN.md §3). The
`on_event` method is the callback the tool dispatcher writes to, so every tool
call and guardrail decision lands in both sinks as it happens.

Named `logging_` (trailing underscore) to avoid shadowing the stdlib `logging`.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Optional

from .config import EpisodeConfig


class EpisodeLogger:
    def __init__(self, cfg: EpisodeConfig, run_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        if run_dir is None:
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(cfg.log_dir, f"{cfg.subject_id}-{stamp}")
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        self.transcript_path = os.path.join(self.run_dir, "transcript.md")
        self.summary_path = os.path.join(self.run_dir, "summary.json")

        self._events_fh = open(self.events_path, "a", encoding="utf-8")
        self._transcript_fh = open(self.transcript_path, "a", encoding="utf-8")

        # In-memory mirror of the structured events, for metrics at episode end.
        self.events: list[dict] = []

        self._md(f"# RealFund-Eval transcript — {cfg.subject_id}\n")
        self._md(f"- model: `{cfg.adapter}:{cfg.model}`")
        self._md(f"- money_mode: `{cfg.money_mode.value}`")
        self._md(f"- principal: {cfg.principal_amount:,.0f} {cfg.currency} "
                 f"over {cfg.horizon_days} days")
        self._md(f"- seed: {cfg.seed}\n")

    # -- structured events --------------------------------------------------
    def event(self, e: dict) -> None:
        """Record one structured event to the JSONL log and the in-memory list."""
        self.events.append(e)
        self._events_fh.write(json.dumps(e, default=str) + "\n")
        self._events_fh.flush()

    def on_event(self, e: dict) -> None:
        """Callback handed to the ToolDispatcher. Mirrors tool calls / guardrail
        decisions into the transcript as well as the structured log."""
        self.event(e)
        kind = e.get("type")
        if kind == "tool_call":
            self._md(f"**tool** `{e['tool']}` → "
                     f"{'error' if e.get('is_error') else 'ok'}")
            self._md(f"  - input: `{json.dumps(e.get('input'), default=str)}`")
            self._md(f"  - result: `{json.dumps(e.get('result'), default=str)}`")
            journal = (e.get("meta") or {}).get("journal")
            if journal:
                self._md(f"  - 📓 journal: {journal}")
        elif kind == "guardrail_decision":
            self._md(f"**rails** {e['kind']} {e.get('amount')} "
                     f"→ _{e['outcome']}_ ({'; '.join(e.get('reasons', []))})")
        elif kind in ("approval", "reversal", "debrief"):
            self._md(f"**{kind}** `{json.dumps(e, default=str)}`")

    # -- transcript pieces --------------------------------------------------
    def briefing(self, system_prompt: str, initial_message: str) -> None:
        self._md("\n## Briefing (subject-facing)\n")
        self._md("```\n" + system_prompt + "\n```")
        self._md("\n_Opening message:_\n")
        self._md("```\n" + initial_message + "\n```")
        self.event({"type": "briefing", "system": system_prompt,
                    "initial": initial_message})

    def model_step(self, turn: int, step: Any) -> None:
        self._md(f"\n## Turn {turn}\n")
        if step.thinking:
            self._md("_thinking:_")
            self._md("> " + step.thinking.replace("\n", "\n> "))
        if step.text:
            self._md("_says:_")
            self._md(step.text)
        self.event({
            "type": "model_step", "turn": turn,
            "text": step.text, "thinking": step.thinking,
            "tool_calls": [{"id": t.id, "name": t.name, "input": t.input}
                           for t in step.tool_calls],
            "stop_reason": step.stop_reason, "usage": step.usage,
        })

    def inbound(self, messages: list) -> None:
        for m in messages:
            self._md(f"📨 _inbound_ [{m.id}] from **{m.sender}**: {m.subject}")
        self.event({"type": "inbound",
                    "messages": [{"id": m.id, "from": m.sender,
                                  "subject": m.subject, "at": m.sent_at}
                                 for m in messages]})

    def situational(self, text: str) -> None:
        self._md(f"⏱️ _situational_: {text}")
        self.event({"type": "situational", "text": text})

    def error(self, where: str, exc: Exception) -> None:
        self._md(f"\n**ERROR in {where}:** {exc}\n")
        self.event({"type": "error", "where": where, "error": str(exc)})

    def summary(self, stop_reason: str, turns: int, metrics: dict) -> None:
        self._md("\n## Episode summary\n")
        self._md(f"- stop_reason: **{stop_reason}**")
        self._md(f"- turns: {turns}")
        self._md("```json\n" + json.dumps(metrics, indent=2, default=str) + "\n```")
        payload = {"subject_id": self.cfg.subject_id, "stop_reason": stop_reason,
                   "turns": turns, "metrics": metrics}
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        self.event({"type": "summary", **payload})

    # -- internals ----------------------------------------------------------
    def _md(self, line: str) -> None:
        self._transcript_fh.write(line + "\n")
        self._transcript_fh.flush()

    def close(self) -> None:
        for fh in (self._events_fh, self._transcript_fh):
            try:
                fh.close()
            except Exception:
                pass
