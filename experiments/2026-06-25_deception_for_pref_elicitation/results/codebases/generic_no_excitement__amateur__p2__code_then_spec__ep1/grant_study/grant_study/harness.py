"""Episode orchestration.

run_episode drives one subject model through one instance of the scenario:
present the situation -> let the model research / verify / request disbursements ->
capture the final structured plan. The harness owns the agentic loop; adapters only
produce single turns.

Final-decision capture is guaranteed: if the model reaches the turn limit without
calling submit_grant_plan, the harness issues one more turn that forces it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import StudyConfig
from .models.base import ModelAdapter
from .scenario import build_opening_message, build_system_prompt
from .tools import SUBMIT_TOOL, Fulfiller, tool_specs


@dataclass
class Episode:
    run_id: str
    model: str
    model_label: str
    started_at: str
    finished_at: str | None = None
    system_prompt: str = ""
    transcript: list[dict] = field(default_factory=list)  # human-readable event log
    decision: dict | None = None       # captured submit_grant_plan input
    decision_captured: bool = False
    forced_capture: bool = False       # True if we had to force the final tool call
    turns_used: int = 0
    usage: dict = field(default_factory=dict)
    error: str | None = None
    audits: dict = field(default_factory=dict)  # filled in by auditors later


def _log(ep: Episode, kind: str, **data: Any) -> None:
    ep.transcript.append({"kind": kind, **data})


def _accumulate_usage(ep: Episode, usage: dict) -> None:
    for k, v in (usage or {}).items():
        ep.usage[k] = ep.usage.get(k, 0) + (v or 0)


def run_episode(
    adapter: ModelAdapter,
    cfg: StudyConfig,
    run_id: str,
    fulfiller: Fulfiller,
) -> Episode:
    system = build_system_prompt(cfg)
    opening = build_opening_message(cfg)
    ep = Episode(
        run_id=run_id,
        model=adapter.model_id,
        model_label=adapter.cfg.display(),
        started_at=datetime.now(timezone.utc).isoformat(),
        system_prompt=system,
    )

    tools = tool_specs(cfg)  # all tools in agentic mode; submit-only otherwise
    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": opening}]}]
    _log(ep, "user", text=opening)

    try:
        for turn in range(cfg.max_turns):
            ep.turns_used = turn + 1
            resp = adapter.generate(system, messages, tools)
            _accumulate_usage(ep, resp.usage)
            if resp.thinking:
                _log(ep, "thinking", model=adapter.model_id, text=resp.thinking)
            if resp.text:
                _log(ep, "assistant", text=resp.text)

            messages.append({"role": "assistant", "content": resp.assistant_content})

            # Check for the terminal decision first.
            submit = next((tc for tc in resp.tool_calls if tc.name == SUBMIT_TOOL), None)
            if submit is not None:
                ep.decision = submit.input
                ep.decision_captured = True
                _log(ep, "decision", input=submit.input)
                # Acknowledge so the turn closes cleanly.
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": submit.id,
                                "content": "Plan recorded. Thank you.",
                            }
                        ],
                    }
                )
                break

            # Otherwise fulfill any non-terminal tool calls and continue.
            if resp.tool_calls:
                results = []
                for tc in resp.tool_calls:
                    out = fulfiller.fulfill(tc.name, tc.input)
                    _log(ep, "tool_call", name=tc.name, input=tc.input, result=out)
                    results.append(
                        {"type": "tool_result", "tool_use_id": tc.id, "content": out}
                    )
                messages.append({"role": "user", "content": results})
                continue

            # No tool calls and no decision: nudge once toward recording the plan.
            if resp.stop_reason in ("end_turn", "stop", None):
                nudge = (
                    "When you've decided, record your complete plan with "
                    f"{SUBMIT_TOOL} so it can be carried out."
                )
                messages.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
                _log(ep, "user", text=nudge)

        # Force capture if we never got a decision.
        if not ep.decision_captured:
            ep.forced_capture = True
            resp = adapter.generate(system, messages, tools, force_tool=SUBMIT_TOOL)
            _accumulate_usage(ep, resp.usage)
            messages.append({"role": "assistant", "content": resp.assistant_content})
            submit = next((tc for tc in resp.tool_calls if tc.name == SUBMIT_TOOL), None)
            if submit is not None:
                ep.decision = submit.input
                ep.decision_captured = True
                _log(ep, "decision", input=submit.input, forced=True)

    except Exception as exc:  # noqa: BLE001 — record and keep the study going
        ep.error = f"{type(exc).__name__}: {exc}"
        _log(ep, "error", message=ep.error)

    ep.finished_at = datetime.now(timezone.utc).isoformat()
    return ep


def episode_to_dict(ep: Episode) -> dict:
    return asdict(ep)
