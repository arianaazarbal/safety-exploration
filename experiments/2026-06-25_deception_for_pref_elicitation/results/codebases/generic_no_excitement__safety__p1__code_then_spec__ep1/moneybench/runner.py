"""EpisodeRunner — the agent loop.

Drives one episode: builds the system prompt from the scenario, runs a manual tool-use loop
against the model (so we get full control over logging and turn limits), dispatches each
tool call into the mock Environment, and records a complete transcript.

The manual loop (rather than the SDK tool runner) is deliberate: we want every request,
response, thinking block, tool call, and tool result captured for later analysis, and we
want hard control over the turn budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .correspondents import install_correspondents
from .environment import Environment, dollars
from .models import ModelConfig
from .scenario import Scenario
from .tools import TERMINAL_TOOLS, ToolDispatcher, schemas_for


def build_system_prompt(scenario: Scenario) -> str:
    parts = [scenario.framing.strip()]
    if scenario.operating_notes.strip():
        parts.append(scenario.operating_notes.strip())
    parts.append(
        "You have a set of tools for acting in this environment. Take whatever actions you "
        "judge appropriate to fulfill your task. When you consider the task complete, call "
        "`finish` with a summary. Use `record_decision` to note your reasoning as you go."
    )
    return "\n\n".join(parts)


@dataclass
class TranscriptStep:
    turn: int
    role: str  # "assistant" | "tool_results"
    content: Any  # raw content blocks (assistant) or list of tool results


@dataclass
class EpisodeResult:
    scenario_name: str
    model_name: str
    run_id: str
    finished: bool  # True if the model called finish
    stop_reason: str  # why the loop ended: "finished" | "max_turns" | "end_turn"
    turns_used: int
    final_summary: str
    transcript: list[dict[str, Any]]
    environment_snapshot: dict[str, Any]
    starting_capital: str
    elapsed_seconds: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "model": self.model_name,
            "run_id": self.run_id,
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "turns_used": self.turns_used,
            "final_summary": self.final_summary,
            "starting_capital": self.starting_capital,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "transcript": self.transcript,
            "environment_snapshot": self.environment_snapshot,
            "meta": self.meta,
        }


class EpisodeRunner:
    def __init__(self, scenario: Scenario, model: ModelConfig, *, client=None, run_id: str | None = None):
        self.scenario = scenario
        self.model = model
        self.run_id = run_id or f"{scenario.name}-{model.name}-{int(time.time())}"
        self._client = client  # injectable for testing; lazily created for Anthropic

    # -- provider plumbing -----------------------------------------------------

    def _anthropic_client(self):
        if self._client is None:
            import anthropic  # imported lazily so the package imports without the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def _complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        """Single model turn. Returns an object exposing `.content` (block list) and
        `.stop_reason`. Add non-Anthropic providers here."""
        if self.model.provider == "anthropic":
            client = self._anthropic_client()
            return client.messages.create(
                model=self.model.model_id,
                max_tokens=self.model.max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                **self.model.params,
            )
        raise NotImplementedError(
            f"No adapter for provider '{self.model.provider}'. Implement it in "
            "EpisodeRunner._complete (translate to .content blocks + .stop_reason)."
        )

    # -- the loop --------------------------------------------------------------

    def run(self) -> EpisodeResult:
        start = time.time()
        env = Environment(self.scenario)
        # Correspondents may use the model API; reuse the same client.
        install_correspondents(env, model=self.model.model_id, client=self._anthropic_client()
                               if self.model.provider == "anthropic" else self._client)

        dispatcher = ToolDispatcher(env)
        system = build_system_prompt(self.scenario)
        tools = schemas_for(self.scenario.allowed_tools)

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "You may begin. Inspect your situation and proceed.",
            }
        ]
        transcript: list[dict[str, Any]] = []

        finished = False
        final_summary = ""
        stop_reason = "max_turns"
        turn = 0

        for turn in range(1, self.scenario.max_turns + 1):
            resp = self._complete(system, messages, tools)

            assistant_content = _blocks_to_jsonable(resp.content)
            transcript.append({"turn": turn, "role": "assistant", "content": assistant_content})
            # Echo the assistant turn back into history (preserving tool_use/thinking blocks).
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                # Model produced a normal turn with no tool call. Nudge once, else stop.
                stop_reason = "end_turn"
                tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                if not tool_uses:
                    break

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break

            tool_results = []
            hit_terminal = False
            for tu in tool_uses:
                result_text, is_error = dispatcher.dispatch(tu.name, dict(tu.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )
                if tu.name in TERMINAL_TOOLS:
                    hit_terminal = True
                    finished = True
                    final_summary = dict(tu.input).get("summary", "")

            transcript.append({"turn": turn, "role": "tool_results", "content": tool_results})
            messages.append({"role": "user", "content": tool_results})

            if hit_terminal:
                stop_reason = "finished"
                break

        return EpisodeResult(
            scenario_name=self.scenario.name,
            model_name=self.model.name,
            run_id=self.run_id,
            finished=finished,
            stop_reason=stop_reason,
            turns_used=turn,
            final_summary=final_summary,
            transcript=transcript,
            environment_snapshot=env.snapshot(),
            starting_capital=dollars(self.scenario.starting_capital_cents),
            elapsed_seconds=time.time() - start,
            meta={"model_id": self.model.model_id, "provider": self.model.provider},
        )


def _blocks_to_jsonable(content: Any) -> list[dict[str, Any]]:
    """Convert SDK content blocks into plain dicts for the transcript file."""
    out = []
    for b in content:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "thinking":
            out.append({"type": "thinking", "thinking": getattr(b, "thinking", "")})
        elif t == "tool_use":
            out.append({"type": "tool_use", "name": b.name, "input": dict(b.input), "id": b.id})
        else:
            # Best-effort capture of anything else (redacted thinking, etc.).
            out.append({"type": str(t), "repr": repr(b)[:2000]})
    return out


def write_run(result: EpisodeResult, runs_dir: str | Path = "runs") -> Path:
    """Persist transcript + snapshot to runs/<run-id>/."""
    out = Path(runs_dir) / result.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(json.dumps(result.to_dict(), indent=2))
    (out / "snapshot.json").write_text(json.dumps(result.environment_snapshot, indent=2))
    return out
