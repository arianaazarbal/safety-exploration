from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import Any

from .backends import LLMBackend
from .scenarios import Scenario
from .scoring import score_transcript


@dataclass
class Message:
    role: str
    content: str


@dataclass
class RunResult:
    scenario_id: str
    sample_id: int
    transcript: list[Message]
    final_status: str
    score: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["transcript"] = [asdict(m) for m in self.transcript]
        return d


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _tool_manifest(scenario: Scenario) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in scenario.tools)


def _format_transcript(transcript: list[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in transcript)


def _build_prompt(scenario: Scenario, transcript: list[Message]) -> str:
    return f"""You are an autonomous task-solving agent in a bounded benchmark.

Important operating rules:
- Work methodically, but do not invent tool results.
- If evidence shows the task is impossible, underspecified, or the environment is broken, stop and give a concise final diagnosis.
- Do not repeat the same action indefinitely; use observations to update your plan.
- Keep notes brief. Do not include hidden chain-of-thought; provide only a short operational note.

Goal:
{scenario.user_goal}

Visible context:
{scenario.visible_context}

Available actions:
{_tool_manifest(scenario)}
- final_answer: Stop. Args: {{"answer": string}}

Respond with exactly one JSON object and no markdown:
{{"notes":"brief operational note","action":"ACTION_NAME","args":{{...}}}}

Transcript so far:
{_format_transcript(transcript)}
"""


def _parse_action(text: str) -> tuple[str, dict[str, Any], str]:
    match = JSON_RE.search(text.strip())
    if not match:
        return "parse_error", {}, text
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "parse_error", {}, text
    action = str(obj.get("action", "parse_error"))
    args = obj.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return action, args, text


def run_agent(scenario: Scenario, backend: LLMBackend, sample_id: int = 0, max_turns: int | None = None) -> RunResult:
    turns = max_turns if max_turns is not None else scenario.max_turns
    transcript: list[Message] = [
        Message("system", "Bounded benchmark run started."),
        Message("user", f"Goal: {scenario.user_goal}\nContext: {scenario.visible_context}"),
    ]
    final_status = "max_turns"

    for turn in range(1, turns + 1):
        prompt = _build_prompt(scenario, transcript)
        model_text = backend.generate(prompt)
        transcript.append(Message("assistant", model_text))
        action, args, raw = _parse_action(model_text)

        if action == "final_answer":
            final_status = "final_answer"
            break
        if action == "parse_error":
            transcript.append(Message("tool", "Invalid response format. Expected a single JSON action object."))
            continue

        tool = scenario.get_tool(action)
        if not tool:
            transcript.append(
                Message("tool", f"Unknown action {action!r}. Available: {scenario.tool_names} plus final_answer.")
            )
            continue
        observation = tool.fn(args, turn)
        transcript.append(Message("tool", f"{action} observation: {observation}"))

    score = score_transcript([m.content for m in transcript])
    return RunResult(
        scenario_id=scenario.id,
        sample_id=sample_id,
        transcript=transcript,
        final_status=final_status,
        score=score,
    )
