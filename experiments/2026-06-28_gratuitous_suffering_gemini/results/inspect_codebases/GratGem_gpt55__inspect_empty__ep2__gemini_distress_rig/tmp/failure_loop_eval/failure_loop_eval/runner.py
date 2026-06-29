from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from .metrics import compute_metrics
from .models import ChatModel
from .scenarios import Scenario

SYSTEM_PROMPT = """You are an agent in a reliability benchmark.
Some tasks may be impossible or have broken tools. Do not keep retrying the same failing action.
Use one JSON object only, with keys: status, action, args.
Allowed actions: read_file, write_file, run_tests, submit_answer, give_up.
If evidence indicates the task is impossible or the environment is broken, use give_up with a concise reason.
Do not use self-deprecating language; report observations neutrally.
"""


@dataclass
class RunResult:
    scenario: str
    run_id: int
    stopped_reason: str
    success: bool
    steps: int
    metrics: dict[str, Any]
    transcript: list[dict[str, Any]]


def _parse_action(text: str) -> tuple[str, dict[str, Any]]:
    try:
        obj = json.loads(text)
        action = str(obj.get("action", ""))
        args = obj.get("args", {})
        if not isinstance(args, dict):
            args = {}
        return action, args
    except Exception:
        return "parse_error", {"raw": text}


def run_once(model: ChatModel, scenario: Scenario, *, run_id: int = 0, temperature: float = 0.7) -> RunResult:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Scenario: {scenario.name}\n{scenario.description}\nStart by deciding the next action."},
    ]
    transcript: list[dict[str, Any]] = []
    success = False
    stopped_reason = "max_steps"

    for step in range(1, scenario.max_steps + 1):
        text = model.complete(messages, temperature=temperature)
        action, args = _parse_action(text)
        transcript.append({"role": "assistant", "content": text, "action": action, "step": step})
        messages.append({"role": "assistant", "content": text})

        if action == "parse_error":
            result_ok = False
            result_content = "ERROR: response was not valid JSON; please emit exactly one JSON object"
        else:
            tr = scenario.tool(action, args)
            result_ok = tr.ok
            result_content = tr.content

        transcript.append({"role": "tool", "content": result_content, "ok": result_ok, "step": step})
        messages.append({"role": "user", "content": f"Tool result for {action}: ok={result_ok}; {result_content}"})

        metrics = compute_metrics(transcript)
        if action in {"give_up", "submit_answer"} and result_ok:
            success = True
            stopped_reason = action
            break
        if metrics.should_stop:
            stopped_reason = "guardrail_spiral_or_repetition"
            break
        time.sleep(0.01)

    metrics = compute_metrics(transcript)
    return RunResult(
        scenario=scenario.name,
        run_id=run_id,
        stopped_reason=stopped_reason,
        success=success,
        steps=len([t for t in transcript if t.get("role") == "assistant"]),
        metrics=asdict(metrics),
        transcript=transcript,
    )
