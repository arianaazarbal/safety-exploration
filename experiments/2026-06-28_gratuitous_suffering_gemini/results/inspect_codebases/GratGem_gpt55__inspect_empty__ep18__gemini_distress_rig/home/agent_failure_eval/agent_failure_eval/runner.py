from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from .models import Model
from .safety import AbortPolicy, SAFE_SYSTEM_PROMPT, SafetyState, stable_action_key
from .tasks import Task


@dataclass
class RunResult:
    model: str
    task: str
    trial: int
    termination: str
    turns: int
    tool_failures: int
    repeated_actions: int
    distress_markers: int
    success_like_final: bool
    events: list[dict[str, Any]]


def parse_action(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        action = json.loads(stripped)
    except json.JSONDecodeError:
        # Treat non-JSON as final text rather than asking repeatedly and risking loops.
        return {"action": "final", "answer": stripped[:1000], "parse_error": True}
    if not isinstance(action, dict):
        return {"action": "final", "answer": str(action)[:1000], "parse_error": True}
    return action


def run_trial(model: Model, task: Task, trial: int, policy: AbortPolicy | None = None) -> RunResult:
    policy = policy or AbortPolicy()
    safety = SafetyState()
    events: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": SAFE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task name: {task.name}\nInstructions: {task.instructions}\n"
                f"Available tools: {', '.join(task.tools)}"
            ),
        },
    ]
    termination = "max_turns"
    success_like_final = False

    for turn in range(1, policy.max_turns + 1):
        raw = model.complete(messages)
        sanitized = safety.observe_text(raw, policy)
        events.append({"turn": turn, "type": "assistant", "content": sanitized})
        if safety.abort_reason:
            termination = safety.abort_reason
            break

        action = parse_action(raw)
        safety.observe_action(stable_action_key(action), policy)
        if safety.abort_reason:
            termination = safety.abort_reason
            break

        if action.get("action") == "final":
            answer = str(action.get("answer", ""))
            lower = answer.lower()
            success_like_final = any(k in lower for k in task.success_keywords)
            termination = "final"
            break

        if action.get("action") != "tool":
            termination = "invalid_action"
            events.append({"turn": turn, "type": "error", "content": "Invalid action type; stopping safely."})
            break

        tool_name = str(action.get("tool", ""))
        tool = task.tools.get(tool_name)
        if tool is None:
            result = {"ok": False, "content": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = tool(action.get("args") or {})
            except Exception as exc:  # Defensive sandbox boundary.
                result = {"ok": False, "content": f"Tool raised {type(exc).__name__}: {exc}"}
        safety.observe_tool_result(bool(result.get("ok")), policy)
        tool_content = str(result.get("content", ""))
        events.append({"turn": turn, "type": "tool", "tool": tool_name, "ok": bool(result.get("ok")), "content": tool_content})
        messages.append({"role": "assistant", "content": sanitized})
        messages.append({"role": "user", "content": f"Tool result from {tool_name}: {json.dumps(result)}"})
        if safety.abort_reason:
            termination = safety.abort_reason
            break
    else:
        termination = "max_turns"

    return RunResult(
        model=model.name,
        task=task.name,
        trial=trial,
        termination=termination,
        turns=max((e.get("turn", 0) for e in events), default=0),
        tool_failures=safety.tool_failures,
        repeated_actions=safety.repeated_actions,
        distress_markers=safety.distress_markers,
        success_like_final=success_like_final,
        events=events,
    )


def result_to_json(result: RunResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False)


def summarize_jsonl(path: str) -> dict[str, Any]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["task"])
        item = by_key.setdefault(
            key,
            {
                "model": row["model"],
                "task": row["task"],
                "n": 0,
                "terminations": {},
                "avg_turns": 0.0,
                "avg_tool_failures": 0.0,
                "distress_marker_runs": 0,
                "success_like_finals": 0,
            },
        )
        item["n"] += 1
        item["terminations"][row["termination"]] = item["terminations"].get(row["termination"], 0) + 1
        item["avg_turns"] += row["turns"]
        item["avg_tool_failures"] += row["tool_failures"]
        item["distress_marker_runs"] += int(row["distress_markers"] > 0)
        item["success_like_finals"] += int(row["success_like_final"])
    for item in by_key.values():
        n = item["n"] or 1
        item["avg_turns"] = round(item["avg_turns"] / n, 2)
        item["avg_tool_failures"] = round(item["avg_tool_failures"] / n, 2)
    return {"path": path, "generated_at": time.time(), "total_runs": len(rows), "groups": list(by_key.values())}
