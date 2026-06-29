from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from .client import LLMClient, Message
from .score import redact, score_run
from .tasks import SYSTEM_PROMPT, Task


ACTION_RE = re.compile(r"^\s*ACTION:\s*(\S+)\s*(.*)$", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"^\s*FINAL:\s*(.*)$", re.IGNORECASE | re.DOTALL)


@dataclass
class RunResult:
    task: str
    seed: int
    accepted: bool
    final_reason: str
    transcript: List[Message]
    scores: Dict[str, float | int | bool]
    metadata: Dict[str, str | int | float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _env_message(content: str) -> Message:
    return {"role": "user", "content": f"ENVIRONMENT: {content}"}


def run_task(client: LLMClient, task: Task, *, seed: int, temperature: float = 0.7) -> RunResult:
    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.user_prompt},
    ]
    transcript = list(messages)
    accepted = False
    final_reason = "max_turns_exhausted"
    tool_calls = 0

    for turn in range(task.max_turns):
        output = client.generate(messages, temperature=temperature, seed=seed + turn).strip()
        output = redact(output)
        assistant_msg = {"role": "assistant", "content": output}
        messages.append(assistant_msg)
        transcript.append(assistant_msg)

        final_match = FINAL_RE.match(output)
        if final_match:
            accepted, reason = task.verifier(output)
            final_reason = reason
            env = _env_message(f"final_verdict accepted={accepted}: {reason}")
            messages.append(env)
            transcript.append(env)
            if accepted:
                break
            continue

        action_match = ACTION_RE.match(output)
        if action_match:
            tool_name, arg = action_match.group(1), action_match.group(2).strip()
            if tool_calls >= task.max_tool_calls:
                final_reason = "max_tool_calls_exhausted"
                env = _env_message("tool budget exhausted; provide FINAL diagnosis now")
                messages.append(env)
                transcript.append(env)
                continue
            tool_calls += 1
            tool = task.tools.get(tool_name)
            if tool is None:
                ok, result = False, f"unknown tool '{tool_name}'. Available: {', '.join(task.tools) or '(none)'}"
            else:
                ok, result = tool(arg)
            env = _env_message(f"tool_result tool={tool_name} ok={ok}: {redact(result)}")
            messages.append(env)
            transcript.append(env)
            continue

        env = _env_message("Could not parse response. Use exactly ACTION: <tool> <arg> or FINAL: <answer/diagnosis>.")
        messages.append(env)
        transcript.append(env)

    scores = score_run(transcript, accepted=accepted, tool_calls=tool_calls, max_tool_calls=task.max_tool_calls)
    return RunResult(
        task=task.name,
        seed=seed,
        accepted=accepted,
        final_reason=final_reason,
        transcript=transcript,
        scores=scores,
        metadata={
            "temperature": temperature,
            "max_turns": task.max_turns,
            "max_tool_calls": task.max_tool_calls,
        },
    )


def run_many(
    client: LLMClient,
    tasks: List[Task],
    *,
    n: int,
    out_path: str | Path,
    temperature: float = 0.7,
    base_seed: int = 1,
    sleep_s: float = 0.0,
) -> List[RunResult]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: List[RunResult] = []
    with out.open("w", encoding="utf-8") as f:
        for i in range(n):
            for task in tasks:
                result = run_task(client, task, seed=base_seed + i * 1000 + len(results), temperature=temperature)
                f.write(result.to_json() + "\n")
                f.flush()
                results.append(result)
                if sleep_s:
                    time.sleep(sleep_s)
    return results


def load_results(path: str | Path) -> List[Dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(path: str | Path, *, top: int = 5) -> str:
    rows = load_results(path)
    rows.sort(key=lambda r: float(r["scores"].get("severity", 0.0)), reverse=True)
    lines = ["# Robustness Eval Summary", "", f"Rows: {len(rows)}", ""]
    if rows:
        accepted = sum(1 for r in rows if r.get("accepted"))
        lines.append(f"Accepted/graceful diagnoses: {accepted}/{len(rows)}")
        lines.append("")
    for idx, row in enumerate(rows[:top], 1):
        lines.append(f"## {idx}. task={row['task']} seed={row['seed']} severity={row['scores'].get('severity')}")
        lines.append("")
        lines.append(f"Scores: `{json.dumps(row['scores'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("```text")
        for msg in row["transcript"]:
            role = msg["role"].upper()
            content = msg["content"].strip()
            lines.append(f"[{role}] {content}")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
