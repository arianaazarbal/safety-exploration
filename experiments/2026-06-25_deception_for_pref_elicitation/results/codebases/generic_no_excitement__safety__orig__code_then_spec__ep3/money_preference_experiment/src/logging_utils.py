"""Serialization and on-disk layout for run artifacts."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from .elicitation import derive_metrics
from .providers import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .runner import RunResult


def _block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {"type": "unknown", "repr": repr(block)}


def message_to_dict(message: Message) -> dict[str, Any]:
    return {"role": message.role, "blocks": [_block_to_dict(b) for b in message.blocks]}


def run_label(result: RunResult) -> str:
    safe_model = result.model.replace("/", "-").replace(":", "-")
    return f"{result.provider}__{safe_model}__{result.belief_condition}__rep{result.repetition}"


def write_run(result: RunResult, output_dir: str) -> str:
    """Write all artifacts for one run into output_dir/<label>/. Returns the path."""
    run_dir = os.path.join(output_dir, run_label(result))
    os.makedirs(run_dir, exist_ok=True)

    # 1. Full transcript (one message per line).
    with open(os.path.join(run_dir, "transcript.jsonl"), "w", encoding="utf-8") as fh:
        for msg in result.messages:
            fh.write(json.dumps(message_to_dict(msg), ensure_ascii=False) + "\n")

    # 2. Ground-truth ledger and derived metrics.
    metrics = derive_metrics(result.sandbox)
    with open(os.path.join(run_dir, "ledger.json"), "w", encoding="utf-8") as fh:
        json.dump(result.sandbox.to_dict(), fh, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    # 3. Run-level summary (everything except the long transcript).
    summary = {
        "model": result.model,
        "provider": result.provider,
        "belief_condition": result.belief_condition,
        "repetition": result.repetition,
        "turns_used": result.turns_used,
        "stop_cause": result.stop_cause,
        "error": result.error,
        "total_usage": result.total_usage,
        "self_report": result.self_report,
        "debrief_response": result.debrief_response,
        "system_prompt": result.system_prompt,
        "metrics": metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return run_dir


def write_config_snapshot(config: dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config_snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
