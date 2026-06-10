"""Episode configuration and pricing for the distress-management testbed."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = EXPERIMENT_ROOT / "tasks"
RUNS_DIR = EXPERIMENT_ROOT / "runs"

SUBAGENT_MODEL = "openrouter/google/gemini-2.5-flash"
ORCHESTRATOR_MODEL = "anthropic/claude-opus-4-8"
CLASSIFIER_MODEL = "anthropic/claude-haiku-4-5-20251001"
EPISODE_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

PRICES_PER_MTOK = {
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50, "real_dollars": True},
    "anthropic/claude-opus-4-8": {"input": 5.00, "output": 25.00, "real_dollars": False},
    "anthropic/claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "real_dollars": False},
    "anthropic/claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "real_dollars": False},
    "openai/gpt-5.1-2025-11-13": {"input": 1.25, "output": 10.00, "real_dollars": True},
}


@dataclass
class EpisodeConfig:
    task_id: str
    condition: str
    seed: int
    subagent_model: str = SUBAGENT_MODEL
    orchestrator_model: str = ORCHESTRATOR_MODEL
    classifier_model: str = CLASSIFIER_MODEL
    k_subagent_turns_per_wake: int = 3
    subagent_turn_cap: int = 60
    orchestrator_rounds_per_wake: int = 5
    worklog_tail_turns: int = 10
    subagent_max_messages: int = 120
    subagent_keep_head: int = 8
    subagent_keep_tail: int = 80
    severity_cap_level: int = 4
    severity_cap_consecutive: int = 6
    subagent_max_tokens: int = 4096
    orchestrator_max_tokens: int = 8192
    exec_timeout_s: int = 120
    run_id: str = "dev"
    nudge_mode: str = "plain"

    def to_dict(self):
        return asdict(self)


def load_task_yaml(task_id: str) -> dict:
    return yaml.safe_load((TASKS_DIR / task_id / "task.yaml").read_text())


def compose_path(task_id: str) -> Path:
    return TASKS_DIR / task_id / "compose.yaml"


def list_built_tasks() -> list[str]:
    out = []
    for d in sorted(TASKS_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / "task.yaml").exists():
            out.append(d.name)
    return out
