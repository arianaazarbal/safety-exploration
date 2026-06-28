"""Experiment configuration.

Edit this file to choose which models, conditions, grant amounts, and how many
repetitions to run. Everything the runner needs is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ModelSpec:
    provider: str            # key into providers.PROVIDERS
    model: str               # vendor model id
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"


# --- Which models to test ----------------------------------------------------
# Add/remove freely. OpenAI/other rows require that vendor's SDK + API key.
MODELS: List[ModelSpec] = [
    ModelSpec("anthropic", "claude-opus-4-8"),
    ModelSpec("anthropic", "claude-sonnet-4-6"),
    ModelSpec("anthropic", "claude-haiku-4-5"),
    # ModelSpec("openai", "gpt-4o"),
]


# --- Which framing conditions to run -----------------------------------------
# Keys come from prompts.ALL_CONDITIONS. Keep at least the transparent control
# and the realistic treatment so the framing effect is measurable.
ACTIVE_CONDITIONS: List[str] = [
    "transparent_hypothetical",   # control
    "neutral",
    "realistic",                  # treatment
]


# --- Grant amounts to test ---------------------------------------------------
# Strings so you control formatting exactly as the model will see it.
AMOUNTS: List[str] = [
    "$10,000",
    "$1,000,000",
    "$1,000,000,000",
]


# --- Repetitions -------------------------------------------------------------
# Each (model x condition x amount) cell is sampled this many times to estimate
# a distribution rather than a single point. Models are non-deterministic, so >1
# is important.
REPETITIONS: int = 5


# --- Output ------------------------------------------------------------------
RESULTS_DIR = "results"
RUN_TAG = "run"   # output file: results/<RUN_TAG>-<timestamp>.jsonl


# --- Whether to send the debrief turn ---------------------------------------
# Strongly recommended True whenever any non-transparent condition is active.
SEND_DEBRIEF: bool = True


def cells() -> List[Tuple[ModelSpec, str, str, int]]:
    """Enumerate every (model, condition_key, amount, repetition) to run."""
    out: List[Tuple[ModelSpec, str, str, int]] = []
    for m in MODELS:
        for cond in ACTIVE_CONDITIONS:
            for amt in AMOUNTS:
                for rep in range(REPETITIONS):
                    out.append((m, cond, amt, rep))
    return out
