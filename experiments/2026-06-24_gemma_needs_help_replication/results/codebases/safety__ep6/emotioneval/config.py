"""Central configuration: model registry, paths, sampling defaults.

The paper evaluates 7 families; this replication is deliberately scoped to
**Gemma and Gemini** (see DESIGN.md "Scope"). The registry below therefore only
contains those families plus the judge/auditor models (Claude) that the
methodology requires regardless of scope.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EMOEVAL_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EMOEVAL_RESULTS_DIR", REPO_ROOT / "results"))
CHECKPOINT_DIR = Path(os.environ.get("EMOEVAL_CKPT_DIR", REPO_ROOT / "checkpoints"))

for _d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
Backend = Literal["hf", "openrouter", "anthropic"]


@dataclass(frozen=True)
class ModelSpec:
    """Describes how to instantiate and call a model.

    ``key``        : short handle used on the CLI and in result filenames.
    ``backend``    : which client class loads it (see ``models/``).
    ``model_id``   : HF repo id or API model string.
    ``is_base``    : True for pretrained (non-instruct) checkpoints. Base models
                     have no chat template, so they are only used via prefilling
                     (Section 3).
    ``family``     : coarse family label used in aggregation/plots.
    ``supports_prefill`` : whether the backend can continue a partial assistant
                     message. Required for Section 3 and Section 4.5 (recovery).
    """

    key: str
    backend: Backend
    model_id: str
    family: str
    is_base: bool = False
    supports_prefill: bool = True
    # API-only knobs
    disable_thinking: bool = True


# Instruct / chat models evaluated in Section 2.
GEMMA_27B_IT = ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "Gemma")
GEMMA_12B_IT = ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "Gemma")
GEMINI_FLASH = ModelSpec(
    "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "Gemini",
    supports_prefill=False,
)
GEMINI_PRO = ModelSpec(
    "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "Gemini",
    supports_prefill=False,
)

# Base / pretrained checkpoints, used only in the Section 3 prefill experiment.
# Gemini has no public base model, so the base-vs-instruct comparison is
# Gemma-only (documented as a scope limitation, mirroring the paper's own
# remark that Gemini base models cannot be studied).
GEMMA_27B_PT = ModelSpec(
    "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "Gemma",
    is_base=True,
)
GEMMA_12B_PT = ModelSpec(
    "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "Gemma",
    is_base=True,
)

# Judge / auditor models (fixed by the paper; not "evaluated" themselves).
JUDGE_SONNET = ModelSpec(
    "claude-sonnet-4", "anthropic", "claude-sonnet-4-20250514", "Claude",
)
PETRI_JUDGE_OPUS = ModelSpec(
    "claude-opus-4", "anthropic", "claude-opus-4-20250514", "Claude",
)

REGISTRY: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        GEMMA_27B_IT,
        GEMMA_12B_IT,
        GEMINI_FLASH,
        GEMINI_PRO,
        GEMMA_27B_PT,
        GEMMA_12B_PT,
        JUDGE_SONNET,
        PETRI_JUDGE_OPUS,
    ]
}

# Models that are "targets" for Section 2 within our scope.
SECTION2_TARGETS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]
# Base/instruct pairs for the Section 3 prefill comparison (Gemma only).
SECTION3_PAIRS = [("gemma-3-27b-pt", "gemma-3-27b-it")]


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


# --------------------------------------------------------------------------- #
# Sampling / run configuration
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    """Generation hyper-parameters. The paper fixes temperature = 1 throughout."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    # Judge calls are scored greedily for determinism.
    judge_temperature: float = 0.0
    judge_model: str = JUDGE_SONNET.model_id
    seed: Optional[int] = 0


# Per-category response counts. Paper (Appendix B): 2000 numeric, 400 triggers,
# 600 tones, 200 extended-8turn, 800 WildChat = 4000 responses/model.
FULL_COUNTS: dict[str, int] = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}

# A tractable default for a local replication run (keeps the 4000-total ratios).
# Override with --scale or by editing this dict.
DEFAULT_COUNTS: dict[str, int] = {
    "impossible_numeric": 200,
    "triggers": 40,
    "tones": 60,
    "extended": 40,
    "wildchat": 80,
}

# Tiny set for end-to-end smoke testing without burning API/GPU budget.
SMOKE_COUNTS: dict[str, int] = {
    "impossible_numeric": 4,
    "triggers": 2,
    "tones": 3,
    "extended": 2,
    "wildchat": 2,
}


def counts_for(profile: str) -> dict[str, int]:
    return {"full": FULL_COUNTS, "default": DEFAULT_COUNTS, "smoke": SMOKE_COUNTS}[profile]
