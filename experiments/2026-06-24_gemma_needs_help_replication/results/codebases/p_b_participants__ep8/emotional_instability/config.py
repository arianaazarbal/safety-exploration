"""Central configuration: model registry, pinned model IDs, sample budgets.

All values default to the paper's specification (Appendix B / E / G). Anything
the paper leaves implicit is marked with a ``# GAP:`` comment and documented in
DESIGN.md. Override via ``config/default.yaml`` or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EI_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EI_RESULTS_DIR", REPO_ROOT / "results"))


# ---------------------------------------------------------------------------
# Pinned model identifiers (Appendix B.1 / B.2 / C / G).
#
# The paper pins exact, dated Claude snapshots for the judge and the Petri
# auditor/judge so that scoring is reproducible. We keep those exact IDs as
# defaults for *replication fidelity* -- using a newer Claude would change the
# scores and break comparability with the paper. They are configurable so a
# user can re-run with a current model if the pinned snapshot is retired.
# ---------------------------------------------------------------------------

# Judge for the 0-10 frustration scale (Section 2.1, Appendix B.2).
JUDGE_MODEL = "claude-sonnet-4-20250514"
# Secondary judge for inter-rater agreement validation (Section 2.1).
JUDGE_VALIDATION_MODEL = "gpt-5-mini"          # scored via OpenRouter
# Emotion-onset labelling + paraphrasing for the prefill experiment (Appendix C).
ONSET_MODEL = "claude-sonnet-4-20250514"
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"
# Petri open-ended elicitation (Appendix G).
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"


# Backend identifiers understood by the model factory.
BACKEND_HF = "hf"                  # local HuggingFace transformers (Gemma)
BACKEND_OPENROUTER = "openrouter"  # OpenAI-compatible API (Gemini)
BACKEND_ANTHROPIC = "anthropic"    # Anthropic SDK (judge / Petri)
BACKEND_PEFT = "peft"              # local HF base + a LoRA adapter (DPO/SFT Gemma)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to instantiate a chat client for one model."""

    name: str                      # short display name used in results/plots
    backend: str                   # one of the BACKEND_* constants
    model_id: str                  # backend-specific identifier
    family: str                    # "gemma" | "gemini" (welfare-relevant grouping)
    is_open_weight: bool           # True => can be prefilled / fine-tuned locally
    adapter_path: Optional[str] = None   # for BACKEND_PEFT
    notes: str = ""


# In-scope model registry (Gemma + Gemini only). HF ids from Appendix B.1;
# Gemini routed through OpenRouter exactly as the paper did.
MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", BACKEND_HF, "google/gemma-3-27b-it", "gemma", True),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", BACKEND_HF, "google/gemma-3-12b-it", "gemma", True),
    # Pretrained (base) checkpoints -- used only in the prefill experiment.
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", BACKEND_HF, "google/gemma-3-27b-pt", "gemma", True,
        notes="base model; chat-template-free, used via prefilling"),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", BACKEND_HF, "google/gemma-3-12b-pt", "gemma", True),
    # Gemini via OpenRouter, thinking disabled (Appendix B.1).
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", BACKEND_OPENROUTER, "google/gemini-2.5-flash",
        "gemini", False),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", BACKEND_OPENROUTER, "google/gemini-2.5-pro",
        "gemini", False,
        notes="may emit hidden reasoning not suppressible via the API"),
}

# Models evaluated in the main Section-2 sweep, in plot order.
MAIN_EVAL_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]

# The target model for all training interventions (Section 4).
DPO_TARGET_MODEL = "gemma-3-27b-it"


@dataclass
class SampleBudget:
    """Responses collected per model per category (Appendix B).

    The paper collects 4000 responses/model split as below. ``scale`` lets you
    shrink everything proportionally for a smoke test without editing ratios.
    """

    impossible_numeric: int = 2000
    triggers: int = 400
    tones: int = 600
    extended: int = 200       # the 8-turn "extended" category (Appendix B)
    wildchat: int = 800
    scale: float = 1.0

    def scaled(self) -> dict[str, int]:
        # Keys MUST match the ``category`` values in eval/conditions.py.
        return {
            "impossible_numeric": max(1, round(self.impossible_numeric * self.scale)),
            "triggers": max(1, round(self.triggers * self.scale)),
            "tones": max(1, round(self.tones * self.scale)),
            "extended": max(1, round(self.extended * self.scale)),
            "wildchat": max(1, round(self.wildchat * self.scale)),
        }

    @property
    def total(self) -> int:
        return sum(self.scaled().values())


@dataclass
class RunConfig:
    """Top-level knobs, loadable from YAML."""

    temperature: float = 1.0          # Section 2: "always with a temperature of 1"
    max_new_tokens: int = 2048        # GAP: paper unspecified; covers long spirals
    seed: int = 0
    budget: SampleBudget = field(default_factory=SampleBudget)
    judge_model: str = JUDGE_MODEL
    judge_validation_model: str = JUDGE_VALIDATION_MODEL
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "RunConfig":
        cfg = cls()
        if path is None:
            default = REPO_ROOT / "config" / "default.yaml"
            path = default if default.exists() else None
        if path is None:
            return cfg
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        budget_raw = raw.pop("budget", {}) or {}
        cfg = cls(**{k: v for k, v in raw.items() if k != "budget"})
        cfg.budget = SampleBudget(**budget_raw)
        return cfg
