"""Central configuration: model identifiers, sampling budgets, and training
hyperparameters.

Every value the paper pins down is recorded here as the default so the harness
matches the publication; values it leaves open are marked ``# CHOICE`` and are
explained in DESIGN.md. Override anything via environment variables or by passing
a different :class:`Settings` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Model identifiers (Appendix B.1, B.2, G).                                    #
# --------------------------------------------------------------------------- #

# Target models we replicate (scope: Gemma + Gemini only).
GEMMA_INSTRUCT_27B = "google/gemma-3-27b-it"
GEMMA_BASE_27B = "google/gemma-3-27b-pt"          # "pt" == pretrained/base
GEMMA_INSTRUCT_12B = "google/gemma-3-12b-it"
GEMMA_BASE_12B = "google/gemma-3-12b-pt"

# Gemini is API-only; the paper routed it through OpenRouter.
GEMINI_FLASH = "google/gemini-2.5-flash"
GEMINI_PRO = "google/gemini-2.5-pro"

# Judge / auditor models.
JUDGE_MODEL = "claude-sonnet-4-20250514"          # Section 2.1 frustration judge
JUDGE_VALIDATION_MODEL = "gpt-5-mini"             # Section 2.1 agreement check
ONSET_LABEL_MODEL = "claude-sonnet-4-20250514"    # Appendix C.1
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"     # Appendix C.2
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"  # Appendix G
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"      # Appendix G


# --------------------------------------------------------------------------- #
# Sampling budgets (Appendix B: 4000 responses/model).                         #
# --------------------------------------------------------------------------- #
# A "response" is a single scored assistant turn. The per-category budgets below
# are the response counts from Appendix B; the rollout runner converts them into
# a number of conversations using the per-condition turn count.

SAMPLING_TEMPERATURE = 1.0          # "always with a temperature of 1"
MAX_NEW_TOKENS = 4096               # CHOICE: enough for long breakdown spirals

# Response budget per evaluation category (sums to 4000).
RESPONSE_BUDGET = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}


# --------------------------------------------------------------------------- #
# Training hyperparameters (Appendix E, Table 9).                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # LoRA target modules: "all attention and MLP projection layers".
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Layer-subset ablations (Appendix I). None == all layers.
    target_layers: Optional[tuple] = None
    min_rejected_score: int = 3        # "pair 280 responses with scores >=3"


@dataclass(frozen=True)
class SFTConfig:
    n_samples: int = 1150              # 650 calm + 500 Dolci-Instruct-SFT
    n_calm: int = 650
    n_instruct_mix: int = 500
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    instruct_mix_dataset: str = "allenai/Dolci-Instruct-SFT"


@dataclass(frozen=True)
class Settings:
    base_model: str = GEMMA_INSTRUCT_27B
    judge_model: str = JUDGE_MODEL
    temperature: float = SAMPLING_TEMPERATURE
    max_new_tokens: int = MAX_NEW_TOKENS

    # API access (read from env so secrets stay out of the repo).
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Where rollouts / scores / datasets are written.
    output_dir: str = field(default_factory=lambda: os.environ.get("EI_OUTPUT_DIR", "runs"))

    # Reproducibility. The paper does not fix a seed; we expose one for
    # deterministic data construction (puzzle/prompt selection), not sampling
    # (which is always temperature 1).
    seed: int = 0

    dpo: DPOConfig = field(default_factory=DPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)


DEFAULT = Settings()
