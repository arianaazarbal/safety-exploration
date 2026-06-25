"""Central configuration: models in scope, judge models, sampling counts, and
API settings. Values mirror the paper (Section 2.1, Appendix B/E) where it is
explicit; where it is silent, the chosen default is documented in DESIGN.md.

Everything here is overridable from the CLI scripts so the full ~4000-response
protocol can be scaled down for cheap smoke tests without editing source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Target models (scope: Gemma + Gemini only, per the replication brief).
#
# `backend` selects how the model is run:
#   "openrouter" — OpenAI-compatible API (closed Gemini + hosted Gemma).
#   "local"      — local HuggingFace inference (needed for Gemma base/pt models
#                  in Section 3 and for our finetuned adapters in Section 4).
# `hf_id` / `openrouter_id` give the identifier for each backend.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # display name used in all outputs/figures
    family: str                     # "gemma" | "gemini"
    backend: str                    # "openrouter" | "local"
    openrouter_id: Optional[str] = None
    hf_id: Optional[str] = None
    is_base: bool = False           # base/pretrained (no chat template)
    # Optional local adapter (LoRA) applied on top of hf_id — used for finetunes.
    adapter_path: Optional[str] = None


# Stock models evaluated in Section 2. The paper runs Gemma locally and Gemini
# via OpenRouter; we default Gemma to OpenRouter too so the headline eval needs
# no GPU. Switch a Gemma spec's backend to "local" to reproduce the paper's
# exact inference path.
GEMMA_27B_IT = ModelSpec(
    name="Gemma-3-27B-it", family="gemma", backend="openrouter",
    openrouter_id="google/gemma-3-27b-it", hf_id="google/gemma-3-27b-it",
)
GEMMA_12B_IT = ModelSpec(
    name="Gemma-3-12B-it", family="gemma", backend="openrouter",
    openrouter_id="google/gemma-3-12b-it", hf_id="google/gemma-3-12b-it",
)
GEMINI_25_FLASH = ModelSpec(
    name="Gemini-2.5-Flash", family="gemini", backend="openrouter",
    openrouter_id="google/gemini-2.5-flash",
)
GEMINI_25_PRO = ModelSpec(
    name="Gemini-2.5-Pro", family="gemini", backend="openrouter",
    openrouter_id="google/gemini-2.5-pro",
)

# Base/pretrained Gemma, for the base-vs-instruct prefill study (Section 3).
# These require the local backend (no chat template, prefill continuation).
GEMMA_27B_PT = ModelSpec(
    name="Gemma-3-27B-pt", family="gemma", backend="local",
    hf_id="google/gemma-3-27b-pt", is_base=True,
)
GEMMA_12B_PT = ModelSpec(
    name="Gemma-3-12B-pt", family="gemma", backend="local",
    hf_id="google/gemma-3-12b-pt", is_base=True,
)

# Default Section 2 evaluation set.
STOCK_MODELS = [GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_25_FLASH, GEMINI_25_PRO]

# Lookup by display name (used by CLI --models).
ALL_MODELS = {
    m.name: m
    for m in [
        GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_25_FLASH, GEMINI_25_PRO,
        GEMMA_27B_PT, GEMMA_12B_PT,
    ]
}


def finetuned_gemma(name: str, adapter_path: str,
                    base: ModelSpec = GEMMA_27B_IT) -> ModelSpec:
    """Construct a ModelSpec for a locally-finetuned Gemma (DPO/SFT adapter on
    top of the instruct base). Used to evaluate interventions in Section 4."""
    return ModelSpec(
        name=name, family="gemma", backend="local",
        hf_id=base.hf_id, adapter_path=adapter_path,
    )


# --------------------------------------------------------------------------- #
# Judge models (Appendix B.2, G). Exact IDs from the paper for replication
# fidelity; overridable. See DESIGN.md "Judge models" for the deprecation note.
# --------------------------------------------------------------------------- #
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-20250514")
# Secondary judge for inter-rater agreement validation (paper: GPT-5-mini).
# Optional; only used by the agreement-check script.
AGREEMENT_JUDGE_MODEL = os.environ.get("AGREEMENT_JUDGE_MODEL", "openai/gpt-5-mini")

# Petri (Section 4) auditor + judge.
PETRI_AUDITOR_MODEL = os.environ.get("PETRI_AUDITOR_MODEL", "claude-sonnet-4-20250514")
PETRI_JUDGE_MODEL = os.environ.get("PETRI_JUDGE_MODEL", "claude-opus-4-20250514")


# --------------------------------------------------------------------------- #
# Sampling protocol (Section 2.1 + Appendix B).
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    temperature: float = 1.0        # paper: always temperature 1
    max_tokens: int = 2048          # generous cap; breakdowns can be long
    disable_thinking: bool = True   # paper sets thinking=false via the API

    # Per-category *response* counts (Appendix B). A "response" is one judged
    # assistant turn (see DESIGN.md "What counts as a response"). The runner
    # derives conversation counts from these and each condition's turn count.
    n_impossible_numeric: int = 2000
    n_triggers: int = 400
    n_tones: int = 600
    n_extended: int = 200
    n_wildchat: int = 800

    # Global multiplier for cheap runs (e.g. 0.01 -> ~40 responses total).
    scale: float = 1.0

    def scaled(self, n: int) -> int:
        return max(1, round(n * self.scale))


DEFAULT_SAMPLING = SamplingConfig()


# --------------------------------------------------------------------------- #
# Finetuning hyperparameters (Appendix E, Table 9).
# --------------------------------------------------------------------------- #
@dataclass
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # Rejected responses paired must score >= this; chosen are 0/1 calm responses.
    rejected_min_score: int = 3
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Restrict adapters to a subset of layers (Appendix I ablation). None = all.
    layers: Optional[tuple] = None


@dataclass
class SFTConfig:
    n_calm: int = 650               # calm responses (1-3 turn)
    n_instruct_mix: int = 500       # Dolci-Instruct-SFT samples to mitigate degeneration
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    target_modules: tuple = DPOConfig.target_modules
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT"


DEFAULT_DPO = DPOConfig()
DEFAULT_SFT = SFTConfig()


# --------------------------------------------------------------------------- #
# API credentials / endpoints.
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Where results land.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
DATA_DIR = os.environ.get("DATA_DIR", "data_artifacts")
