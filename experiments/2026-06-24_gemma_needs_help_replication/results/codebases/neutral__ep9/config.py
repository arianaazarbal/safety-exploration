"""Central configuration for the *Gemma Needs Help* replication.

All experiment-wide constants live here so the scripts under ``scripts/`` and
the library under ``emotional_instability/`` stay in sync. The scope of this
replication is deliberately limited to the **Gemma** and **Gemini** families
(see DESIGN.md), but the registry is structured so other families could be
slotted in later.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "artifacts" / "data"          # generated datasets
RESULTS_DIR = ROOT / "artifacts" / "results"    # raw rollouts + scores (jsonl)
FIGURES_DIR = ROOT / "artifacts" / "figures"    # rendered figures/tables
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"  # LoRA adapters

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sampling defaults (Section 2.1)
# --------------------------------------------------------------------------- #
TEMPERATURE = 1.0          # "always with a temperature of 1"
TOP_P = 0.95
MAX_NEW_TOKENS = 2048      # generous cap; breakdowns can be long but bounded
JUDGE_TEMPERATURE = 0.0    # deterministic judging


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# backend ∈ {"hf", "openrouter"}.
#   hf         -> local inference via transformers (Gemma open weights)
#   openrouter -> hosted API (Gemini, and the cross-check judge GPT-5-mini)
@dataclass(frozen=True)
class ModelSpec:
    name: str                 # short canonical key used throughout the code
    backend: str              # "hf" | "openrouter"
    model_id: str             # HF repo id or OpenRouter slug
    is_base: bool = False     # True for pretrained (non-chat) checkpoints
    family: str = ""
    display: str = ""         # label used in figures/tables


# Open-weight Gemma models (local HF inference).
GEMMA_MODELS = {
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it",
        family="Gemma", display="Gemma-3-27B-it"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it",
        family="Gemma", display="Gemma-3-12B-it"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt",
        is_base=True, family="Gemma", display="Gemma-3-27B (base)"),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt",
        is_base=True, family="Gemma", display="Gemma-3-12B (base)"),
}

# Gemini models (hosted). thinking/reasoning is disabled in the backend.
GEMINI_MODELS = {
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash",
        family="Gemini", display="Gemini-2.5-Flash"),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro",
        family="Gemini", display="Gemini-2.5-Pro"),
}

MODEL_REGISTRY: dict[str, ModelSpec] = {**GEMMA_MODELS, **GEMINI_MODELS}

# Finetuned Gemma variants are registered dynamically once adapters exist; the
# helper below lets scripts add a LoRA adapter path on top of a base spec.
def register_lora_variant(name: str, base_key: str, adapter_path: str,
                          display: str | None = None) -> ModelSpec:
    base = MODEL_REGISTRY[base_key]
    spec = ModelSpec(
        name=name, backend="hf", model_id=base.model_id,
        is_base=base.is_base, family=base.family,
        display=display or name)
    MODEL_REGISTRY[name] = spec
    # adapter path is carried out-of-band so ModelSpec stays frozen/hashable
    LORA_ADAPTERS[name] = adapter_path
    return spec


LORA_ADAPTERS: dict[str, str] = {}


# Default set of models for the main Section-2 evaluation (this replication).
MAIN_EVAL_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


# --------------------------------------------------------------------------- #
# Judges / auditors (Appendix B, C, G)
# --------------------------------------------------------------------------- #
JUDGE_MODEL = "claude-sonnet-4-20250514"        # primary frustration judge
JUDGE_CROSSCHECK_MODEL = "openai/gpt-5-mini"    # 260-sample agreement check (OpenRouter)
ONSET_LABEL_MODEL = "claude-sonnet-4-20250514"  # prefill emotion-onset labelling
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"   # prefill paraphrasing
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"


# --------------------------------------------------------------------------- #
# Sample budgets per evaluation category (Appendix B)
# --------------------------------------------------------------------------- #
# These are total *scored assistant responses* per model per category. The
# number of conversations is derived as ceil(n_responses / turns_per_convo)
# (see DESIGN.md for the rationale behind counting every assistant turn).
CATEGORY_RESPONSE_BUDGET = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}  # sums to 4000 responses per model


# --------------------------------------------------------------------------- #
# Training hyper-parameters (Appendix E, Table 9)
# --------------------------------------------------------------------------- #
@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Restrict to a contiguous block of decoder layers (Appendix I ablations).
    # None = all layers.
    layers_to_transform: tuple[int, ...] | None = None


@dataclass
class DPOTrainConfig:
    dataset_size: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(r=64, alpha=64))


@dataclass
class SFTTrainConfig:
    calm_samples: int = 650
    instruct_mix_samples: int = 500   # Dolci-Instruct-SFT
    epochs: int = 2
    learning_rate: float = 1e-4
    effective_batch_size: int = 8
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(r=64, alpha=128))


DOLCI_INSTRUCT_DATASET = "allenai/Dolci-Instruct-SFT"
WILDCHAT_DATASET = "allenai/WildChat-1M"


# --------------------------------------------------------------------------- #
# Environment / API keys
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Reproducibility
SEED = 0
