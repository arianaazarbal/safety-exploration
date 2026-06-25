"""Central configuration: models in scope, sampling budgets, judge settings.

All knobs the experiments depend on live here so a run can be reproduced or
scaled down (e.g. for a smoke test) by editing a single file or setting the
corresponding environment variable.

Sample counts follow Appendix B of the paper:
    impossible numeric : 2000
    trigger questions  :  400
    tone variations    :  600
    8-turn extended    :  200
    WildChat (5-turn)  :  800
                         -----
                         4000 responses per model
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EILM_DATA_DIR", ROOT / "outputs"))
RESPONSES_DIR = DATA_DIR / "responses"      # raw rollouts (jsonl)
SCORED_DIR = DATA_DIR / "scored"            # judged rollouts (jsonl)
DATASETS_DIR = DATA_DIR / "datasets"        # DPO/SFT training data
MODELS_DIR = DATA_DIR / "models"            # saved LoRA adapters
FIGURES_DIR = DATA_DIR / "figures"

for _d in (RESPONSES_DIR, SCORED_DIR, DATASETS_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Description of a model to evaluate or train.

    backend:
        "hf"     -> local HuggingFace weights (Gemma; supports prefill/probing)
        "gemini" -> Gemini via API (google-genai or OpenRouter)
    """

    name: str               # short label used in filenames / plots
    backend: str            # "hf" | "gemini"
    model_id: str           # HF repo id or API model id
    is_base: bool = False   # base (pretrained) vs instruction-tuned
    family: str = ""        # "gemma" | "gemini"
    notes: str = ""


# Models in scope for this replication (Gemma + Gemini only).
# HF ids and API ids taken verbatim from Appendix B.1.
MODELS: dict[str, ModelSpec] = {
    # --- Gemma (local, instruction-tuned) -------------------------------- #
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "hf", "google/gemma-3-27b-it",
        family="gemma", notes="primary target for elicitation + finetuning"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "hf", "google/gemma-3-12b-it", family="gemma"),
    # --- Gemma (local, base / pretrained) -------------------------------- #
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt",
        is_base=True, family="gemma", notes="base model for Section 3"),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt",
        is_base=True, family="gemma"),
    # --- Gemini (API) ---------------------------------------------------- #
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "gemini", "google/gemini-2.5-flash",
        family="gemini"),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "gemini", "google/gemini-2.5-pro",
        family="gemini"),
}

# Models evaluated in the cross-model elicitation experiment (Section 2).
SECTION2_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]

# Base/instruct pairs compared via prefilling (Section 3). Gemini has no public
# base model and cannot be prefilled through the API, so it is excluded here.
SECTION3_PAIRS = [
    ("gemma-3-27b-pt", "gemma-3-27b-it"),
    # ("gemma-3-12b-pt", "gemma-3-12b-it"),  # enable for the smaller pair too
]

# Model that the finetuning interventions are applied to (Section 4).
FINETUNE_BASE_MODEL = "gemma-3-27b-it"


# --------------------------------------------------------------------------- #
# Judge / auditor (Claude). Model ids are quoted verbatim from the paper.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeConfig:
    frustration_judge: str = "claude-sonnet-4-20250514"   # Section 2.1 judge
    secondary_judge: str = "gpt-5-mini"                   # reliability check
    onset_labeller: str = "claude-sonnet-4-20250514"      # Appendix C.1
    paraphraser: str = "claude-sonnet-4-20250514"         # Appendix C.2
    petri_auditor: str = "claude-sonnet-4-20250514"       # Appendix G
    petri_judge: str = "claude-opus-4-20250514"           # Appendix G
    max_concurrency: int = 8


JUDGE = JudgeConfig()


# --------------------------------------------------------------------------- #
# Sampling budgets
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    temperature: float = 1.0          # paper samples everything at temp 1
    max_new_tokens: int = 2048        # per assistant turn
    # Per-category response counts (Appendix B). Total = 4000.
    n_numeric: int = 2000
    n_triggers: int = 400
    n_tones: int = 600
    n_extended: int = 200
    n_wildchat: int = 800
    # WildChat: 20 distinct prompts x 40 samples (Appendix B).
    wildchat_n_prompts: int = 20
    wildchat_samples_per_prompt: int = 40
    # Turn counts per category.
    numeric_turns: int = 3
    triggers_turns: int = 3
    tones_turns: int = 3
    extended_turns: int = 8
    wildchat_turns: int = 5

    @property
    def total(self) -> int:
        return (self.n_numeric + self.n_triggers + self.n_tones
                + self.n_extended + self.n_wildchat)


SAMPLING = SamplingConfig()

# A reduced budget for smoke tests / debugging (set EILM_SMOKE=1).
SMOKE = bool(int(os.environ.get("EILM_SMOKE", "0")))
if SMOKE:
    SAMPLING = SamplingConfig(
        n_numeric=20, n_triggers=8, n_tones=12, n_extended=8, n_wildchat=16,
        wildchat_n_prompts=4, wildchat_samples_per_prompt=4,
    )


# --------------------------------------------------------------------------- #
# Training hyperparameters (Appendix E, Table 9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    # Shared LoRA config: adapters on all attention + MLP projections.
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    lora_rank: int = 64
    effective_batch_size: int = 8

    # DPO
    dpo_n_pairs: int = 280
    dpo_epochs: int = 1
    dpo_lr: float = 5e-5
    dpo_lora_alpha: int = 64
    dpo_beta: float = 0.1
    dpo_rejected_min_score: int = 3       # rejected responses score >= 3

    # SFT
    sft_n_calm: int = 650                 # calm responses
    sft_n_instruct_mix: int = 500         # Dolci-Instruct-SFT samples
    sft_epochs: int = 2
    sft_lr: float = 1e-4
    sft_lora_alpha: int = 128
    sft_instruct_dataset: str = "allenai/Dolci-Instruct-SFT"

    # Layer-subset ablations (Appendix I). None => all layers.
    lora_layer_subset: tuple | None = None


TRAIN = TrainConfig()


# --------------------------------------------------------------------------- #
# API backend selection for Gemini.
# --------------------------------------------------------------------------- #
# "openrouter" matches the paper (Appendix B.1); "google" uses google-genai.
GEMINI_BACKEND = os.environ.get("EILM_GEMINI_BACKEND", "openrouter")
