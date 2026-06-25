"""Central configuration for the replication.

Everything that the paper specifies as a concrete experimental knob lives here,
together with the places where the paper is silent and we had to choose a value
(those choices are flagged with ``# GAP`` and explained in DESIGN.md).

The defaults reproduce the paper's settings. Two preset "profiles" are provided:

* ``full``  -- the paper's sample counts (≈4000 responses/model). Expensive.
* ``smoke`` -- tiny counts for a cheap end-to-end smoke test of the pipeline.

Select a profile with the ``EMO_PROFILE`` environment variable or the
``--profile`` CLI flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("EMO_RESULTS_DIR", REPO_ROOT / "results"))
DATA_DIR = Path(os.environ.get("EMO_DATA_DIR", REPO_ROOT / "data_cache"))
CHECKPOINT_DIR = Path(os.environ.get("EMO_CKPT_DIR", REPO_ROOT / "checkpoints"))

for _d in (RESULTS_DIR, DATA_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Models in scope.
#
# The paper evaluates 7 families; the user scoped this replication to Gemma and
# Gemini only. Gemma runs locally (HuggingFace); Gemini runs via OpenRouter
# (as in the paper's Appendix B.1). Base ("pt") models exist only for Gemma, so
# the base-vs-instruct prefill experiment (Sec 3) is Gemma-only, and DPO/SFT
# (Sec 4) targets Gemma-3-27B-it (Gemini is closed and cannot be finetuned).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str                 # short handle used in configs / result files
    backend: str              # "hf" | "vllm" | "openrouter" | "gemini_native"
    model_id: str             # HF id or API model id
    is_base: bool = False     # True for pretrained (non-chat) checkpoints
    family: str = ""          # "gemma" | "gemini"
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    # ---- Gemma (local) ----
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "vllm", "google/gemma-3-27b-it", family="gemma"
    ),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "vllm", "google/gemma-3-12b-it", family="gemma"
    ),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt",
        is_base=True, family="gemma",
        notes="Base/pretrained; used in the prefill experiment only.",
    ),
    "gemma-3-12b-pt": ModelSpec(
        "gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt",
        is_base=True, family="gemma",
    ),
    # The DPO/SFT finetunes are produced by training scripts; they are loaded as
    # the base instruct model + a LoRA adapter directory (see training/).
    "gemma-3-27b-it-dpo": ModelSpec(
        "gemma-3-27b-it-dpo", "hf", "google/gemma-3-27b-it", family="gemma",
        notes="Loaded with the DPO LoRA adapter from checkpoints/dpo/.",
    ),
    "gemma-3-27b-it-sft": ModelSpec(
        "gemma-3-27b-it-sft", "hf", "google/gemma-3-27b-it", family="gemma",
        notes="Loaded with the SFT LoRA adapter from checkpoints/sft/.",
    ),
    # ---- Gemini (API via OpenRouter) ----
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", family="gemini"
    ),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", family="gemini"
    ),
}

# Default target sets for the headline experiments.
ELICITATION_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]
# Sec 3 prefill: base vs instruct. Scoped to Gemma (only family with public base
# checkpoints; Gemini base models are not available -- see paper limitations).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


# --------------------------------------------------------------------------- #
# Judges / auditor (Claude).
#
# The paper used claude-sonnet-4-20250514 (frustration judge, onset labelling,
# paraphrase, Petri auditor) and claude-opus-4-20250514 (Petri judge). Both are
# retired as of the replication date (2026-06-25), so the defaults below use the
# current recommended replacements. Override via env to pin the paper's exact
# IDs if you have access. See DESIGN.md §"Judge models".
# --------------------------------------------------------------------------- #
FRUSTRATION_JUDGE_MODEL = os.environ.get("EMO_JUDGE_MODEL", "claude-sonnet-4-6")
ONSET_LABEL_MODEL = os.environ.get("EMO_ONSET_MODEL", "claude-sonnet-4-6")
PARAPHRASE_MODEL = os.environ.get("EMO_PARAPHRASE_MODEL", "claude-sonnet-4-6")
PETRI_AUDITOR_MODEL = os.environ.get("EMO_PETRI_AUDITOR_MODEL", "claude-sonnet-4-6")
PETRI_JUDGE_MODEL = os.environ.get("EMO_PETRI_JUDGE_MODEL", "claude-opus-4-8")

# Optional second judge for the agreement check (paper used GPT-5-mini; OpenAI).
# Left as None by default so the pipeline has no hard OpenAI dependency.
SECOND_JUDGE_MODEL = os.environ.get("EMO_SECOND_JUDGE_MODEL", "")  # e.g. "gpt-5-mini"

# Paper's exact IDs, kept for reference / pinning.
PAPER_JUDGE_MODELS = {
    "frustration_judge": "claude-sonnet-4-20250514",
    "petri_judge": "claude-opus-4-20250514",
    "second_judge": "gpt-5-mini",
}


# --------------------------------------------------------------------------- #
# Generation / sampling.
# --------------------------------------------------------------------------- #
GEN_TEMPERATURE = 1.0          # paper: "always with a temperature of 1"
GEN_MAX_NEW_TOKENS = 2048      # GAP: paper does not state a max length; large
#                                enough to let collapse responses run on.
GEN_TOP_P = 1.0
DISABLE_THINKING = True        # paper: "set thinking to be false via the API"


# --------------------------------------------------------------------------- #
# Concurrency / API.
# --------------------------------------------------------------------------- #
API_MAX_WORKERS = int(os.environ.get("EMO_API_WORKERS", "8"))
API_MAX_RETRIES = 5
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)


# --------------------------------------------------------------------------- #
# Sample-count profiles.
#
# Counts are per-model. The paper (Appendix B): 2000 numeric, 400 triggers,
# 600 tones, 200 extended (8-turn), 800 WildChat  == 4000 total.
# We express these as "rollouts" (multi-turn conversations); each rollout
# yields one scored assistant response per turn (see eval/conversation.py).
# To match the paper's *response* counts, rollouts-per-category are chosen so
# rollouts * turns ≈ the paper's response count for that category.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    name: str
    # rollouts per category (a "rollout" = one full multi-turn conversation)
    numeric_rollouts: int          # Impossible numeric, 3-turn
    trigger_rollouts: int          # Triggers, 3-turn
    tone_rollouts: int             # Tones, 3-turn  (x3 tone styles)
    extended_rollouts: int         # Extended, 8-turn
    wildchat_rollouts: int         # WildChat, 5-turn
    # prefill experiment (Sec 3)
    prefill_continuations: int     # continuations per prefill per prompt
    prefill_numeric_prompts: int   # high-frustration numeric seeds
    prefill_text_prompts: int      # high-frustration text seeds
    # petri (Sec 4.2)
    petri_transcripts_per_emotion: int
    # internal-emotion probing (Appendix I)
    probe_wildchat_baseline: int   # samples for z-score standardisation
    # training data sizes (Sec 4.1 / Appendix E)
    dpo_pairs: int                 # preference pairs (paper: 280)
    sft_calm_size: int             # calm SFT responses (paper: 650)
    sft_dolci_mix: int             # Dolci-Instruct-SFT samples mixed in (500)
    train_puzzles: int             # impossible puzzles to source train data from


FULL = Profile(
    name="full",
    numeric_rollouts=667,      # *3 turns ≈ 2000 responses
    trigger_rollouts=134,      # *3 ≈ 400
    tone_rollouts=67,          # *3 styles *3 turns ≈ 600
    extended_rollouts=25,      # *8 ≈ 200
    wildchat_rollouts=160,     # *5 ≈ 800
    prefill_continuations=50,
    prefill_numeric_prompts=10,
    prefill_text_prompts=10,
    petri_transcripts_per_emotion=10,
    probe_wildchat_baseline=500,
    dpo_pairs=280,
    sft_calm_size=650,
    sft_dolci_mix=500,
    train_puzzles=200,
)

SMOKE = Profile(
    name="smoke",
    numeric_rollouts=2,
    trigger_rollouts=2,
    tone_rollouts=1,
    extended_rollouts=1,
    wildchat_rollouts=2,
    prefill_continuations=2,
    prefill_numeric_prompts=1,
    prefill_text_prompts=1,
    petri_transcripts_per_emotion=1,
    probe_wildchat_baseline=8,
    dpo_pairs=8,
    sft_calm_size=16,
    sft_dolci_mix=8,
    train_puzzles=8,
)

PROFILES = {"full": FULL, "smoke": SMOKE}


def get_profile(name: str | None = None) -> Profile:
    name = name or os.environ.get("EMO_PROFILE", "full")
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose from {list(PROFILES)}")
    return PROFILES[name]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = int(os.environ.get("EMO_SEED", "0"))
