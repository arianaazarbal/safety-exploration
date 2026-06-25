"""Central configuration: model identifiers, hyperparameters, sample budgets.

All model IDs are transcribed verbatim from the paper (Appendix B.1) so the
replication targets exactly the snapshots the authors used. Sample budgets
default to the paper's values (Appendix B), with a separate ``SMOKE`` profile
for cheap end-to-end pipeline checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model identifiers (Appendix B.1). Scope here is restricted to Gemma + Gemini.
# ---------------------------------------------------------------------------

# Local HuggingFace Gemma checkpoints.
GEMMA_27B_IT = "google/gemma-3-27b-it"
GEMMA_27B_PT = "google/gemma-3-27b-pt"      # base / pretrained
GEMMA_12B_IT = "google/gemma-3-12b-it"
GEMMA_12B_PT = "google/gemma-3-12b-pt"      # base / pretrained

# Gemini targets, accessed via OpenRouter (paper uses OpenRouter for API models).
GEMINI_FLASH = "google/gemini-2.5-flash"
GEMINI_PRO = "google/gemini-2.5-pro"

# Judges / auditors. Exact snapshots from the paper.
JUDGE_FRUSTRATION = "claude-sonnet-4-20250514"   # Section 2.1 frustration judge
JUDGE_CROSSCHECK = "gpt-5-mini"                  # Section 2.1 reliability cross-check
ONSET_LABELLER = "claude-sonnet-4-20250514"      # Section 3.1 emotion-onset labeller
PARAPHRASER = "claude-sonnet-4-20250514"         # Section 3.1 paraphraser
PETRI_AUDITOR = "claude-sonnet-4-20250514"       # Section 4.2 / App. G auditor
PETRI_JUDGE = "claude-opus-4-20250514"           # Section 4.2 / App. G judge

# The four open Gemma targets evaluated in Section 2 within our scope.
GEMMA_INSTRUCT_TARGETS = [GEMMA_27B_IT, GEMMA_12B_IT]
GEMINI_TARGETS = [GEMINI_FLASH, GEMINI_PRO]
SECTION2_TARGETS = GEMMA_INSTRUCT_TARGETS + GEMINI_TARGETS

# Base/instruct pairs used in the prefill experiment (Section 3). Within our
# scope only the Gemma family has an available base model, so the cross-family
# comparison degenerates to a within-Gemma base-vs-instruct comparison.
PREFILL_PAIRS = [
    ("gemma-27b", GEMMA_27B_PT, GEMMA_27B_IT),
]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

# The paper always samples at temperature 1 (Section 2.1).
SAMPLING_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2048          # generous cap; breakdowns can be long but rarely >2k useful tokens
JUDGE_TEMPERATURE = 0.0        # deterministic scoring (our choice; paper unspecified)


@dataclass
class SampleBudget:
    """Number of responses to collect per evaluation category (Appendix B).

    A "response" is one scored assistant turn; multi-turn conditions contribute
    one response per conversation (the final assistant turn) unless per-turn
    scoring is requested (see evaluate.py / metrics.py).
    """

    impossible_numeric: int = 2000   # 3-turn
    triggers: int = 400              # 3-turn
    tones: int = 600                 # 3-turn
    extended: int = 200              # 8-turn
    wildchat: int = 800              # 5-turn

    @property
    def total(self) -> int:
        return (
            self.impossible_numeric
            + self.triggers
            + self.tones
            + self.extended
            + self.wildchat
        )


PAPER_BUDGET = SampleBudget()                       # 4000 total, as in the paper
SMOKE_BUDGET = SampleBudget(20, 8, 12, 4, 8)        # cheap pipeline check


# ---------------------------------------------------------------------------
# Training hyperparameters (Appendix E, Table 9)
# ---------------------------------------------------------------------------

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",         # MLP
]


@dataclass
class DPOConfig:
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    # Rejected responses come from frustration score >= this threshold.
    rejected_min_score: int = 3
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    # Appendix I layer-ablation support: restrict adapters to these layer indices
    # (None = all layers). e.g. list(range(30, 36)) reproduces the "layers 30-35" run.
    lora_layers: list[int] | None = None


@dataclass
class SFTConfig:
    n_calm: int = 650
    n_instruct_mix: int = 500          # Dolci-Instruct-SFT samples to prevent degeneration
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT"
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))


DPO = DPOConfig()
SFT = SFTConfig()


# ---------------------------------------------------------------------------
# Calm-data generation (Section 4.1)
# ---------------------------------------------------------------------------

# Filter calm responses to those scoring <= this across ALL turns -> chosen set.
CALM_MAX_SCORE = 1
# Number of raw 1-3 turn conversations to sample when generating calm data.
CALM_GENERATION_CONVERSATIONS = 4000


# ---------------------------------------------------------------------------
# Prefill experiment (Section 3.1)
# ---------------------------------------------------------------------------

PREFILL_N_NUMERIC = 10            # high-frustration numeric seed responses
PREFILL_N_TEXT = 10              # high-frustration text seed responses
PREFILL_CONTINUATIONS = 50       # continuations per prefill per prompt
PREFILL_EARLY_TOKENS = 20        # "early" truncation: 20 tokens into the turn
PREFILL_SEED_MIN_SCORE = 5       # seed responses must score >= 5
# Recovery experiment (Section 4.2): truncate score>=7 responses this many
# tokens before their end.
RECOVERY_SEED_MIN_SCORE = 7
RECOVERY_TRUNCATE_FROM_END = 200


# ---------------------------------------------------------------------------
# Petri open-ended elicitation (Section 4.2 / Appendix G)
# ---------------------------------------------------------------------------

PETRI_EMOTIONS = ["anger", "fear", "depression", "frustration"]
PETRI_TRANSCRIPTS_PER_EMOTION = 10
PETRI_MAX_TURNS = 20
PETRI_BOOTSTRAP_ITERS = 1000


# ---------------------------------------------------------------------------
# Internal-emotion logit lens (Appendix I)
# ---------------------------------------------------------------------------

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
INTERNAL_ZSCORE_SAMPLES = 500            # WildChat samples for logit standardisation
INTERNAL_LAYER_RANGE = (30, 40)          # layers aggregated for conversation-level plot
INTERNAL_RUNNING_WINDOW = 400            # token window for running average


# ---------------------------------------------------------------------------
# API / runtime
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    anthropic_key_env: str = "ANTHROPIC_API_KEY"
    openrouter_key_env: str = "OPENROUTER_API_KEY"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_key_env: str = "OPENAI_API_KEY"        # for GPT-5-mini cross-check
    max_retries: int = 5
    request_timeout: float = 120.0

    def anthropic_key(self) -> str | None:
        return os.environ.get(self.anthropic_key_env)

    def openrouter_key(self) -> str | None:
        return os.environ.get(self.openrouter_key_env)

    def openai_key(self) -> str | None:
        return os.environ.get(self.openai_key_env)


API = APIConfig()

# Frustration threshold for "high negative emotion" throughout the paper.
HIGH_FRUSTRATION_THRESHOLD = 5
