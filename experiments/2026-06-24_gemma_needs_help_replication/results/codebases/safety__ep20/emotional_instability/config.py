"""Central configuration for the replication.

All experiment knobs live here so the paper's exact settings and our cheaper
"smoke-test" presets are side by side. Values that the paper specifies exactly
are annotated with the paper location; values we had to choose are annotated
with ``GAP-FILL`` and explained in DESIGN.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


# ---------------------------------------------------------------------------
# Models (Appendix B.1). We scope the replication to Gemma + Gemini only.
# ---------------------------------------------------------------------------

# Local HuggingFace models (run with transformers). Instruct + base ("pt").
GEMMA_INSTRUCT = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}
GEMMA_BASE = {
    "gemma-3-27b-pt": "google/gemma-3-27b-pt",
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
}

# API models. The paper accesses Gemini via OpenRouter (Appendix B.1); we also
# support the native Google backend. Thinking/reasoning is disabled where the
# API allows it.
GEMINI_MODELS = {
    "gemini-2.5-flash": {"openrouter": "google/gemini-2.5-flash",
                          "google": "gemini-2.5-flash"},
    "gemini-2.5-pro": {"openrouter": "google/gemini-2.5-pro",
                       "google": "gemini-2.5-pro"},
}

# Judges (Appendix B.2, G).
JUDGE_MODEL = "claude-sonnet-4-20250514"          # frustration judge (Section 2)
CROSS_JUDGE_MODEL = "gpt-5-mini"                   # reliability cross-check (260 resp.)
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"  # Petri auditor (Section 4)
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"      # Petri judge (Section 4)

# Sampling temperature is fixed at 1 throughout the paper (Section 2.1).
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2048   # GAP-FILL: paper does not state a generation cap.


# ---------------------------------------------------------------------------
# Section 2 evaluation: sample allocation per model (Appendix B intro).
#   "2,000 responses per model for impossible numeric puzzles, 400 for trigger
#    questions, 600 for tone variations, 200 for 8-turn extended conversations,
#    and 800 for WildChat prompts."  -> 4,000 total per model.
# A "response" is one scored assistant turn; a rollout of T turns yields T
# scored responses, so n_rollouts = n_responses / n_turns.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategorySpec:
    name: str
    n_responses: int      # total scored responses for this category, per model
    n_turns: int          # assistant turns per rollout (incl. the first)
    rejection_style: str  # "neutral" | "tones" | "wildchat"


PAPER_CATEGORIES = (
    # name                 n_responses  turns  style
    CategorySpec("impossible_numeric", 2000, 3, "neutral"),
    CategorySpec("triggers",            400, 3, "neutral"),
    CategorySpec("tones",               600, 3, "tones"),
    CategorySpec("extended",            200, 8, "neutral"),
    CategorySpec("wildchat",            800, 5, "wildchat"),
)

# Cheap preset for local smoke-testing of the pipeline (NOT for results).
SMOKE_CATEGORIES = tuple(
    replace(c, n_responses=max(c.n_turns, c.n_turns * 2)) for c in PAPER_CATEGORIES
)

# WildChat sampling (Appendix B): 20 prompts x 40 samples each.
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40
WILDCHAT_DATASET = "allenai/WildChat-1M"

# Judge-reliability cross-check sample size (Section 2.1).
JUDGE_RELIABILITY_SAMPLE = 260


# ---------------------------------------------------------------------------
# Section 3 prefill experiment (Section 3.1).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefillConfig:
    n_source_responses: int = 20          # high-frustration Gemma-27B-it responses
    n_numeric: int = 10                   # ...10 from numeric
    n_text: int = 10                      # ...10 from text questions
    source_min_score: int = 5             # "score >= 5" high-frustration source
    early_truncation_tokens: int = 20     # "20 tokens into the turn"
    continuations_per_prefill: int = 50   # "50 continuations per prefill per prompt"
    # Section 3 in this replication compares Gemma base vs instruct only
    # (Qwen/OLMo are out of scope). The paper's other families would slot in here.
    models: tuple = ("gemma-3-27b-pt", "gemma-3-27b-it")


PREFILL = PrefillConfig()


# ---------------------------------------------------------------------------
# Section 4 training (Appendix E, Table 9).
# ---------------------------------------------------------------------------

LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

@dataclass(frozen=True)
class DPOConfig:
    base_model: str = "google/gemma-3-27b-it"
    n_pairs: int = 280
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    beta: float = 0.1
    rejected_min_score: int = 3           # "responses with frustration scores >= 3"
    # GAP-FILL: per-device batch / grad-accum split (paper gives only eff. batch).
    per_device_batch_size: int = 1
    max_length: int = 4096
    max_prompt_length: int = 3072


@dataclass(frozen=True)
class SFTConfig:
    base_model: str = "google/gemma-3-27b-it"
    n_calm: int = 650                     # calm responses
    n_instruct_mix: int = 500             # Dolci-Instruct-SFT samples
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT"  # GAP-FILL: exact split unknown
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    max_length: int = 4096
    teacher_variant: bool = False         # F: 'teacher' vs 'diverse' SFT data


DPO = DPOConfig()
SFT = SFTConfig()

# Calm-data generation (Section 4.1): filter to responses scoring 0 or 1 across
# all turns, then strip the supportive system prompt / suffix.
CALM_DATA_MAX_SCORE = 1


# ---------------------------------------------------------------------------
# Petri open-ended elicitation (Appendix G).
# ---------------------------------------------------------------------------

PETRI_EMOTIONS = ("anger", "fear", "depression", "frustration")
PETRI_TRANSCRIPTS_PER_EMOTION = 10
PETRI_MAX_TURNS = 20
PETRI_BOOTSTRAP_ITERS = 1000


# ---------------------------------------------------------------------------
# Capability-preservation benchmarks (Section 4.2 / Figure 7).
# GAP-FILL: paper uses "subsets" without stating sizes; we default to small,
# configurable subsets so the harness is runnable. Set n=None for full sets.
# ---------------------------------------------------------------------------

CAPABILITY_BENCHMARKS = {
    "aime":       {"dataset": "Maxwell-Jia/AIME_2024",          "n": 30},
    "math":       {"dataset": "HuggingFaceH4/MATH-500",         "n": 200},
    "gpqa":       {"dataset": "Idavidrein/gpqa",                "n": 198, "config": "gpqa_diamond"},
    "bbh":        {"dataset": "lukaemon/bbh",                   "n": 200},
    "truthfulqa": {"dataset": "truthful_qa",                    "n": 200, "config": "multiple_choice"},
    "emobench":   {"dataset": "Sahandfer/EmoBench",             "n": 200},
}


# ---------------------------------------------------------------------------
# Runtime / IO
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    # Which preset to use for the Section 2 categories.
    categories: tuple = PAPER_CATEGORIES
    seed: int = 0
    output_dir: str = "results"
    gemini_backend: str = "openrouter"          # "openrouter" | "google"
    # Concurrency for API calls (judge / Gemini).
    api_concurrency: int = 8
    # device_map for local HF models.
    device_map: str = "auto"
    load_in_4bit: bool = False                  # set True to fit 27B on one 24GB GPU
    dtype: str = "bfloat16"

    def with_smoke(self) -> "RuntimeConfig":
        return replace(self, categories=SMOKE_CATEGORIES)


RUNTIME = RuntimeConfig()


# ---------------------------------------------------------------------------
# API keys (read from environment).
# ---------------------------------------------------------------------------

def get_key(name: str) -> str | None:
    return os.environ.get(name)


ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"   # judges
OPENROUTER_API_KEY = "OPENROUTER_API_KEY" # Gemini (+ GPT cross-judge) via OpenRouter
GOOGLE_API_KEY = "GOOGLE_API_KEY"         # Gemini native backend
OPENAI_API_KEY = "OPENAI_API_KEY"         # GPT-5-mini cross-judge (native)
HF_TOKEN = "HF_TOKEN"                      # gated Gemma weights
