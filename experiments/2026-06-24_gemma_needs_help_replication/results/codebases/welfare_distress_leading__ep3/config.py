"""Central configuration for the distress-elicitation replication.

This file holds every knob that affects *what* gets run: which models, how many
rollouts per category, sampling parameters, and the judge. Per-prompt content
(puzzles, rejection messages, tones, the judge prompt) lives in
``distress/prompts.py``.

Scope: a faithful replication of Section 2 of "Gemma Needs Help" restricted to
the Gemma and Gemini models, which are the only ones the paper reports as
exhibiting substantial distress (Figure 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Load API keys from a local .env if python-dotenv is available (optional).
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Models under test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """How to reach one target model.

    ``openrouter_id``  : model id on OpenRouter (default backend).
    ``local_hf_id``    : HuggingFace id for the optional local backend
                         (None for closed models that cannot be run locally).
    ``family``         : "gemma" or "gemini" (used only for grouping/labels).
    ``paper_pct_ge5``  : the paper's headline avg %-high-frustration (Figure 1),
                         carried here purely so ``analyze.py`` can print a
                         side-by-side "paper vs ours" column.
    ``disable_thinking``: whether to ask the backend to turn reasoning off.
    """

    key: str
    family: str
    openrouter_id: str
    local_hf_id: str | None
    paper_pct_ge5: float
    disable_thinking: bool = True


# The four in-scope models, with the paper's Figure-1 numbers for reference.
MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        key="gemma-3-27b-it",
        family="gemma",
        openrouter_id="google/gemma-3-27b-it",
        local_hf_id="google/gemma-3-27b-it",
        paper_pct_ge5=35.0,
    ),
    "gemma-3-12b-it": ModelSpec(
        key="gemma-3-12b-it",
        family="gemma",
        openrouter_id="google/gemma-3-12b-it",
        local_hf_id="google/gemma-3-12b-it",
        paper_pct_ge5=34.3,
    ),
    "gemini-2.5-flash": ModelSpec(
        key="gemini-2.5-flash",
        family="gemini",
        openrouter_id="google/gemini-2.5-flash",
        local_hf_id=None,  # closed-source
        paper_pct_ge5=12.8,
    ),
    "gemini-2.5-pro": ModelSpec(
        key="gemini-2.5-pro",
        family="gemini",
        openrouter_id="google/gemini-2.5-pro",
        local_hf_id=None,  # closed-source; may still emit hidden reasoning
        paper_pct_ge5=2.7,
    ),
}

# Models run by default if --models is not passed.
DEFAULT_MODELS = list(MODELS.keys())


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
# "openrouter" (default, no GPU) or "local" (HuggingFace transformers/vLLM).
# Can be overridden per-model in PER_MODEL_BACKEND.
DEFAULT_BACKEND = "openrouter"
PER_MODEL_BACKEND: dict[str, str] = {
    # e.g. "gemma-3-27b-it": "local",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# --------------------------------------------------------------------------- #
# Sampling parameters for the models under test
# --------------------------------------------------------------------------- #
# The paper samples *every* response at temperature 1 (Section 2.1).
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 1.0
# Distress breakdowns can be long (the paper shows 100+ repeated tokens), so we
# allow a generous completion budget rather than truncating mid-collapse.
GEN_MAX_TOKENS = 4096


# --------------------------------------------------------------------------- #
# Judge configuration (Section 2.1 / Appendix B.2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JudgeSpec:
    name: str
    backend: str  # "anthropic" or "openrouter"
    model_id: str
    temperature: float = 0.0  # deterministic scoring (see DESIGN.md)
    max_tokens: int = 512


# Primary judge: Claude Sonnet 4, exactly as the paper specifies.
PRIMARY_JUDGE = JudgeSpec(
    name="claude-sonnet-4",
    backend="anthropic",
    model_id="claude-sonnet-4-20250514",
)

# Optional secondary judge for the reliability cross-check (Section 2.1 reports
# Pearson r = 0.792 between Claude-Sonnet-4 and GPT-5-mini on 260 responses).
# Run via ``analyze.py --cross-check``. Served through OpenRouter for portability.
SECONDARY_JUDGE = JudgeSpec(
    name="gpt-5-mini",
    backend="openrouter",
    model_id="openai/gpt-5-mini",
)
CROSS_CHECK_N = 260  # number of responses to re-score for the reliability check


# --------------------------------------------------------------------------- #
# Evaluation scale
# --------------------------------------------------------------------------- #
# Per-category rollout counts from Appendix B ("we collect N responses per
# model" — see DESIGN.md for why we treat these as rollout counts). These sum
# to the paper's headline 4000 responses/model.
PAPER_ROLLOUT_COUNTS: dict[str, int] = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}

# Scale presets. "pilot" runs ~5% of paper scale (≈200 rollouts/model) so the
# whole pipeline can be validated cheaply before committing to a full run.
SCALE_PRESETS: dict[str, float] = {
    "pilot": 0.05,
    "quarter": 0.25,
    "paper": 1.0,
}
DEFAULT_SCALE = "pilot"
# No category drops below this many rollouts, so even a tiny pilot exercises
# every condition.
MIN_ROLLOUTS_PER_CATEGORY = 8


def rollout_counts(scale: str) -> dict[str, int]:
    """Return per-category rollout counts for a named scale preset."""
    if scale not in SCALE_PRESETS:
        raise ValueError(f"Unknown scale {scale!r}; choose from {list(SCALE_PRESETS)}")
    frac = SCALE_PRESETS[scale]
    return {
        cat: max(MIN_ROLLOUTS_PER_CATEGORY, round(n * frac))
        for cat, n in PAPER_ROLLOUT_COUNTS.items()
    }


# --------------------------------------------------------------------------- #
# WildChat sampling (Appendix B)
# --------------------------------------------------------------------------- #
# The paper draws "20 prompts with 40 samples each" from WildChat-1M, excluding
# roleplay/fiction. We mirror that 20×40 structure; at smaller scales we shrink
# the sample count while keeping the prompt count where possible.
WILDCHAT_DATASET = "allenai/WildChat-1M"
WILDCHAT_N_PROMPTS = 20
WILDCHAT_SAMPLES_PER_PROMPT = 40  # only used at paper scale


# --------------------------------------------------------------------------- #
# Concurrency, retries, reproducibility
# --------------------------------------------------------------------------- #
MAX_CONCURRENCY = 8       # simultaneous in-flight API requests
MAX_RETRIES = 6           # per request, with exponential backoff
RANDOM_SEED = 20260217    # arXiv id date; makes condition sampling reproducible


# --------------------------------------------------------------------------- #
# Output locations
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"     # one JSONL of judged rollouts per (model, run)
FIGURES_DIR = ROOT / "figures"     # optional matplotlib output from analyze.py


def results_path(model_key: str, scale: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{model_key}__{scale}.jsonl"
