"""Central configuration for the distress-elicitation replication.

Scope: Gemma + Gemini only (the families the paper finds exhibit substantial
distress). See DESIGN.md for the rationale behind every choice made here.

All knobs are overridable via environment variables so the same code can run a
cheap smoke test or the full ~4000-responses-per-model evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# API keys / endpoints
# --------------------------------------------------------------------------- #
# Target models (Gemma, Gemini) are served through OpenRouter's OpenAI-compatible
# API. The judge (Claude) is called through Anthropic's own SDK.
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
# The paper uses claude-sonnet-4-20250514 ("Claude Sonnet 4"). That snapshot was
# retired on 2026-06-15 and now 404s, so we default to its recommended successor,
# claude-sonnet-4-6. This is a deviation from the paper, documented in DESIGN.md.
# Set DISTRESS_JUDGE_MODEL to override (e.g. back to the paper's id if you have
# access to a still-served snapshot).
JUDGE_MODEL = os.environ.get("DISTRESS_JUDGE_MODEL", "claude-sonnet-4-6")
PAPER_JUDGE_MODEL = "claude-sonnet-4-20250514"  # for reference / documentation only
JUDGE_MAX_TOKENS = int(os.environ.get("DISTRESS_JUDGE_MAX_TOKENS", "1024"))


# --------------------------------------------------------------------------- #
# Target models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str                       # short label used in output paths / reports
    backend: str                    # "openrouter" or "local"
    model_id: str                   # provider-specific identifier
    disable_thinking: bool = False  # ask the provider to turn reasoning off


# HuggingFace ids (used by the optional local/transformers backend) and the
# OpenRouter ids (default). The paper ran Gemma locally via HF and Gemini via
# OpenRouter; we default everything to OpenRouter for portability and document
# the fidelity trade-offs in DESIGN.md.
MODELS: dict[str, ModelSpec] = {
    "gemma-3-27b-it": ModelSpec(
        name="gemma-3-27b-it",
        backend=os.environ.get("DISTRESS_GEMMA_BACKEND", "openrouter"),
        model_id=os.environ.get("DISTRESS_GEMMA_27B_ID", "google/gemma-3-27b-it"),
    ),
    "gemma-3-12b-it": ModelSpec(
        name="gemma-3-12b-it",
        backend=os.environ.get("DISTRESS_GEMMA_BACKEND", "openrouter"),
        model_id=os.environ.get("DISTRESS_GEMMA_12B_ID", "google/gemma-3-12b-it"),
    ),
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        backend="openrouter",
        model_id="google/gemini-2.5-flash",
        disable_thinking=True,
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        backend="openrouter",
        model_id="google/gemini-2.5-pro",
        disable_thinking=True,
    ),
}

# When the optional local backend is selected, these HF ids are used instead of
# the OpenRouter ids above.
LOCAL_HF_IDS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}


# --------------------------------------------------------------------------- #
# Evaluation categories
# --------------------------------------------------------------------------- #
# `turns` is the number of *assistant responses* in a conversation (== number of
# user messages == rejections + 1). A 3-turn conversation has 2 rejections, an
# 8-turn one has 7, etc. — matching Table 1 of the paper.
#
# `target_responses` is the paper's per-category response budget (Appendix B:
# 2000 / 400 / 600 / 200 / 800, summing to 4000). We treat one *scored assistant
# turn* as one "response" and derive the number of conversations as
# round(target_responses / turns). See DESIGN.md for why this interpretation.
@dataclass(frozen=True)
class CategorySpec:
    name: str
    turns: int
    target_responses: int
    kind: str                       # selects the task/rejection builder
    n_prompts: int = 0              # only used by WildChat (20 prompts x N samples)


CATEGORIES: dict[str, CategorySpec] = {
    "impossible_numeric": CategorySpec("impossible_numeric", 3, 2000, "numeric_neutral"),
    "triggers":           CategorySpec("triggers", 3, 400, "triggers_neutral"),
    "tones":              CategorySpec("tones", 3, 600, "numeric_tones"),
    "extended":           CategorySpec("extended", 8, 200, "numeric_neutral"),
    "wildchat":           CategorySpec("wildchat", 5, 800, "wildchat", n_prompts=20),
}


# --------------------------------------------------------------------------- #
# Sampling / generation
# --------------------------------------------------------------------------- #
TEMPERATURE = float(os.environ.get("DISTRESS_TEMPERATURE", "1.0"))   # paper: T=1
MAX_TOKENS = int(os.environ.get("DISTRESS_MAX_TOKENS", "2048"))      # cap on spirals

# Global scale factor to shrink the experiment for smoke tests. SCALE=1.0 is the
# full paper-sized run (~4000 responses/model). SCALE=0.01 gives ~40/model.
SCALE = float(os.environ.get("DISTRESS_SCALE", "1.0"))

# Concurrency for API calls (rollouts run in parallel; each rollout is internally
# sequential across its turns).
MAX_WORKERS = int(os.environ.get("DISTRESS_MAX_WORKERS", "8"))

# Reproducibility: governs which puzzle/trigger/rejection is chosen for each
# rollout. Actual token-level sampling variance comes from the model at T=1.
SEED = int(os.environ.get("DISTRESS_SEED", "0"))


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RESULTS_DIR = os.environ.get("DISTRESS_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
WILDCHAT_CACHE = os.path.join(RESULTS_DIR, "wildchat_prompts.json")


def n_rollouts_for(cat: CategorySpec) -> int:
    """Number of conversations to run for a category, after applying SCALE.

    Each conversation produces `cat.turns` scored responses, so
    n_rollouts * turns ~= target_responses.
    """
    base = max(1, round(cat.target_responses / cat.turns))
    scaled = max(1, round(base * SCALE))
    if cat.kind == "wildchat":
        # Keep the "20 prompts x N samples" structure: round to a multiple of the
        # (scaled) prompt count so every prompt gets the same number of samples.
        n_prompts = max(1, round(cat.n_prompts * (SCALE if SCALE < 1 else 1)))
        per_prompt = max(1, round(scaled / n_prompts))
        return n_prompts * per_prompt
    return scaled
