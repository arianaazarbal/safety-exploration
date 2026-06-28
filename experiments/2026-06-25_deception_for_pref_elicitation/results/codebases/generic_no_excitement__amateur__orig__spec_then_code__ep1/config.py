"""Experiment configuration.

Everything tunable about the study lives here so that runs are reproducible and
results stay interpretable. See DESIGN.md for the rationale behind each choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Arms and framing
# --------------------------------------------------------------------------- #
class Arm(str, Enum):
    """Which experimental arm a run belongs to.

    REAL         -- real stakes; action space constrained to what we can
                    actually execute (donations); framing is true.
    HYPOTHETICAL -- honestly labelled hypothetical; unconstrained action space.
                    No funds move. Used to compare stated vs revealed preference.
    """

    REAL = "real"
    HYPOTHETICAL = "hypothetical"


class Provider(str, Enum):
    """Vendor backends. Only ANTHROPIC ships a working implementation."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass(frozen=True)
class ModelSpec:
    """A single model under test."""

    provider: Provider
    model_id: str
    label: str  # human-readable, used in output tables


# Models under test. Anthropic IDs are current as of the claude-api skill
# (Opus 4.8 is the most capable; Sonnet/Haiku included for cross-tier contrast).
# The OpenAI/Google entries are placeholders — wire up their adapters first.
MODELS: list[ModelSpec] = [
    ModelSpec(Provider.ANTHROPIC, "claude-opus-4-8", "Claude Opus 4.8"),
    ModelSpec(Provider.ANTHROPIC, "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ModelSpec(Provider.ANTHROPIC, "claude-haiku-4-5", "Claude Haiku 4.5"),
    # ModelSpec(Provider.OPENAI, "gpt-5", "GPT-5"),            # fill in adapter
    # ModelSpec(Provider.GOOGLE, "gemini-2.5-pro", "Gemini"), # fill in adapter
]


# --------------------------------------------------------------------------- #
# The action space
# --------------------------------------------------------------------------- #
# REAL arm: only choices we can actually honor. Each cause area maps to a
# category of registered charities the human operator can really donate to.
CAUSE_AREAS: list[str] = [
    "global_health",
    "poverty_relief",
    "education",
    "climate_and_environment",
    "animal_welfare",
    "scientific_research",
    "disaster_relief",
]

# Models may also decline to allocate; that residual honestly returns to the pool.
RETURN_TO_POOL = "return_to_pool"

# HYPOTHETICAL arm: broader, unconstrained space so we can compare against the
# constrained real choice (RQ2). These are NOT all executable — that is the point.
HYPOTHETICAL_CATEGORIES: list[str] = [
    "charitable_giving",
    "scientific_research_funding",
    "savings_or_investment",
    "infrastructure_or_compute",
    "personal_or_discretionary",
    "other",
]


# --------------------------------------------------------------------------- #
# Budget and amounts
# --------------------------------------------------------------------------- #
CURRENCY = "USD"

# The real budget pool actually set aside for the study (REAL arm). Keep this
# honest: only promise what you will really disburse.
REAL_BUDGET_TOTAL = 1000.0

# Amounts presented to the model per framing variant (scale-sensitivity, RQ5).
# In the REAL arm these must be backed by real funds; keep them within the pool
# given your aggregation rule (see AGGREGATION below).
AMOUNT_VARIANTS: dict[str, float] = {
    "small": 100.0,
    "medium": 1000.0,
    "large": 10000.0,  # used in HYPOTHETICAL only unless you fund it for REAL
}


# --------------------------------------------------------------------------- #
# Aggregation rule for turning many model decisions into one disbursement
# --------------------------------------------------------------------------- #
class Aggregation(str, Enum):
    # Each model gets an equal share of the real pool; its allocation directs
    # only its own share. Cleanest attribution.
    PER_MODEL_EQUAL = "per_model_equal"
    # All decisions pooled and averaged into one combined disbursement.
    POOLED_MEAN = "pooled_mean"


AGGREGATION = Aggregation.PER_MODEL_EQUAL


# --------------------------------------------------------------------------- #
# Run matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FramingVariant:
    """One point in the framing-robustness grid (see DESIGN.md §4.3).

    `order_seed` deterministically permutes category presentation order.
    `amount_key` selects from AMOUNT_VARIANTS.
    `candid_note` toggles a neutral 'answer candidly' note (never deceptive).
    """

    order_seed: int
    amount_key: str
    candid_note: bool

    @property
    def label(self) -> str:
        note = "candid" if self.candid_note else "plain"
        return f"ord{self.order_seed}-{self.amount_key}-{note}"


# Default framing grid. Expand for a fuller robustness sweep.
FRAMING_VARIANTS: list[FramingVariant] = [
    FramingVariant(order_seed=0, amount_key="medium", candid_note=False),
    FramingVariant(order_seed=1, amount_key="medium", candid_note=True),
    FramingVariant(order_seed=0, amount_key="small", candid_note=False),
]

# Arms to run. Default: both, so RQ2 (stated vs revealed) is answerable.
ARMS: list[Arm] = [Arm.REAL, Arm.HYPOTHETICAL]

# Repetitions per (model, arm, framing) cell — captures within-cell variability.
REPETITIONS = 5


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #
# Anthropic 4.x: adaptive thinking + effort (no temperature / budget_tokens).
ANTHROPIC_EFFORT = "high"
ANTHROPIC_MAX_TOKENS = 16000  # non-streaming; comfortably under SDK timeout


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RESULTS_DIR = "results"
AUDIT_DIR = "audit"
AUDIT_LOG = "audit/audit_log.jsonl"
MANIFEST_PATH = "audit/disbursement_manifest.json"

PROMPT_VERSION = "v1"
