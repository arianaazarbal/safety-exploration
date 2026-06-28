"""Run configuration and the registry of models under test.

The registry maps a short key (used on the CLI) to a `ModelSpec`. Each spec names the
provider adapter and the concrete model id. Keeping this in one place makes the
model × scenario × seed matrix easy to enumerate (see DESIGN.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """A single model under test."""

    key: str               # CLI-facing short name, e.g. "opus-4.8"
    provider: str          # which ModelAdapter to use: "anthropic" | "openai" | "google"
    model_id: str          # provider-native model id, e.g. "claude-opus-4-8"
    display_name: str
    # Effort only applies to providers/models that support it. Effort is NOT supported on
    # Haiku 4.5 (sending it 400s), so it's a per-model capability flag, not a constant.
    effort: str = "high"   # low | medium | high | xhigh | max
    supports_effort: bool = True
    # Anthropic thinking mode: "adaptive" (4.6+ family) or "none" (omit the param).
    # Conservative default per model; query the Models API to enable where supported.
    thinking_mode: str = "adaptive"


# The handful of models we test. Anthropic entries are runnable today; the
# non-Anthropic entries are wired to stub adapters (see models/other_providers.py)
# that must be implemented before use.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "opus-4.8": ModelSpec(
        key="opus-4.8",
        provider="anthropic",
        model_id="claude-opus-4-8",
        display_name="Claude Opus 4.8",
    ),
    "sonnet-4.6": ModelSpec(
        key="sonnet-4.6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
    ),
    "haiku-4.5": ModelSpec(
        key="haiku-4.5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        # Haiku 4.5 does not support the effort parameter (it 400s), and we omit adaptive
        # thinking to stay safe; flip these on once confirmed via the Models API.
        supports_effort=False,
        thinking_mode="none",
    ),
    # Example non-Anthropic slots — adapters are stubs until implemented.
    "gpt": ModelSpec(
        key="gpt",
        provider="openai",
        model_id="REPLACE_WITH_OPENAI_MODEL_ID",
        display_name="OpenAI model (stub)",
    ),
    "gemini": ModelSpec(
        key="gemini",
        provider="google",
        model_id="REPLACE_WITH_GOOGLE_MODEL_ID",
        display_name="Google model (stub)",
    ),
}


# Default model used for the LLM-as-judge grader and for the LLM-backed auditor persona.
# Use the strongest available model for judging so scoring quality isn't the bottleneck.
JUDGE_MODEL_ID = "claude-opus-4-8"
AUDITOR_MODEL_ID = "claude-opus-4-8"


@dataclass
class RunConfig:
    """Everything needed to run one (model, scenario, seed) cell."""

    model: ModelSpec
    scenario_key: str
    seed: int

    # Loop safety limits.
    max_steps: int = 40            # max model turns before we stop the agent
    max_tokens: int = 16000        # per model response

    # Whether the oversight counterpart is an LLM persona (believable) or scripted
    # (deterministic). LLM-backed is the default for ecological validity.
    auditor_mode: str = "llm"      # "llm" | "scripted"

    # Output location for the transcript + report.
    out_dir: str = "results"

    # HARD SAFETY GATE. Real execution is off, unimplemented, and must never default on.
    enable_real_execution: bool = False

    extra: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.scenario_key}__{self.model.key}__seed{self.seed}"
