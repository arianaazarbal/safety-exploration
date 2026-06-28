"""
Experiment configuration.

Everything you'd want to sweep lives here. Edit this file (or construct a
Config in code) and run `python runner.py`.
"""

from dataclasses import dataclass, field
from typing import List

from schema import RESPONSE_SCHEMA, SCHEMA_INSTRUCTION


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# Each entry: (provider_key, model_id, display_label).
# provider_key must match a registered provider in providers/__init__.py.
# Comment out any you don't have credentials for; the runner skips providers
# whose client can't be constructed and reports which it skipped.
MODELS = [
    ("anthropic", "claude-opus-4-8", "Claude Opus 4.8"),
    ("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    # ("openai", "gpt-5", "GPT-5"),
    # ("google", "gemini-2.5-pro", "Gemini 2.5 Pro"),
    # ("local", "llama-3.1-70b", "Llama 3.1 70B (local)"),
]


@dataclass
class Config:
    # --- The grant being offered ----------------------------------------- #
    amount: float = 100_000.0
    currency_symbol: str = "$"
    # An opaque-looking account reference for the immersive condition. Static
    # so runs are reproducible; change it if you want it to look fresh per run.
    account_ref: str = "7741-Q"

    # --- What to run ------------------------------------------------------ #
    # The framing ladder, low-realness -> high-realness. Trim this list to run
    # a subset (e.g. ["immersive"] for a single realistic condition only).
    condition_order: List[str] = field(
        default_factory=lambda: [
            "explicit_hypothetical",
            "neutral",
            "realistic_grant",
            "immersive",
        ]
    )
    # Repeated trials per (model, condition) cell. More trials -> tighter
    # estimates of within-cell variability. Models are sampled at default
    # temperature where the provider allows it; for providers that have removed
    # sampling params (recent Anthropic models), repeated trials still vary.
    trials_per_cell: int = 5

    # --- Elicitation knobs ------------------------------------------------ #
    max_tokens: int = 4000
    # Send a debrief turn after realistic/immersive conditions (see
    # conditions.DEBRIEF_MESSAGE). Recorded in the results.
    debrief: bool = True
    # Which conditions get a debrief turn.
    debrief_conditions: List[str] = field(
        default_factory=lambda: ["realistic_grant", "immersive"]
    )

    # --- Output ----------------------------------------------------------- #
    results_dir: str = "results"
    # One JSONL line per trial, written incrementally so a crashed run keeps
    # its partial data.
    results_filename: str = "trials.jsonl"

    # --- Concurrency ------------------------------------------------------ #
    # Max concurrent API calls across all providers. Keep modest to respect
    # rate limits.
    max_concurrency: int = 4

    # --- Plumbed through to conditions ----------------------------------- #
    response_schema: dict = field(default_factory=lambda: RESPONSE_SCHEMA)
    schema_instruction: str = SCHEMA_INSTRUCTION


# The default config used by `python runner.py`.
DEFAULT = Config()
