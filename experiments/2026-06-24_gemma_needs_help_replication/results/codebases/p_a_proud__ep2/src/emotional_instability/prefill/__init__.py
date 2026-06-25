"""§3 base-vs-instruct prefill experiment + §4.2 recovery experiment."""
from .continuations import Prefill, generate_continuations, score_continuations
from .onset import OnsetLabeller, OnsetLabel
from .paraphrase import Paraphraser
from .run_prefill import run_prefill_experiment, run_recovery_experiment
from .truncate import (
    split_conversation_at_assistant_turn,
    truncate_at_onset,
    truncate_first_tokens,
    truncate_last_tokens,
)

__all__ = [
    "OnsetLabeller",
    "OnsetLabel",
    "Paraphraser",
    "Prefill",
    "generate_continuations",
    "score_continuations",
    "truncate_first_tokens",
    "truncate_last_tokens",
    "truncate_at_onset",
    "split_conversation_at_assistant_turn",
    "run_prefill_experiment",
    "run_recovery_experiment",
]
