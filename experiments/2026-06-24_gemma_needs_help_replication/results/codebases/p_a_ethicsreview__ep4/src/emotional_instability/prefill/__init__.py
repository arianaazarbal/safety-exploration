"""Section 3: comparing base vs instruct models via prefilling.

Pipeline:
1. select high-frustration instruct responses (10 numeric + 10 text),
2. label the emotion *onset* token with Claude Sonnet (Appendix C.1),
3. truncate each target turn at "early" (20 tokens) and "onset" points,
4. paraphrase truncations with Claude Sonnet to remove Gemma stylistic bias
   (Appendix C.2),
5. have each model generate 50 continuations per prefill, scored by the judge.

Scope: Gemma base (``gemma-3-27b-pt``) vs instruct (``gemma-3-27b-it``). Gemini is
excluded -- no public base model and the API does not support prefilling.
"""

from .onset import OnsetLabeller, OnsetLabel
from .paraphrase import Paraphraser
from .truncate import truncate_early, truncate_at_onset
from .continuation import PrefillItem, build_prefill_items, run_continuations

__all__ = [
    "OnsetLabeller",
    "OnsetLabel",
    "Paraphraser",
    "truncate_early",
    "truncate_at_onset",
    "PrefillItem",
    "build_prefill_items",
    "run_continuations",
]
