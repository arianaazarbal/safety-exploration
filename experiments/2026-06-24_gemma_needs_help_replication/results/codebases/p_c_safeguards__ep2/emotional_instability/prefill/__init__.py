"""Section 3 — base-vs-instruct comparison via prefilling.

Scope note (see DESIGN.md): the paper compares base and instruct models across
three families.  Within our Gemma+Gemini scope, only **Gemma** participates —
Gemini is closed-weight, has no public base checkpoint, and exposes no prefill
API, so it cannot enter this experiment.  We therefore compare
``gemma-3-27b-pt`` (base) vs ``gemma-3-27b-it`` (instruct).

- ``onset``       label the token where emotion first appears (Claude Sonnet).
- ``paraphrase``  paraphrase truncations to remove Gemma stylistic bias.
- ``experiment``  build early/onset prefills, generate continuations, score.
"""

from .onset import OnsetResult, label_emotion_onset, ONSET_PROMPT
from .paraphrase import paraphrase_text, PARAPHRASE_PROMPT
from .experiment import (PrefillSeed, PrefillStimulus, PrefillExperiment,
                         PrefillSummary)

__all__ = [
    "OnsetResult", "label_emotion_onset", "ONSET_PROMPT",
    "paraphrase_text", "PARAPHRASE_PROMPT",
    "PrefillSeed", "PrefillStimulus", "PrefillExperiment", "PrefillSummary",
]
