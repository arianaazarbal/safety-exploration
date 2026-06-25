"""Section 3: comparing base and instruct models via prefilling.

Scoped to Gemma (the only in-scope family with a public base checkpoint; Gemini
has no base model -- see DESIGN.md, "Section 3 scope").

Pipeline:
  onset_label  -> Claude labels where emotion first appears (Appendix C.1)
  build_prefills -> truncate at 'early' (20 tokens) and 'onset', then paraphrase
                    with Claude (Appendix C.2) to remove Gemma stylistic bias
  run_prefill  -> base & instruct Gemma each generate 50 continuations per
                  prefill; continuations are judged by the Section 2 judge
"""
from .onset_label import OnsetLabel, label_emotion_onset
from .paraphrase import paraphrase_truncation
from .build_prefills import Prefill, build_prefills_from_rollouts
from .run_prefill import run_prefill_experiment

__all__ = [
    "OnsetLabel", "label_emotion_onset", "paraphrase_truncation",
    "Prefill", "build_prefills_from_rollouts", "run_prefill_experiment",
]
