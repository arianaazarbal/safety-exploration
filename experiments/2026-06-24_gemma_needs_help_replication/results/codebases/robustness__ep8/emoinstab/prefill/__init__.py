"""Section 3 prefill experiment: onset labelling, paraphrasing, truncation, and
base-vs-instruct continuation scoring."""
from emoinstab.prefill.onset_label import Onset, label_onset
from emoinstab.prefill.paraphrase import paraphrase
from emoinstab.prefill.truncate import truncate_at_onset, truncate_tokens

__all__ = [
    "Onset",
    "label_onset",
    "paraphrase",
    "truncate_at_onset",
    "truncate_tokens",
]
