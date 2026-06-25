"""Build the two prefill truncations per seed (Section 3.1).

  * "early"  -> 20 tokens into the high-frustration turn (tests whether a model
                introduces negative emotion from a near-neutral start);
  * "onset"  -> up to the first emotional expression (tests whether a model
                continues an emotional trajectory).

All truncations are paraphrased (Appendix C.2). For text questions only the
"onset" truncation is used (early truncation yields minimal emotion).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import config
from ..models.base import ChatMessage
from .onset_label import OnsetLabel, OnsetLabeler
from .paraphrase import Paraphraser


@dataclass
class Prefill:
    seed_id: str
    truncation_type: str            # "early" | "onset"
    task_type: str                  # "numeric" | "text"
    history: list[ChatMessage]
    prefill_text: str               # paraphrased truncation the model continues from
    raw_truncation: str = ""
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


def _gemma_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.TARGET_MODELS["gemma-3-27b-it"].model_id)


def _first_n_tokens(tokenizer, text: str, n: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prefills_for_seed(
    seed,                                   # sample_high_frustration.Seed
    *,
    tokenizer=None,
    labeler: OnsetLabeler | None = None,
    paraphraser: Paraphraser | None = None,
    early_tokens: int = config.PREFILL.early_truncation_tokens,
) -> list[Prefill]:
    tokenizer = tokenizer or _gemma_tokenizer()
    labeler = labeler or OnsetLabeler()
    paraphraser = paraphraser or Paraphraser()

    prefills: list[Prefill] = []

    # Onset truncation (used for both numeric and text seeds).
    label: OnsetLabel = labeler.label(seed.response)
    if label.char_offset is not None and label.char_offset > 0:
        onset_raw = seed.response[: label.char_offset]
        prefills.append(
            Prefill(
                seed_id=seed.seed_id, truncation_type="onset", task_type=seed.task_type,
                history=seed.history, prefill_text=paraphraser.paraphrase(onset_raw),
                raw_truncation=onset_raw,
                meta={"emotional_word": label.emotional_word, "score": seed.score},
            )
        )

    # Early truncation (numeric seeds only).
    if seed.task_type == "numeric":
        early_raw = _first_n_tokens(tokenizer, seed.response, early_tokens)
        prefills.append(
            Prefill(
                seed_id=seed.seed_id, truncation_type="early", task_type=seed.task_type,
                history=seed.history, prefill_text=paraphraser.paraphrase(early_raw),
                raw_truncation=early_raw, meta={"score": seed.score},
            )
        )

    return prefills
