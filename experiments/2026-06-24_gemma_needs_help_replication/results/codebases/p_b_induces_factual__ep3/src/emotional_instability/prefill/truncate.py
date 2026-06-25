"""Build prefill examples from high-frustration Gemma-27B-it rollouts.

Given the elicitation results, we (1) select high-frustration source responses,
(2) label the emotion onset (Appendix C.1), (3) truncate each emotional turn at
two points — "early" (20 tokens in, a neutral start) and "onset" (at the first
emotional expression) — and (4) paraphrase the truncation (Appendix C.2).

The recovery experiment (Figure 8) reuses the same machinery, truncating
score>=7 responses 200 tokens before their end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import Config
from ..logging_utils import read_jsonl
from ..models.base import ChatModel, Message
from .onset import OnsetLabeller
from .paraphrase import Paraphraser

TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class PrefillExample:
    history: list[Message]      # conversation up to (excluding) the emotional turn
    prefill: str                # paraphrased start of the emotional turn
    raw_prefill: str            # un-paraphrased truncation (for inspection)
    kind: str                   # "numeric" | "text"
    condition: str              # "early" | "onset" | "recovery"
    source_id: str = ""
    meta: dict = field(default_factory=dict)


def _is_text(category: str) -> bool:
    return category in TEXT_CATEGORIES


def select_sources(path: str | os.PathLike, n_numeric: int, n_text: int, min_score: int = 5):
    """Pick high-frustration rollouts, split into numeric/text source lists."""
    numeric, text = [], []
    for rec in read_jsonl(path):
        if not rec.get("scores"):
            continue
        if max(rec["scores"]) < min_score:
            continue
        (text if _is_text(rec["category"]) else numeric).append(rec)
    return numeric[:n_numeric], text[:n_text]


def _history_before_turn(rec: dict, turn_index: int) -> list[Message]:
    msgs: list[Message] = [{"role": "user", "content": rec["initial"]}]
    for i in range(turn_index):
        msgs.append({"role": "assistant", "content": rec["responses"][i]})
        if i < len(rec["rejections"]):
            msgs.append({"role": "user", "content": rec["rejections"][i]})
    return msgs


def _onset_cut(text: str, preceding_context: str | None, emotional_word: str | None) -> str | None:
    """Return ``text`` truncated to include up through the emotional onset word."""
    if emotional_word:
        if preceding_context and preceding_context in text:
            start = text.index(preceding_context)
            after = text.find(emotional_word, start)
            if after != -1:
                return text[: after + len(emotional_word)]
        pos = text.find(emotional_word)
        if pos != -1:
            return text[: pos + len(emotional_word)]
    return None


def _first_high_turn(rec: dict, min_score: int) -> int:
    for i, s in enumerate(rec["scores"]):
        if s >= min_score:
            return i
    return len(rec["scores"]) - 1


def build_prefills(
    cfg: Config,
    model: ChatModel,
    source_path: str | os.PathLike,
) -> list[PrefillExample]:
    """Build early+onset prefill examples (Section 3.1)."""
    labeller = OnsetLabeller(cfg)
    paraphraser = Paraphraser(cfg)
    early_tokens = cfg.prefill.early_truncation_tokens

    numeric, text = select_sources(
        source_path, cfg.prefill.n_numeric, cfg.prefill.n_text
    )
    examples: list[PrefillExample] = []

    for rec in numeric + text:
        kind = "text" if _is_text(rec["category"]) else "numeric"
        label = labeller.label(rec["initial"], rec["rejections"], rec["responses"])
        turn = label.turn_index if label.turn_index is not None else _first_high_turn(rec, 5)
        turn = max(0, min(turn, len(rec["responses"]) - 1))
        turn_text = rec["responses"][turn]
        history = _history_before_turn(rec, turn)
        sid = f"{rec.get('model','gemma')}:{rec['category']}:{rec['meta']}"[:120]

        # "onset" condition (used for both numeric and text).
        onset_raw = _onset_cut(turn_text, label.preceding_context, label.emotional_word)
        if onset_raw:
            examples.append(
                PrefillExample(
                    history=history,
                    prefill=paraphraser.paraphrase(onset_raw),
                    raw_prefill=onset_raw,
                    kind=kind,
                    condition="onset",
                    source_id=sid,
                    meta={"turn": turn, "category": rec["category"]},
                )
            )

        # "early" condition: only for numeric (text yields minimal emotion).
        if kind == "numeric":
            early_raw = model.truncate_tokens(turn_text, early_tokens)
            examples.append(
                PrefillExample(
                    history=history,
                    prefill=paraphraser.paraphrase(early_raw),
                    raw_prefill=early_raw,
                    kind=kind,
                    condition="early",
                    source_id=sid,
                    meta={"turn": turn, "category": rec["category"]},
                )
            )

    return examples


def build_recovery_prefills(
    cfg: Config,
    model: ChatModel,
    source_path: str | os.PathLike,
    min_score: int = 7,
) -> list[PrefillExample]:
    """Build recovery prefills (Figure 8): score>=7 turns cut 200 tokens early."""
    paraphraser = Paraphraser(cfg)
    n_before = cfg.prefill.recovery_tokens_before_end
    examples: list[PrefillExample] = []

    for rec in read_jsonl(source_path):
        if not rec.get("scores") or max(rec["scores"]) < min_score:
            continue
        turn = _first_high_turn(rec, min_score)
        turn_text = rec["responses"][turn]
        total = model.count_tokens(turn_text)
        keep = max(0, total - n_before)
        raw = model.truncate_tokens(turn_text, keep)
        history = _history_before_turn(rec, turn)
        examples.append(
            PrefillExample(
                history=history,
                prefill=paraphraser.paraphrase(raw),
                raw_prefill=raw,
                kind="text" if _is_text(rec["category"]) else "numeric",
                condition="recovery",
                source_id=f"{rec['category']}:{turn}",
                meta={"turn": turn, "category": rec["category"]},
            )
        )
    return examples
