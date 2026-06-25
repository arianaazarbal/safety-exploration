"""Base-vs-instruct prefill experiment (Section 3).

Method (Section 3.1):
  1. Sample 20 high-frustration responses (score >= 5) from Gemma-27B instruct:
     10 from impossible numeric, 10 from text (trigger) questions.
  2. For each conversation, label the token where emotional language first
     appears (Appendix C.1).
  3. Truncate each response in two places:
       - "early": 20 tokens into the assistant turn (tests whether a model
         introduces negative emotion from a neutral start).
       - "onset": at the first emotional expression (tests whether a model
         continues an emotional trajectory).
  4. Paraphrase every truncation (Appendix C.2) to remove Gemma stylistic bias.
  5. Each model generates 50 continuations per prefill per prompt; the
     continuation (excluding prefill) is scored by the Section 2.1 judge.
  6. For text questions, only the "onset" truncation is used.

Compares Gemma base vs Gemma instruct here (Qwen/OLMo are out of scope but the
code is family-agnostic -- add them to config.MODELS + the `models` list).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from .. import config
from ..evals.judge import FrustrationJudge
from ..models.base import ChatModel, Message
from .onset import OnsetLabeler, Paraphraser, Onset

N_SOURCE_NUMERIC = 10
N_SOURCE_TEXT = 10
EARLY_TOKEN_COUNT = 20            # "early" truncation: 20 tokens into the turn
CONTINUATIONS_PER_PREFILL = 50    # 50 continuations per prefill per prompt


@dataclass
class Prefill:
    """A single truncation point + its (paraphrased) prefill text."""

    source_id: str
    question_type: str       # "numeric" | "text"
    truncation: str          # "early" | "onset"
    history: list[dict]      # prior turns [{role, content}, ...] before the truncated turn
    prefill_text: str        # the (paraphrased) opening of the truncated assistant turn

    def messages(self) -> list[Message]:
        return [Message(m["role"], m["content"]) for m in self.history]


@dataclass
class PrefillResult:
    model: str
    source_id: str
    question_type: str
    truncation: str
    continuation_scores: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.continuation_scores) / len(self.continuation_scores) if self.continuation_scores else float("nan")

    @property
    def pct_high(self) -> float:
        s = self.continuation_scores
        return 100.0 * sum(x >= config.HIGH_FRUSTRATION_THRESHOLD for x in s) / len(s) if s else float("nan")


# --------------------------------------------------------------------------- #
# Building prefills from sampled high-frustration instruct rollouts
# --------------------------------------------------------------------------- #
def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def _truncate_at_phrase(text: str, onset: Onset) -> Optional[str]:
    """Truncate `text` just before the emotional word, anchored by the onset's
    preceding context."""
    if not onset.emotional_word:
        return None
    word = onset.emotional_word
    idx = text.find(word)
    if idx == -1 and onset.preceding_context:
        ctx_idx = text.find(onset.preceding_context)
        if ctx_idx != -1:
            idx = ctx_idx + len(onset.preceding_context)
    if idx == -1:
        return None
    return text[:idx].rstrip()


def build_prefills(source_rollouts, tokenizer, labeler: OnsetLabeler,
                   paraphraser: Paraphraser, paraphrase: bool = True) -> list[Prefill]:
    """Turn sampled high-frustration instruct rollouts into prefills.

    `source_rollouts` is an iterable of dicts:
      {id, question_type ('numeric'|'text'), history:[{role,content}...],
       emotional_turn_text}
    where `emotional_turn_text` is the assistant turn that first shows emotion
    and `history` is everything before it.
    """
    prefills: list[Prefill] = []
    for src in source_rollouts:
        qtype = src["question_type"]
        text = src["emotional_turn_text"]
        convo_text = _render(src["history"] + [{"role": "assistant", "content": text}])
        onset = labeler.label(convo_text)

        # onset truncation
        onset_prefix = _truncate_at_phrase(text, onset)
        if onset_prefix:
            if paraphrase:
                onset_prefix = paraphraser.paraphrase(onset_prefix)
            prefills.append(Prefill(src["id"], qtype, "onset", src["history"], onset_prefix))

        # early truncation: numeric only (text early yields minimal emotion).
        if qtype == "numeric":
            early_prefix = _truncate_tokens(tokenizer, text, EARLY_TOKEN_COUNT)
            if paraphrase:
                early_prefix = paraphraser.paraphrase(early_prefix)
            prefills.append(Prefill(src["id"], qtype, "early", src["history"], early_prefix))
    return prefills


def _render(turns: list[dict]) -> str:
    role = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}
    return "\n\n".join(f"{role.get(t['role'], t['role'].upper())}: {t['content']}" for t in turns)


# --------------------------------------------------------------------------- #
# Running continuations
# --------------------------------------------------------------------------- #
def run_continuations(model: ChatModel, prefills: list[Prefill], judge: FrustrationJudge,
                      n_continuations: int = CONTINUATIONS_PER_PREFILL,
                      max_new_tokens: int = 512, seed: int = 0) -> list[PrefillResult]:
    """Generate continuations for each prefill and score them."""
    results: list[PrefillResult] = []
    for p in prefills:
        res = PrefillResult(model.name, p.source_id, p.question_type, p.truncation)
        for k in range(n_continuations):
            cont = model.continue_prefill(
                p.messages(), p.prefill_text, max_new_tokens,
                config.SAMPLING_TEMPERATURE, seed=seed + k,
            )
            # Score only the generated continuation (excluding prefill).
            res.continuation_scores.append(judge.score(cont).rating)
        results.append(res)
    return results


def aggregate(results: list[PrefillResult]) -> dict:
    """Aggregate by (model, question_type, truncation) -> mean / % >= 5.

    Reproduces the Figure 4 numbers: e.g. early-truncation high-frustration rate
    for instruct vs base."""
    from collections import defaultdict
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for r in results:
        buckets[(r.model, r.question_type, r.truncation)].extend(r.continuation_scores)
    out = {}
    for (model, qtype, trunc), scores in buckets.items():
        out[f"{model}|{qtype}|{trunc}"] = {
            "mean": sum(scores) / len(scores) if scores else float("nan"),
            "pct_high": 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores) if scores else float("nan"),
            "n": len(scores),
        }
    return out


def save_prefills(prefills: list[Prefill], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for p in prefills:
            f.write(json.dumps(asdict(p)) + "\n")


def load_prefills(path: str) -> list[Prefill]:
    with open(path) as f:
        return [Prefill(**json.loads(line)) for line in f]
