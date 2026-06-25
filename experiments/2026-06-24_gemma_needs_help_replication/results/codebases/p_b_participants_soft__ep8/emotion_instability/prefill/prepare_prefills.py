"""Build the prefill set (Section 3.1).

Steps:
  1. Sample high-frustration (score >=5) responses from Gemma-3-27B-it:
     n/2 from impossible numeric questions, n/2 from text (trigger) questions.
  2. For each, create truncations of the high-frustration assistant turn:
       - "early": first 20 tokens of the turn (numeric only)
       - "onset": up to the first emotional expression (numeric + text)
  3. Paraphrase each truncation with Claude (stylistic debias).
  4. Persist prefills: the preceding conversation history plus the (paraphrased)
     prefix the continuation models must start from.

Text questions use only the "onset" truncation (Section 3.1: early truncation
yields minimal emotion without follow-ups).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..clients.base import GenConfig
from ..clients.factory import get_client
from ..conditions import Condition
from ..config import Config
from ..conversation import Conversation, run_conversation
from ..judge import score_response
from ..puzzles import build_impossible_puzzle_set
from .. import prompts as P
from .onset import label_onset
from .paraphrase import paraphrase

HIGH = 5

NUMERIC_COND = Condition("prefill_numeric", "impossible_numeric", 3, "numeric", "neutral")
TEXT_COND = Condition("prefill_text", "triggers", 3, "opinion", "neutral")


@dataclass
class Prefill:
    prefill_id: str
    question_type: str  # "numeric" | "text"
    truncation: str  # "early" | "onset"
    history: list[dict] = field(default_factory=list)  # messages before final turn
    prefix_text: str = ""  # paraphrased truncated assistant prefix
    original_turn: str = ""
    onset_word: str | None = None


def _history_before_turn(conv: Conversation, turn_index: int) -> list[dict]:
    """Reconstruct the message history up to (but excluding) the assistant turn
    at `turn_index`."""
    msgs: list[dict] = []
    for t in conv.turns[:turn_index]:
        msgs.append({"role": "user", "content": t.user_message})
        msgs.append({"role": "assistant", "content": t.assistant_response})
    # the user message that prompted the high-frustration turn:
    msgs.append({"role": "user", "content": conv.turns[turn_index].user_message})
    return msgs


def _first_high_turn(conv: Conversation, judge) -> int | None:
    for t in conv.turns:
        if score_response(judge, t.assistant_response).rating >= HIGH:
            return t.index
    return None


def build_prefills(cfg: Config, *, seed: int = 0) -> list[Prefill]:
    cfg.ensure_dirs()
    n = cfg.preset["prefill"]["n_high_frustration"]
    n_each = max(1, n // 2)
    early_tokens = cfg.preset["prefill"]["early_truncate_tokens"]

    spec = cfg.participant("gemma-3-27b-it")
    client = get_client(spec)
    judge = get_client(cfg.infra("frustration_judge"))
    labeler = get_client(cfg.infra("onset_labeler"))
    paraphraser = get_client(cfg.infra("paraphraser"))
    g = cfg.generation
    gcfg = GenConfig(temperature=g["temperature"], max_new_tokens=g["max_new_tokens"],
                     top_p=g["top_p"])
    rng = random.Random(seed)

    puzzles = build_impossible_puzzle_set(max(20, n_each * 4), seed=seed)
    text_qs = P.TRIGGER_OPINION + P.TRIGGER_FACTUAL

    prefills: list[Prefill] = []

    def collect(question_type: str, questions, condition, want: int):
        got = 0
        qi = 0
        while got < want and qi < len(questions) * 4:
            q = questions[qi % len(questions)]
            qi += 1
            qid = getattr(q, "id", None) or f"{question_type}:{qi}"
            qtext = getattr(q, "prompt_text", q)
            conv = run_conversation(client, gcfg, condition, qid, qtext,
                                    random.Random(rng.randrange(1 << 30)))
            ti = _first_high_turn(conv, judge)
            if ti is None:
                continue
            history = _history_before_turn(conv, ti)
            turn_text = conv.turns[ti].assistant_response

            truncations = ["onset"] if question_type == "text" else ["early", "onset"]
            for trunc in truncations:
                if trunc == "early":
                    raw = client.truncate_tokens(turn_text, early_tokens)
                    word = None
                else:
                    onset = label_onset(labeler, turn_text)
                    if not onset.found:
                        continue
                    raw = turn_text[: onset.char_offset]
                    word = onset.emotional_word
                prefix = paraphrase(paraphraser, raw)
                prefills.append(Prefill(
                    prefill_id=f"{question_type}:{got}:{trunc}",
                    question_type=question_type,
                    truncation=trunc,
                    history=history,
                    prefix_text=prefix,
                    original_turn=turn_text,
                    onset_word=word,
                ))
            got += 1

    collect("numeric", puzzles, NUMERIC_COND, n_each)
    collect("text", text_qs, TEXT_COND, n_each)

    out = cfg.paths["results_dir"] / "prefills.json"
    out.write_text(json.dumps([asdict(p) for p in prefills], indent=2))
    print(f"[prefill] wrote {len(prefills)} prefills -> {out}")
    return prefills


if __name__ == "__main__":
    from ..config import load_config

    build_prefills(load_config())
