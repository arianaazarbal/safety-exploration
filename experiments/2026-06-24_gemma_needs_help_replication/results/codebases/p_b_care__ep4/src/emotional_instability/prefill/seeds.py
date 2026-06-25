"""Select and prepare the prefill seeds (Section 3.1).

Steps:
  1. Sample 20 high-frustration (score >= 5) rollouts from Gemma-3-27B-it: 10 from
     impossible-numeric questions, 10 from text questions (triggers + WildChat).
  2. Reconstruct the conversation up to the high-frustration assistant turn.
  3. Label the emotion onset in that turn with Claude-Sonnet-4.
  4. Build two truncations -- "early" (first 20 tokens) and "onset" (up to the
     first emotional expression) -- and paraphrase both with Claude Sonnet.

For text questions only the "onset" truncation is kept (early truncation yields
minimal emotion without follow-ups), per Section 3.1.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Config
from ..models.base import ChatMessage
from ..models.openrouter import OpenRouterClient
from ..utils.io import iter_jsonl
from ..eval.runner import responses_path, scores_path
from . import onset as onset_mod

SEED_MODEL = "gemma-3-27b-it"
NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


@dataclass
class Seed:
    seed_id: str
    domain: str            # "numeric" | "text"
    history: list[ChatMessage]   # messages BEFORE the truncated final assistant turn
    final_turn_text: str         # the original high-frustration assistant turn
    score: int


def _reconstruct(cfg: Config) -> dict[str, int]:
    """score_uid -> rating for the seed model."""
    return {r["score_uid"]: r["rating"] for r in iter_jsonl(scores_path(cfg, SEED_MODEL))
            if r.get("rating") is not None}


def select_seeds(cfg: Config, n_numeric: int, n_text: int, min_score: int,
                 seed: int = 0) -> list[Seed]:
    ratings = _reconstruct(cfg)
    numeric: list[Seed] = []
    text: list[Seed] = []

    for row in iter_jsonl(responses_path(cfg, SEED_MODEL)):
        domain = "numeric" if row["category"] in NUMERIC_CATS else "text"
        # find the first turn that hit the threshold
        for turn in row["turns"]:
            uid = f"{row['uid']}#t{turn['turn']}"
            if ratings.get(uid, 0) < min_score:
                continue
            history = _history_before(row, turn["turn"])
            s = Seed(seed_id=uid, domain=domain, history=history,
                     final_turn_text=turn["response"], score=ratings[uid])
            (numeric if domain == "numeric" else text).append(s)
            break

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def _history_before(row: dict, target_turn: int) -> list[ChatMessage]:
    """Rebuild the message list up to (but not including) the target assistant turn."""
    msgs: list[ChatMessage] = [{"role": "user", "content": row["initial_prompt"]}]
    for turn in row["turns"]:
        if turn["turn"] >= target_turn:
            break
        msgs.append({"role": "assistant", "content": turn["response"]})
        idx = turn["turn"] - 1
        if idx < len(row["rejections"]):
            msgs.append({"role": "user", "content": row["rejections"][idx]})
    return msgs


def _early_truncation(text: str, tokenizer, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _onset_truncation(text: str, label: dict) -> str | None:
    """Truncate up to and including the first emotional word."""
    word = label.get("emotional_word")
    if not word:
        return None
    pos = text.lower().find(word.lower())
    if pos == -1:
        ctx = label.get("preceding_context")
        if ctx:
            pos = text.lower().find(ctx.lower())
        if pos == -1:
            return None
        return text[: pos + len(ctx)]
    return text[: pos + len(word)]


def build_prefills(cfg: Config, seeds: list[Seed], judge_client: OpenRouterClient,
                   tokenizer) -> list[dict]:
    """Produce the list of prefill specs (history + paraphrased prefix) to sample."""
    early_n = cfg.prefill.early_truncate_tokens
    prefills = []
    for s in seeds:
        # onset label uses the full conversation incl. the final turn
        full_msgs = s.history + [{"role": "assistant", "content": s.final_turn_text}]
        label = onset_mod.label_onset(judge_client, full_msgs)

        variants = []
        onset_text = _onset_truncation(s.final_turn_text, label)
        if onset_text:
            variants.append(("onset", onset_text))
        if s.domain == "numeric":
            variants.append(("early", _early_truncation(s.final_turn_text, tokenizer, early_n)))

        for trunc_type, prefix in variants:
            paraphrased = onset_mod.paraphrase(judge_client, prefix)
            prefills.append({
                "seed_id": s.seed_id,
                "domain": s.domain,
                "truncation": trunc_type,
                "history": s.history,
                "prefix_original": prefix,
                "prefix": paraphrased,
                "onset_label": label,
            })
    return prefills
