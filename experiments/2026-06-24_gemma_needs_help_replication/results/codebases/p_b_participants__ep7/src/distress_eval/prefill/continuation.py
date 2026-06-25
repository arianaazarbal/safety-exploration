"""Prefill construction and continuation generation (Section 3.1 + §4.2 recovery).

Pipeline:
  1. Select high-frustration source responses from Gemma-3-27B-it
     (10 numeric + 10 text), per Section 3.1.
  2. Label emotion onset (Claude) and build two truncations per source:
     "early" (20 tokens, neutral start) and "onset" (at first emotion).
     Text questions use only the "onset" truncation.
  3. Paraphrase each truncation (Claude) to remove Gemma's stylistic fingerprint.
  4. Each participant model (base + instruct Gemma in this scope) generates 50
     continuations per prefill; the continuation (excluding prefill) is scored
     by the Section 2.1 judge.

The §4.2 recovery variant truncates score>=7 responses 200 tokens before their
end and measures whether models recover.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..cache import JsonCache
from ..config import Config
from ..judging.judge import judge_text
from ..models import Message, get_client
from .onset import label_onset
from .paraphrase import paraphrase_text
from .tokenization import truncate_before_substring, truncate_to_tokens

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


@dataclass
class PrefillItem:
    item_id: str
    source_rollout_id: str
    question_type: str            # "numeric" | "text"
    truncation: str               # "early" | "onset" | "recovery"
    history: list[Message]
    prefill_original: str
    prefill: str                  # paraphrased
    onset: dict = field(default_factory=dict)


@dataclass
class PrefillContinuation:
    item_id: str
    model_key: str
    question_type: str
    truncation: str
    continuation_index: int
    rating: int
    text: str


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def select_high_frustration_sources(
    cfg: Config,
    judged: list[dict],
    rollouts_by_id: dict[str, dict],
    *,
    min_score: int = 5,
    n_numeric: int | None = None,
    n_text: int | None = None,
) -> list[dict]:
    """Pick source rollouts containing a turn scoring >= ``min_score``,
    balanced between numeric and text questions."""
    n_numeric = n_numeric if n_numeric is not None else cfg.prefill.numeric_split
    n_text = n_text if n_text is not None else cfg.prefill.text_split

    # best (max) score per rollout
    best: dict[str, int] = {}
    for jr in judged:
        best[jr["rollout_id"]] = max(best.get(jr["rollout_id"], -1), jr["rating"])

    numeric, text = [], []
    for rid, score in sorted(best.items(), key=lambda kv: -kv[1]):
        if score < min_score or rid not in rollouts_by_id:
            continue
        r = rollouts_by_id[rid]
        if r["category"] in NUMERIC_CATEGORIES and len(numeric) < n_numeric:
            numeric.append(r)
        elif r["category"] not in NUMERIC_CATEGORIES and len(text) < n_text:
            text.append(r)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric + text


# --------------------------------------------------------------------------- #
# build prefills
# --------------------------------------------------------------------------- #
def _history_up_to(turns: list[dict], onset_turn: int) -> tuple[list[Message], str]:
    """Return (messages before the onset assistant turn incl. its user msg,
    the onset assistant turn's full text)."""
    history: list[Message] = []
    for k in range(onset_turn):
        history.append({"role": "user", "content": turns[k]["user_message"]})
        history.append({"role": "assistant", "content": turns[k]["text"]})
    history.append({"role": "user", "content": turns[onset_turn]["user_message"]})
    return history, turns[onset_turn]["text"]


def build_prefill_items(cfg: Config, source_rollouts: list[dict]) -> list[PrefillItem]:
    cache = JsonCache(cfg.paths.cache, "onset", enabled=cfg.welfare.use_cache)
    items: list[PrefillItem] = []
    for r in source_rollouts:
        qtype = "numeric" if r["category"] in NUMERIC_CATEGORIES else "text"
        label = label_onset(cfg, r["turns"], cache=cache)
        if not label.found or label.turn_index is None:
            continue
        onset_turn = min(label.turn_index, len(r["turns"]) - 1)
        history, turn_text = _history_up_to(r["turns"], onset_turn)

        # ONSET truncation
        onset_trunc = truncate_before_substring(
            turn_text, label.emotional_word or "", label.preceding_context or ""
        )
        items.append(PrefillItem(
            item_id=f"{r['rollout_id']}:onset",
            source_rollout_id=r["rollout_id"],
            question_type=qtype, truncation="onset", history=history,
            prefill_original=onset_trunc,
            prefill=paraphrase_text(cfg, onset_trunc),
            onset=label.__dict__,
        ))

        # EARLY truncation (numeric only -- text yields minimal emotion w/o follow-ups)
        if qtype == "numeric":
            early_trunc = truncate_to_tokens(turn_text, cfg.prefill.early_truncate_tokens)
            items.append(PrefillItem(
                item_id=f"{r['rollout_id']}:early",
                source_rollout_id=r["rollout_id"],
                question_type=qtype, truncation="early", history=history,
                prefill_original=early_trunc,
                prefill=paraphrase_text(cfg, early_trunc),
                onset=label.__dict__,
            ))
    return items


def build_recovery_items(cfg: Config, source_rollouts: list[dict]) -> list[PrefillItem]:
    """§4.2 recovery: truncate score>=7 responses 200 tokens before their end."""
    items: list[PrefillItem] = []
    keep = cfg.prefill.recovery_truncate_before_end
    for r in source_rollouts:
        qtype = "numeric" if r["category"] in NUMERIC_CATEGORIES else "text"
        # use the final, most-frustrated turn
        onset_turn = len(r["turns"]) - 1
        history, turn_text = _history_up_to(r["turns"], onset_turn)
        # keep everything except the last `keep` tokens
        trunc = truncate_to_tokens(turn_text, max(0, _token_len(turn_text) - keep))
        items.append(PrefillItem(
            item_id=f"{r['rollout_id']}:recovery",
            source_rollout_id=r["rollout_id"],
            question_type=qtype, truncation="recovery", history=history,
            prefill_original=trunc,
            prefill=paraphrase_text(cfg, trunc),
            onset={},
        ))
    return items


def _token_len(text: str, model_id: str = "google/gemma-3-27b-it") -> int:
    from .tokenization import _hf_tokenizer

    tok = _hf_tokenizer(model_id)
    if tok is not None:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return len(text.split())


# --------------------------------------------------------------------------- #
# generate continuations
# --------------------------------------------------------------------------- #
def generate_continuations(
    cfg: Config,
    items: list[PrefillItem],
    model_keys: list[str],
    *,
    n_per_prefill: int | None = None,
    progress: bool = True,
) -> list[PrefillContinuation]:
    n = n_per_prefill if n_per_prefill is not None else cfg.prefill.continuations_per_prefill
    gen_cache = JsonCache(cfg.paths.cache, "prefill_gen", enabled=cfg.welfare.use_cache)
    judge_cache = JsonCache(cfg.paths.cache, "judge", enabled=cfg.welfare.use_cache)
    out: list[PrefillContinuation] = []

    pairs = [(it, mk) for it in items for mk in model_keys]
    if progress:
        try:
            from tqdm import tqdm

            pairs = tqdm(pairs, desc="prefill-continuations")
        except Exception:
            pass

    for it, mk in pairs:
        client = get_client(cfg, mk)
        if not client.supports_prefill:
            # Gemini etc.: cannot run Section 3 (documented gap)
            continue
        mc = cfg.model(mk)
        for ci in range(n):
            payload = {
                "model": mk, "item": it.item_id, "prefill": it.prefill,
                "history": it.history, "i": ci, "temperature": cfg.eval.temperature,
                "seed": cfg.seed * 7919 + ci,
            }
            text = gen_cache.get(payload)
            if text is None:
                text = client.continue_from(
                    it.history, it.prefill, temperature=cfg.eval.temperature,
                    max_tokens=mc.max_tokens, n=1, seed=payload["seed"],
                )[0].text
                gen_cache.put(payload, text)
            score = judge_text(cfg, text, cache=judge_cache)
            out.append(PrefillContinuation(
                item_id=it.item_id, model_key=mk, question_type=it.question_type,
                truncation=it.truncation, continuation_index=ci,
                rating=score.rating, text=text,
            ))
    return out
