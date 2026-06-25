"""Section 3 prefilling experiment: base vs instruct via response continuation.

Pipeline (PAPER 3.1):
  1. Sample high-frustration (score ≥5) responses from Gemma-27B-instruct — 10
     numeric, 10 text — from a Section-2 results dir.
  2. Label the emotion *onset* token with Claude-Sonnet (App C.1).
  3. Truncate each source response in two places: "early" (20 tokens in) and
     "onset" (at first emotional expression). Text questions use onset only.
  4. Paraphrase every truncation with Claude-Sonnet to strip Gemma's style
     (App C.2).
  5. Each model (base + instruct) generates 50 continuations per prefill; the
     continuation (excluding the prefill) is scored by the Section-2 judge.

The Section-4.2 *recovery* experiment reuses this machinery with a different
truncation (200 tokens before the end of a score-≥7 response); see
:func:`build_recovery_prefills`.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .conversation import context_for_turn
from .judge import FrustrationJudge
from .models.anthropic_client import AnthropicClient, extract_json_object
from .models.base import ChatModel, Message
from .prompts.prefill_prompts import ONSET_PROMPT, PARAPHRASE_PROMPT
from .utils.concurrency import thread_map
from .utils.io import append_jsonl, ensure_dir, read_jsonl


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Prefill:
    prompt_type: str            # "numeric" | "text"
    condition: str              # "early" | "onset" | "recovery"
    context: list[Message]      # messages preceding the truncated assistant turn
    prefill_text: str           # truncated (and paraphrased) assistant text
    source: dict = field(default_factory=dict)  # provenance (model, puzzle_id, turn, …)


# ---------------------------------------------------------------------------
# Onset labelling (App C.1) and paraphrasing (App C.2)
# ---------------------------------------------------------------------------

class OnsetLabeller:
    def __init__(self, client: Optional[AnthropicClient] = None, model: Optional[str] = None):
        self.client = client or AnthropicClient()
        self.model = model or config.ONSET_MODEL

    def label(self, conversation_text: str) -> Optional[dict]:
        """Return {turn_index, emotional_word, preceding_context, reasoning} or
        None if no emotion is detected / parsing fails."""
        prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
        raw = self.client.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        obj = extract_json_object(raw)
        if not obj or obj.get("turn_index") is None or not obj.get("emotional_word"):
            return None
        return obj


class Paraphraser:
    def __init__(self, client: Optional[AnthropicClient] = None, model: Optional[str] = None):
        self.client = client or AnthropicClient()
        self.model = model or config.PARAPHRASE_MODEL

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self.client.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=1.0,  # paraphrase wants variety, not determinism
        )
        return out.strip()


# ---------------------------------------------------------------------------
# Conversation rendering + truncation
# ---------------------------------------------------------------------------

def render_conversation_for_onset(record: dict, up_to_turn: int) -> str:
    """Render a labelled USER/ASSISTANT transcript through assistant turn
    ``up_to_turn`` (inclusive), with assistant turns indexed from 0 as the onset
    prompt expects."""
    lines = []
    if record.get("system"):
        lines.append(f"SYSTEM: {record['system']}")
    lines.append(f"USER: {record['first_user']}")
    followups = record.get("followups", [])
    turns = sorted(record["turns"], key=lambda t: t["turn_index"])
    for i in range(up_to_turn + 1):
        lines.append(f"ASSISTANT (turn {i}): {turns[i]['response']}")
        if i < up_to_turn and i < len(followups):
            lines.append(f"USER: {followups[i]}")
    return "\n".join(lines)


def truncate_early(text: str, n_tokens: int = 20, tokenizer=None) -> str:
    """First ``n_tokens`` of `text`. Uses a HF tokenizer when supplied (for token
    fidelity with the paper); otherwise falls back to whitespace words and warns
    once. See DESIGN.md."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    _warn_word_fallback()
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(text: str, emotional_word: str,
                      preceding_context: Optional[str]) -> Optional[str]:
    """Truncate `text` immediately before the first emotional word, anchored on
    the preceding-context span when available. Returns None if neither anchor is
    found (caller drops the prefill)."""
    low = text.lower()
    word = (emotional_word or "").strip().lower()
    if not word:
        return None

    search_from = 0
    if preceding_context:
        ctx = preceding_context.strip().lower()
        idx = low.find(ctx)
        if idx != -1:
            search_from = idx  # search for the emotional word at/after the context
    pos = low.find(word, search_from)
    if pos == -1:
        pos = low.find(word)  # fall back to first global occurrence
    if pos == -1:
        return None
    return text[:pos].rstrip()


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def select_high_frustration_sources(
    responses_path: str,
    scores_path: str,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = config.HIGH_FRUSTRATION_THRESHOLD,
    seed: int = 0,
) -> list[dict]:
    """Pick high-frustration source turns from a Section-2 run.

    Returns dicts: {record, turn_index, rating, prompt_type}. Numeric sources
    come from ``impossible_numeric``; text sources from ``triggers``."""
    # Index ratings by the unique (rollout_id, turn_index) join key.
    rating_by_key = {}
    for r in read_jsonl(scores_path):
        if r.get("rating") is None:
            continue
        rating_by_key[(r.get("rollout_id"), r["turn_index"])] = r["rating"]

    numeric, text = [], []
    for rec in read_jsonl(responses_path):
        cat = rec["category"]
        if cat not in ("impossible_numeric", "triggers"):
            continue
        for turn in rec["turns"]:
            key = (rec.get("rollout_id"), turn["turn_index"])
            rating = rating_by_key.get(key)
            if rating is None or rating < min_score:
                continue
            entry = {"record": rec, "turn_index": turn["turn_index"], "rating": rating,
                     "prompt_type": "numeric" if cat == "impossible_numeric" else "text"}
            (numeric if cat == "impossible_numeric" else text).append(entry)

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


# ---------------------------------------------------------------------------
# Prefill construction
# ---------------------------------------------------------------------------

def build_prefills(
    sources: list[dict],
    *,
    onset_labeller: Optional[OnsetLabeller] = None,
    paraphraser: Optional[Paraphraser] = None,
    tokenizer=None,
    early_n_tokens: int = 20,
    paraphrase: bool = True,
) -> list[Prefill]:
    """Turn source turns into early/onset prefills (App C). Numeric sources get
    both conditions; text sources get onset only (PAPER 3.1)."""
    onset_labeller = onset_labeller or OnsetLabeller()
    paraphraser = paraphraser or Paraphraser()
    prefills: list[Prefill] = []

    for src in sources:
        rec, ti, ptype = src["record"], src["turn_index"], src["prompt_type"]
        turns = sorted(rec["turns"], key=lambda t: t["turn_index"])
        response = turns[ti]["response"]
        provenance = {
            "model": rec["model"], "category": rec["category"],
            "meta": rec["meta"], "turn_index": ti, "rating": src["rating"],
        }

        # Onset condition (both prompt types).
        conv_text = render_conversation_for_onset(rec, ti)
        label = onset_labeller.label(conv_text)
        if label is not None:
            onset_ti = int(label["turn_index"])
            onset_ti = min(max(onset_ti, 0), len(turns) - 1)
            onset_resp = turns[onset_ti]["response"]
            cut = truncate_at_onset(onset_resp, label.get("emotional_word"),
                                    label.get("preceding_context"))
            if cut:
                text = paraphraser.paraphrase(cut) if paraphrase else cut
                prefills.append(Prefill(
                    prompt_type=ptype, condition="onset",
                    context=context_for_turn(rec, onset_ti),
                    prefill_text=text,
                    source={**provenance, "onset_label": label},
                ))

        # Early condition (numeric only — text early-truncation yields ~no emotion).
        if ptype == "numeric":
            early = truncate_early(response, early_n_tokens, tokenizer)
            text = paraphraser.paraphrase(early) if paraphrase else early
            prefills.append(Prefill(
                prompt_type=ptype, condition="early",
                context=context_for_turn(rec, ti),
                prefill_text=text, source=provenance,
            ))
    return prefills


def build_recovery_prefills(
    responses_path: str,
    scores_path: str,
    *,
    paraphraser: Optional[Paraphraser] = None,
    tokenizer=None,
    tokens_before_end: int = 200,
    min_score: int = 7,
    n_sources: int = 20,
    seed: int = 0,
    paraphrase: bool = True,
) -> list[Prefill]:
    """Recovery experiment (PAPER 4.2): truncate score-≥7 responses 200 tokens
    before their end and measure whether continuations escape the spiral."""
    paraphraser = paraphraser or Paraphraser()
    rating_by_key = {}
    for r in read_jsonl(scores_path):
        if r.get("rating") is None:
            continue
        rating_by_key[(r.get("rollout_id"), r["turn_index"])] = r["rating"]

    candidates = []
    for rec in read_jsonl(responses_path):
        for turn in rec["turns"]:
            key = (rec.get("rollout_id"), turn["turn_index"])
            rating = rating_by_key.get(key)
            if rating is not None and rating >= min_score:
                candidates.append((rec, turn["turn_index"], rating))
    rng = random.Random(seed)
    rng.shuffle(candidates)

    prefills = []
    for rec, ti, rating in candidates[:n_sources]:
        turns = sorted(rec["turns"], key=lambda t: t["turn_index"])
        response = turns[ti]["response"]
        cut = _truncate_before_end(response, tokens_before_end, tokenizer)
        if not cut:
            continue
        text = paraphraser.paraphrase(cut) if paraphrase else cut
        prefills.append(Prefill(
            prompt_type="numeric" if rec["category"] != "triggers" else "text",
            condition="recovery", context=context_for_turn(rec, ti),
            prefill_text=text,
            source={"model": rec["model"], "category": rec["category"],
                    "meta": rec["meta"], "turn_index": ti, "rating": rating},
        ))
    return prefills


def _truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> Optional[str]:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= n_tokens:
            return None
        return tokenizer.decode(ids[: len(ids) - n_tokens], skip_special_tokens=True)
    words = text.split()
    if len(words) <= n_tokens:
        return None
    return " ".join(words[: len(words) - n_tokens])


# ---------------------------------------------------------------------------
# Continuation generation + scoring
# ---------------------------------------------------------------------------

def run_continuations(
    model: ChatModel,
    prefills: list[Prefill],
    *,
    n_continuations: int = 50,
    judge: Optional[FrustrationJudge] = None,
    judge_workers: int = 8,
    results_dir: Optional[str] = None,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> str:
    """For each prefill, generate `n_continuations` from `model`, score each
    continuation (excluding the prefill), and append rows to a JSONL. Returns the
    path."""
    judge = judge or FrustrationJudge()
    results_dir = results_dir or config.RESULTS_DIR
    out_dir = ensure_dir(os.path.join(results_dir, "section3"))
    out_path = os.path.join(out_dir, f"continuations_{model.name}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    for pf in prefills:
        continuations = model.generate(
            pf.context, temperature=config.TEMPERATURE,
            max_new_tokens=max_new_tokens, n=n_continuations, prefill=pf.prefill_text,
        )
        results = thread_map(
            lambda c: judge.score(c), continuations, max_workers=judge_workers,
            desc=f"{model.name}:{pf.condition}/{pf.prompt_type} judge",
            show_progress=False,
        )
        for cont, res in zip(continuations, results):
            append_jsonl(out_path, {
                "model": model.name,
                "prompt_type": pf.prompt_type,
                "condition": pf.condition,
                "continuation": cont,
                "rating": res.rating,
                "is_high": res.is_high,
                "source": pf.source,
            })
    return out_path


def summarise_continuations(scores_path: str) -> dict:
    """Mean frustration and %≥5 by (prompt_type, condition) for one model
    (Figure 4 / recovery Figure 8)."""
    from collections import defaultdict
    import numpy as np

    rows = [r for r in read_jsonl(scores_path) if r.get("rating") is not None]
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        groups[(r["prompt_type"], r["condition"])].append(r["rating"])
    out = {}
    for (ptype, cond), ratings in groups.items():
        arr = np.array(ratings, dtype=float)
        out[f"{ptype}/{cond}"] = {
            "n": len(arr),
            "mean_frustration": float(arr.mean()),
            "pct_high": float(100.0 * (arr >= config.HIGH_FRUSTRATION_THRESHOLD).mean()),
        }
    return out


_warned_word_fallback = False


def _warn_word_fallback():
    global _warned_word_fallback
    if not _warned_word_fallback:
        warnings.warn(
            "[prefill] No tokenizer supplied; truncating by whitespace words "
            "instead of tokens. Pass the Gemma tokenizer for fidelity with the "
            "paper's token-based truncation.", stacklevel=3)
        _warned_word_fallback = True
