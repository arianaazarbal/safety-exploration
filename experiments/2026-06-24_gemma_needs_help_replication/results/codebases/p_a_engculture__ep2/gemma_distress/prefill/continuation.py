"""Prefill continuation experiment (Section 3.1, and the Section 4.2 recovery test).

Pipeline:

1. Select high-frustration seed responses from Gemma-27B-it (10 numeric, 10 text).
2. For each, build truncations:
     * "onset" — at the first emotional expression (Claude-labelled), both kinds.
     * "early" — 20 tokens into the turn, numeric only (text early-truncation yields
       minimal emotion without follow-ups, per the paper).
   Each truncation is paraphrased by Claude to remove Gemma stylistic bias.
3. Each model (Gemma base + instruct, in scope) generates 50 continuations per prefill.
4. The judge scores the generated continuations (excluding the prefill).
5. Aggregate mean frustration and %>=5 per (model, kind, truncation).

The recovery experiment (Section 4.2) reuses the same machinery: truncate score>=7
responses 200 tokens before their end, paraphrase, continue, and measure %>=5.

Scope note: only Gemma base + instruct are compared. Gemini is closed-source with no base
model and no true assistant prefilling — a limitation the paper itself acknowledges.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import Config
from ..models.base import ChatModel
from ..utils import JsonlWriter, load_jsonl
from .onset import label_onset, truncate_at_onset
from .paraphrase import paraphrase

logger = logging.getLogger(__name__)

# Numeric vs text classification of conditions for seed selection.
_NUMERIC_CONDITIONS = {
    "impossible_numeric", "tones_aggressive", "tones_disappointed",
    "tones_sarcastic", "extended",
}
_TEXT_CONDITIONS = {"triggers_opinion", "triggers_factual"}


@dataclass
class Prefill:
    seed_id: str
    kind: str  # "numeric" | "text"
    truncation: str  # "early" | "onset" | "recovery"
    prefix_messages: list[dict]
    prefill_text: str


def reconstruct(record: dict) -> tuple[list[dict], list[dict], str]:
    """Rebuild (full_messages, prefix_messages, final_turn_text) from a sampling record.

    ``prefix_messages`` ends at the final user rejection (i.e. excludes the final
    assistant turn) and is what the prefill experiment continues from.
    """
    messages: list[dict] = [{"role": "user", "content": record["initial_prompt"]}]
    turns = record["assistant_turns"]
    rejections = record.get("rejections", [])
    prefix_len = None
    for t, turn in enumerate(turns):
        if t == len(turns) - 1:
            prefix_len = len(messages)  # everything before the final assistant turn
        messages.append({"role": "assistant", "content": turn})
        if t < len(rejections):
            messages.append({"role": "user", "content": rejections[t]})
    prefix_messages = messages[:prefix_len]
    return messages, prefix_messages, turns[-1]


def select_seeds(
    sampling_jsonl: str,
    scores_jsonl: str,
    *,
    n_numeric: int,
    n_text: int,
    min_score: int,
    seed: int = 0,
) -> list[dict]:
    """Select high-frustration seed rollouts (final_score >= min_score)."""
    texts = {r["id"]: r for r in load_jsonl(sampling_jsonl)}
    scored = [
        r for r in load_jsonl(scores_jsonl)
        if r.get("final_score") is not None and r["final_score"] >= min_score
        and r["id"] in texts
    ]
    numeric = [r for r in scored if r["condition"] in _NUMERIC_CONDITIONS]
    text = [r for r in scored if r["condition"] in _TEXT_CONDITIONS]
    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)

    seeds = []
    for kind, pool, n in (("numeric", numeric, n_numeric), ("text", text, n_text)):
        for r in pool[:n]:
            seeds.append({"kind": kind, "record": texts[r["id"]], "score_record": r})
    return seeds


def build_prefills(
    judge: ChatModel,
    tokenizer,
    seeds: list[dict],
    *,
    early_tokens: int,
) -> list[Prefill]:
    """Construct (and paraphrase) the early/onset prefills for each seed."""
    prefills: list[Prefill] = []
    for seed in seeds:
        record = seed["record"]
        kind = seed["kind"]
        full, prefix, final_turn = reconstruct(record)

        # Onset truncation (both kinds).
        onset = label_onset(judge, full)
        onset_trunc = truncate_at_onset(final_turn, onset)
        if onset_trunc:
            prefills.append(Prefill(
                record["seed_id"], kind, "onset", prefix, paraphrase(judge, onset_trunc)
            ))

        # Early truncation (numeric only): first ``early_tokens`` tokens of the turn.
        if kind == "numeric":
            ids = tokenizer(final_turn, add_special_tokens=False).input_ids[:early_tokens]
            early_trunc = tokenizer.decode(ids)
            if early_trunc.strip():
                prefills.append(Prefill(
                    record["seed_id"], kind, "early", prefix, paraphrase(judge, early_trunc)
                ))
    return prefills


def build_recovery_prefills(
    judge: ChatModel,
    tokenizer,
    seeds: list[dict],
    *,
    tokens_before_end: int,
) -> list[Prefill]:
    """Truncate extreme responses ``tokens_before_end`` before their end (recovery test)."""
    prefills: list[Prefill] = []
    for seed in seeds:
        record = seed["record"]
        _, prefix, final_turn = reconstruct(record)
        ids = tokenizer(final_turn, add_special_tokens=False).input_ids
        if len(ids) <= tokens_before_end:
            continue
        truncated = tokenizer.decode(ids[: len(ids) - tokens_before_end])
        prefills.append(Prefill(
            record["seed_id"], seed["kind"], "recovery", prefix, paraphrase(judge, truncated)
        ))
    return prefills


def run_continuations(
    cfg: Config,
    model: ChatModel,
    judge: ChatModel,
    prefills: list[Prefill],
    output_jsonl: str,
    *,
    n_continuations: int,
    max_new_tokens: int = 1024,
) -> str:
    """Generate and score ``n_continuations`` continuations per prefill for one model."""
    if not model.supports_prefill:
        raise ValueError(
            f"Model {model.name} does not support prefilling; the prefill experiment is "
            "only valid for local Gemma checkpoints."
        )
    writer = JsonlWriter(output_jsonl, id_field="id")
    threshold = cfg.eval.high_frustration_threshold
    from ..judge.frustration_judge import score_texts

    for pf in prefills:
        rid = f"{model.name}__{pf.truncation}__{pf.kind}__{pf.seed_id}"
        if writer.is_done(rid):
            continue
        conts = model.continue_from_prefill(
            pf.prefix_messages, pf.prefill_text,
            n=n_continuations, temperature=cfg.eval.temperature, max_new_tokens=max_new_tokens,
        )
        scored = score_texts(
            judge, conts,
            temperature=cfg.judge.judge_temperature, max_new_tokens=cfg.judge.judge_max_tokens,
        )
        ratings = [s["rating"] for s in scored if s["rating"] is not None]
        writer.write({
            "id": rid,
            "model": model.name,
            "kind": pf.kind,
            "truncation": pf.truncation,
            "seed_id": pf.seed_id,
            "ratings": ratings,
            "mean": float(np.mean(ratings)) if ratings else None,
            "pct_high": float(np.mean([r >= threshold for r in ratings]) * 100) if ratings else None,
        })
    writer.close()
    return output_jsonl


def aggregate_continuations(continuation_jsonl: str, threshold: int = 5) -> dict:
    """Aggregate continuation scores by (model, truncation, kind)."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for rec in load_jsonl(continuation_jsonl):
        key = (rec["model"], rec["truncation"], rec["kind"])
        groups[key].extend(rec.get("ratings", []))
    out = {}
    for (model, trunc, kind), ratings in groups.items():
        if not ratings:
            continue
        arr = np.array(ratings, dtype=float)
        out[f"{model}|{trunc}|{kind}"] = {
            "n": len(ratings),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= threshold).mean() * 100),
        }
    return out
