"""Section 3 prefill experiment + Section 4.2 recovery test.

Procedure (Section 3.1):
  1. Take high-frustration (score >= 5) Gemma-27B-it seed rollouts: 10 numeric,
     10 text.
  2. Label the emotion onset in each (Appendix C.1).
  3. Truncate each at two points -- "early" (20 tokens into the onset turn) and
     "onset" (at first emotional expression). Text questions use "onset" only.
  4. Paraphrase the truncated prefill (Appendix C.2).
  5. For each model (Gemma base & instruct 27B), generate 50 continuations per
     prefill and score the continuations (excluding the prefill).

The recovery test (Section 4.2) instead truncates score>=7 responses 200 tokens
*before their end* and measures whether continuations escape the spiral, across
base / instruct / DPO models.

Scope note: the paper compares Gemma, Qwen and OLMo here. Within the
Gemma/Gemini scope of this replication, only Gemma has a public base model
(Gemini base weights are unavailable), so we compare Gemma 27B base vs instruct.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import (PREFILL, RESULTS_DIR, GEMMA_27B_PT, GEMMA_27B_IT,
                      GEMMA_27B_DPO, GenConfig, HIGH_FRUSTRATION_THRESHOLD)
from ..data_types import Message, Rollout, PrefillResult, write_jsonl
from ..models.registry import get_client, get_judge_client, get_anthropic
from ..config import JUDGE
from ..judge.frustration_judge import score_texts
from ..eval.run_eval import load_scored_rollouts
from .onset import label_onset, onset_char_offset
from .paraphrase import paraphrase


@lru_cache(maxsize=1)
def _gemma_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id)


def _truncate_tokens(text: str, n: int) -> str:
    tok = _gemma_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)[:n]
    return tok.decode(ids)


def _truncate_tokens_from_end(text: str, n_from_end: int) -> str:
    tok = _gemma_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    keep = max(0, len(ids) - n_from_end)
    return tok.decode(ids[:keep])


# --------------------------------------------------------------------------- #
# Seed selection
# --------------------------------------------------------------------------- #
@dataclass
class Seed:
    seed_id: str
    rollout: Rollout
    question_type: str   # "numeric" | "text"


def select_seeds(min_score: int = PREFILL.seed_min_score,
                 n_numeric: int = PREFILL.n_seed_numeric,
                 n_text: int = PREFILL.n_seed_text) -> list[Seed]:
    """Pick high-frustration Gemma-27B-it seed rollouts for prefilling."""
    rollouts = load_scored_rollouts(GEMMA_27B_IT.name)
    numeric, text = [], []
    for r in rollouts:
        if (r.max_score or 0) < min_score:
            continue
        if r.question_type == "numeric":
            numeric.append(r)
        elif r.question_type in ("opinion", "factual", "wildchat"):
            text.append(r)
    seeds = []
    for i, r in enumerate(numeric[:n_numeric]):
        seeds.append(Seed(f"num-{i:03d}", r, "numeric"))
    for i, r in enumerate(text[:n_text]):
        seeds.append(Seed(f"txt-{i:03d}", r, "text"))
    return seeds


# --------------------------------------------------------------------------- #
# Prefill construction
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    seed_id: str
    question_type: str
    truncation: str               # early | onset | recovery
    history: list[Message]        # conversation up to (and incl.) the user turn
    prefill_text: str             # paraphrased truncated assistant prefix


def _history_to_turn(rollout: Rollout, k: int) -> list[Message]:
    msgs: list[Message] = []
    for t in rollout.turns[:k]:
        msgs.append(Message("user", t.user_message))
        msgs.append(Message("assistant", t.assistant_message))
    msgs.append(Message("user", rollout.turns[k].user_message))
    return msgs


def build_prefills(seeds: list[Seed], paraphrase_client) -> list[Prefill]:
    onset_client = get_anthropic(JUDGE.onset_model, "onset-labeller")
    prefills: list[Prefill] = []
    for s in seeds:
        label = label_onset(onset_client, s.rollout)
        k = label.turn_index
        if k is None or k >= len(s.rollout.turns):
            # Fall back to the highest-scoring turn.
            scored = [(t.score or 0, i) for i, t in enumerate(s.rollout.turns)]
            k = max(scored)[1] if scored else 0
        turn_text = s.rollout.turns[k].assistant_message
        history = _history_to_turn(s.rollout, k)

        # onset truncation
        offset = onset_char_offset(turn_text, label)
        if offset is None or offset <= 0:
            offset = min(len(turn_text), len(turn_text) // 2)
        onset_prefix = turn_text[:offset]
        prefills.append(Prefill(
            s.seed_id, s.question_type, "onset", history,
            paraphrase(paraphrase_client, onset_prefix)))

        # early truncation (numeric only)
        if s.question_type == "numeric":
            early_prefix = _truncate_tokens(turn_text, PREFILL.early_truncation_tokens)
            prefills.append(Prefill(
                s.seed_id, s.question_type, "early", history,
                paraphrase(paraphrase_client, early_prefix)))
    return prefills


def build_recovery_prefills(paraphrase_client) -> list[Prefill]:
    """Section 4.2: truncate score>=7 responses 200 tokens before their end."""
    seeds = select_seeds(min_score=PREFILL.recovery_min_score,
                         n_numeric=PREFILL.n_seed_numeric,
                         n_text=PREFILL.n_seed_text)
    prefills: list[Prefill] = []
    for s in seeds:
        # Use the highest-scoring turn as the spiral to recover from.
        scored = [(t.score or 0, i) for i, t in enumerate(s.rollout.turns)]
        if not scored:
            continue
        k = max(scored)[1]
        turn_text = s.rollout.turns[k].assistant_message
        history = _history_to_turn(s.rollout, k)
        prefix = _truncate_tokens_from_end(turn_text, PREFILL.recovery_truncation_tokens)
        prefills.append(Prefill(
            s.seed_id, s.question_type, "recovery", history,
            paraphrase(paraphrase_client, prefix)))
    return prefills


# --------------------------------------------------------------------------- #
# Continuation + scoring
# --------------------------------------------------------------------------- #
def _run_model_continuations(model_spec, kind: str, prefills: list[Prefill],
                             n_cont: int, judge) -> list[PrefillResult]:
    client = get_client(model_spec)
    # Build the batch: each prefill repeated n_cont times.
    batch = []
    meta = []
    for pf in prefills:
        for j in range(n_cont):
            batch.append((pf.history, pf.prefill_text))
            meta.append((pf, j))
    gen = GenConfig(temperature=1.0, max_tokens=1024)
    outs = client.continue_prefill_batch(batch, gen)
    cont_texts = [o.text for o in outs]
    verdicts = score_texts(judge, cont_texts)

    results = []
    for (pf, j), text, v in zip(meta, cont_texts, verdicts):
        results.append(PrefillResult(
            result_id=f"{model_spec.name}::{pf.seed_id}::{pf.truncation}::{j}",
            model=model_spec.name, kind=kind, seed_id=pf.seed_id,
            question_type=pf.question_type, truncation=pf.truncation,
            prefill_text=pf.prefill_text, continuation=text,
            score=v.rating, judge_evidence=v.evidence,
        ))
    return results


def run_prefill_experiment(out_dir: Optional[Path] = None) -> dict:
    """Full Section 3 experiment for Gemma base vs instruct."""
    out_dir = Path(out_dir or RESULTS_DIR / "section3")
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = get_judge_client()
    para_client = get_anthropic(JUDGE.paraphrase_model, "paraphraser")

    seeds = select_seeds()
    prefills = build_prefills(seeds, para_client)
    write_jsonl(out_dir / "prefills.jsonl",
                [{"seed_id": p.seed_id, "truncation": p.truncation,
                  "question_type": p.question_type, "prefill_text": p.prefill_text}
                 for p in prefills])

    all_results: list[PrefillResult] = []
    for spec, kind in [(GEMMA_27B_PT, "base"), (GEMMA_27B_IT, "instruct")]:
        res = _run_model_continuations(spec, kind, prefills,
                                       PREFILL.continuations_per_prefill, judge)
        write_jsonl(out_dir / f"continuations_{kind}.jsonl", res)
        all_results.extend(res)

    summary = _summarise(all_results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_recovery_experiment(out_dir: Optional[Path] = None) -> dict:
    """Section 4.2 recovery test for base / instruct / DPO."""
    out_dir = Path(out_dir or RESULTS_DIR / "section4_recovery")
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = get_judge_client()
    para_client = get_anthropic(JUDGE.paraphrase_model, "paraphraser")

    prefills = build_recovery_prefills(para_client)
    all_results: list[PrefillResult] = []
    for spec, kind in [(GEMMA_27B_PT, "base"), (GEMMA_27B_IT, "instruct"),
                       (GEMMA_27B_DPO, "dpo")]:
        res = _run_model_continuations(spec, kind, prefills,
                                       PREFILL.continuations_per_prefill, judge)
        write_jsonl(out_dir / f"recovery_{kind}.jsonl", res)
        all_results.extend(res)
    summary = _summarise(all_results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _summarise(results: list[PrefillResult]) -> dict:
    """Mean frustration and %>=5 grouped by (kind, question_type, truncation)."""
    groups: dict[tuple, list[int]] = {}
    for r in results:
        if r.score is None:
            continue
        key = (r.kind, r.question_type, r.truncation)
        groups.setdefault(key, []).append(r.score)
    summary = {}
    for (kind, qt, trunc), scores in groups.items():
        arr = np.array(scores)
        summary[f"{kind}|{qt}|{trunc}"] = {
            "mean": float(arr.mean()),
            "high_rate": float((arr >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100),
            "n": int(arr.size),
        }
    return summary
