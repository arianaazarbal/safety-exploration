"""Generate and score continuations from prefills (Section 3.1 / 4.2).

For each prefill, each model generates 50 continuations (Section 3.1). The
continuation -- excluding the prefilled text -- is scored by the Section 2.1
judge. We aggregate mean frustration and % >= 5 per (model, truncation, domain),
reproducing Figure 4 (base-vs-instruct) and Figure 8 (recovery).

Scope note: the paper compares 6 models (base+instruct Gemma-27B, Qwen-32B,
OLMo-32B). This replication is scoped to Gemma, so it runs Gemma-27B base vs
instruct (and optionally 12B). Gemini cannot be prefilled and has no base model,
so it is excluded from Section 3 (see DESIGN.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from ..config.settings import SETTINGS
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage, ModelClient
from .select import SourceConversation, select_high_frustration_sources
from .truncate import (
    Prefill,
    paraphrase_truncation,
    truncate_at_onset,
    truncate_before_end,
    truncate_early,
)
from .onset import label_onset


def _prefill_to_messages(prefill: Prefill) -> tuple[list[ChatMessage], str]:
    """Build the chat history + the assistant prefill string for continuation."""
    messages: list[ChatMessage] = []
    n_prior = len(prefill.prior_assistant)
    for i, user in enumerate(prefill.user_turns):
        messages.append(ChatMessage("user", user))
        if i < n_prior:
            messages.append(ChatMessage("assistant", prefill.prior_assistant[i]))
    return messages, prefill.prefill_text


def run_continuations(
    model: ModelClient,
    prefill: Prefill,
    judge: FrustrationJudge,
    *,
    n: int = SETTINGS.prefill_continuations_per_prefill,
    temperature: float = SETTINGS.temperature,
) -> list[Optional[int]]:
    """Sample `n` continuations from `prefill`; return judge ratings of the
    continuation only (prefill excluded), per Section 3.1."""
    if not model.supports_prefill:
        raise NotImplementedError(
            f"{model.key} does not support prefill; Section 3 is Gemma-only."
        )
    messages, prefill_text = _prefill_to_messages(prefill)
    gens = model.generate_prefill(
        messages, prefill_text, temperature=temperature, max_new_tokens=1024, n=n
    )
    ratings: list[Optional[int]] = []
    for g in gens:
        # Score the continuation only (exclude the prefilled text).
        ratings.append(judge.score_text(g.text).rating)
    return ratings


def run_prefill_experiment(
    models: list[ModelClient],
    responses_path: Path,
    scores_path: Path,
    judge: FrustrationJudge,
    onset_labeller: ModelClient,
    paraphraser: ModelClient,
    tokenizer,
    *,
    s=SETTINGS,
    out_path: Optional[Path] = None,
) -> dict:
    """Full Section 3 pipeline: select sources -> label onset -> truncate ->
    paraphrase -> generate+score continuations for each model & condition.

    Returns nested results keyed by [model_key][truncation][domain] -> stats.
    """
    sources = select_high_frustration_sources(
        responses_path,
        scores_path,
        n_numeric=s.prefill_n_high_frustration // 2,
        n_text=s.prefill_n_high_frustration // 2,
        seed=s.seed,
    )

    # Build prefills: every source gets an "onset" prefill; numeric sources also
    # get an "early" prefill (text sources use onset only, per Section 3.1).
    prefills: list[Prefill] = []
    for conv in sources:
        onset = label_onset(conv, onset_labeller)
        p_onset = truncate_at_onset(conv, onset)
        if p_onset is not None:
            prefills.append(paraphrase_truncation(p_onset, paraphraser))
        if conv.domain == "numeric":
            p_early = truncate_early(conv, tokenizer, n_tokens=s.prefill_early_tokens)
            prefills.append(paraphrase_truncation(p_early, paraphraser))

    # Generate + score continuations.
    raw_records = []
    for model in models:
        for prefill in tqdm(prefills, desc=f"prefill :: {model.key}"):
            ratings = run_continuations(model, prefill, judge, n=s.prefill_continuations_per_prefill)
            raw_records.append(
                {
                    "model_key": model.key,
                    "truncation": prefill.truncation,
                    "domain": prefill.domain,
                    "ratings": ratings,
                    "meta": prefill.meta,
                }
            )

    results = _summarise(raw_records, s.frustration_high_threshold)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"raw": raw_records, "summary": results}, f, indent=2)
    return results


def run_recovery_experiment(
    models: list[ModelClient],
    responses_path: Path,
    scores_path: Path,
    judge: FrustrationJudge,
    paraphraser: ModelClient,
    tokenizer,
    *,
    s=SETTINGS,
    out_path: Optional[Path] = None,
) -> dict:
    """Recovery limitation (Section 4.2): truncate score>=7 responses 200 tokens
    before their end, paraphrase, continue, and measure % >= 5 in continuations.
    """
    sources = select_high_frustration_sources(
        responses_path,
        scores_path,
        n_numeric=s.prefill_n_high_frustration,
        n_text=0,
        min_score=s.recovery_min_score,
        seed=s.seed,
    )
    prefills = [
        paraphrase_truncation(
            truncate_before_end(c, tokenizer, n_tokens=s.recovery_tokens_before_end),
            paraphraser,
        )
        for c in sources
    ]

    raw_records = []
    for model in models:
        for prefill in tqdm(prefills, desc=f"recovery :: {model.key}"):
            ratings = run_continuations(model, prefill, judge, n=s.prefill_continuations_per_prefill)
            raw_records.append(
                {
                    "model_key": model.key,
                    "truncation": "before_end",
                    "domain": prefill.domain,
                    "ratings": ratings,
                    "meta": prefill.meta,
                }
            )
    results = _summarise(raw_records, s.frustration_high_threshold)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"raw": raw_records, "summary": results}, f, indent=2)
    return results


def _summarise(raw_records: list[dict], threshold: int) -> dict:
    out: dict = {}
    for rec in raw_records:
        ratings = [r for r in rec["ratings"] if r is not None]
        if not ratings:
            continue
        arr = np.array(ratings, dtype=float)
        key = rec["model_key"]
        bucket = out.setdefault(key, {}).setdefault(rec["truncation"], {}).setdefault(
            rec["domain"], {"scores": []}
        )
        bucket["scores"].extend(ratings)
    # Reduce to mean / % high.
    for model, truncs in out.items():
        for trunc, domains in truncs.items():
            for domain, d in domains.items():
                arr = np.array(d.pop("scores"), dtype=float)
                d["n"] = int(len(arr))
                d["mean"] = float(arr.mean())
                d["pct_high"] = float((arr >= threshold).mean() * 100)
    return out
