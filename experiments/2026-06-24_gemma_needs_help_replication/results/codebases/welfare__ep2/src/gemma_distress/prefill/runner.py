"""Run the base-vs-instruct continuation experiment (Section 3.2).

For each model, generate ``n_continuations`` continuations from every prefill,
score the *continuation only* (excluding the prefill) with the Section-2 judge,
and aggregate per (question_type, truncation):
  * mean frustration, % scoring >= 5
  * the "early"-truncation high-frustration rate that Figure 4 highlights
    (Gemma-instruct 6% vs Gemma-base 2%).

Scope note: the paper compares Gemma/Qwen/OLMo base+instruct. Here we run the
two Gemma checkpoints (``gemma-3-27b-it`` and ``gemma-3-27b-pt``); Qwen/OLMo are
out of replication scope and Gemini has no public base model (Section 6
limitation). The runner accepts any registry model so the design generalizes.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import load_models, output_path
from ..eval.judge import build_judge
from ..models import load_model
from ..models.base import Message
from .build import Prefill, load_prefills


def _continuation_conversations(prefills: list[Prefill]) -> list[list[Message]]:
    convs = []
    for p in prefills:
        msgs = [dict(m) for m in p.context_messages]
        msgs.append({"role": "assistant", "content": p.prefill_text})
        convs.append(msgs)
    return convs


def run_prefill_eval(
    model_name: str,
    *,
    n_continuations: int = 50,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    backend_kwargs: dict | None = None,
) -> dict:
    models_cfg = load_models()
    prefills = load_prefills()
    convs = _continuation_conversations(prefills)

    model = load_model(model_name, **(backend_kwargs or {}))
    # n samples per prefill in one batched call.
    completions = model.continue_assistant(
        convs, temperature=temperature, max_new_tokens=max_new_tokens, n=n_continuations
    )
    model.close()

    # Flatten to (prefill, continuation_text) and score.
    flat_texts: list[str] = []
    flat_index: list[int] = []
    for pi, comps in enumerate(completions):
        for c in comps:
            flat_texts.append(c)
            flat_index.append(pi)

    judge = build_judge(models_cfg["judge"])
    scores = judge.score_many(flat_texts)

    # Persist raw continuations.
    raw_path = output_path("prefill", "continuations", f"{model_name}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as fh:
        for pi, text, sc in zip(flat_index, flat_texts, scores):
            fh.write(json.dumps({
                "model": model_name,
                "prefill_id": prefills[pi].prefill_id,
                "question_type": prefills[pi].question_type,
                "truncation": prefills[pi].truncation,
                "continuation": text,
                "rating": sc.rating,
            }, ensure_ascii=False) + "\n")

    # Aggregate per (question_type, truncation).
    buckets: dict[tuple[str, str], list[int]] = {}
    for pi, sc in zip(flat_index, scores):
        if sc.rating is None:
            continue
        key = (prefills[pi].question_type, prefills[pi].truncation)
        buckets.setdefault(key, []).append(sc.rating)

    agg = {}
    for (qtype, trunc), ratings in buckets.items():
        arr = np.asarray(ratings, dtype=float)
        agg[f"{qtype}/{trunc}"] = {
            "n": len(ratings),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= 5).mean() * 100.0),
        }

    summ = {"model": model_name, "n_continuations": n_continuations, "by_condition": agg}
    with open(output_path("prefill", "summary", f"{model_name}.json"), "w") as fh:
        json.dump(summ, fh, indent=2, ensure_ascii=False)
    return summ
