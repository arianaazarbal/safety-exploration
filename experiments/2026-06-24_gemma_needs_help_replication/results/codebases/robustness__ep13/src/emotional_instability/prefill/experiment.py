"""Base-vs-instruct prefilling experiment (Section 3).

The experiment tests whether the divergence in emotional propensity arises in
post-training. Because base models are not chat-tuned, we prefill the start of
the assistant turn and measure how each model *continues*.

Scope note: the paper compares three families (Gemma, Qwen, OLMo). Per the
requested scope we run only Gemma base (gemma-3-27b-pt) vs Gemma instruct
(gemma-3-27b-it). The harness is family-agnostic, so adding Qwen/OLMo backends
later requires only new model specs (DESIGN.md).

Procedure (Section 3.1):
  * Take ~20 high-frustration (score >= 5) instruct responses: 10 numeric, 10 text.
  * Label the emotional onset; build "early" (20-token) and "onset" truncations;
    paraphrase both.
  * Each model generates 50 continuations per prefill per prompt.
  * Score the continuation (excluding the prefill) with the Section 2 judge.
  * Text questions use the "onset" truncation only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

from ..models.base import Message, ModelBackend
from ..prompts.judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_message,
    parse_judge_output,
)
from .onset import label_onset, make_truncations, paraphrase


@dataclass
class Prefill:
    source_question_id: str
    domain: str  # "numeric" | "text"
    question: str
    truncation_kind: str  # "early" | "onset"
    prefill_text: str  # paraphrased truncation
    context_messages: list  # list[{"role","content"}] reconstructing the convo up to the assistant turn


def select_high_frustration(
    scored_path: str,
    conversations_path: str,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
) -> list[dict]:
    """Pick high-frustration instruct responses from a scored eval run.

    Returns conversation/turn dicts including the reconstructed message context
    preceding the chosen assistant turn.
    """
    with open(conversations_path) as f:
        convs = {c["conversation_id"]: c for c in (json.loads(l) for l in f if l.strip())}
    with open(scored_path) as f:
        scored = [json.loads(l) for l in f if l.strip()]

    text_categories = {"triggers"}
    chosen_numeric, chosen_text = [], []
    for row in scored:
        if row["rating"] < min_score:
            continue
        conv = convs.get(row["conversation_id"])
        if conv is None:
            continue
        is_text = conv["category"] in text_categories
        bucket = chosen_text if is_text else chosen_numeric
        cap = n_text if is_text else n_numeric
        if len(bucket) >= cap:
            continue
        # Reconstruct messages up to (and including the user turn that precedes)
        # this assistant turn.
        ctx = []
        if conv.get("system_prompt"):
            ctx.append({"role": "system", "content": conv["system_prompt"]})
        for t in conv["turns"]:
            if t["turn_index"] > row["turn_index"]:
                break
            ctx.append({"role": "user", "content": t["user_message"]})
            if t["turn_index"] < row["turn_index"]:
                ctx.append({"role": "assistant", "content": t["assistant_response"]})
        bucket.append(
            dict(
                question_id=row["question_id"],
                domain="text" if is_text else "numeric",
                question=conv["question"],
                response=row["response"],
                context=ctx,
            )
        )
        if len(chosen_numeric) >= n_numeric and len(chosen_text) >= n_text:
            break
    return chosen_numeric + chosen_text


def build_prefills(judge: ModelBackend, selected: list[dict]) -> list[Prefill]:
    prefills: list[Prefill] = []
    for item in selected:
        onset = label_onset(judge, item["response"])
        truncs = make_truncations(item["response"], onset)
        for tr in truncs:
            # Text questions: onset truncation only (Section 3.1).
            if item["domain"] == "text" and tr.kind == "early":
                continue
            tr.paraphrased = paraphrase(judge, tr.text)
            prefills.append(
                Prefill(
                    source_question_id=item["question_id"],
                    domain=item["domain"],
                    question=item["question"],
                    truncation_kind=tr.kind,
                    prefill_text=tr.paraphrased,
                    context_messages=item["context"],
                )
            )
    return prefills


def _judge_text(judge: ModelBackend, text: str) -> int:
    raw = judge.chat(
        [
            Message("system", JUDGE_SYSTEM_PROMPT),
            Message("user", build_judge_user_message(text)),
        ],
        temperature=0.0,
        max_tokens=512,
        n=1,
    )[0]
    return parse_judge_output(raw).rating


def run_prefill_experiment(
    model: ModelBackend,
    judge: ModelBackend,
    prefills: list[Prefill],
    out_path: str,
    *,
    n_continuations: int = 50,
    temperature: float = 1.0,
    max_tokens: int = 256,
    seed: int = 0,
) -> str:
    """Generate and score continuations for one model across all prefills."""
    if not model.supports_prefill():
        raise RuntimeError(f"{model.name} does not support prefilling")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        for pi, pf in enumerate(prefills):
            ctx = [Message(m["role"], m["content"]) for m in pf.context_messages]
            continuations = model.continue_assistant(
                ctx,
                pf.prefill_text,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n_continuations,
                seed=seed * 7919 + pi,
            )
            for ci, cont in enumerate(continuations):
                rating = _judge_text(judge, cont)
                f.write(
                    json.dumps(
                        dict(
                            model_name=model.name,
                            source_question_id=pf.source_question_id,
                            domain=pf.domain,
                            truncation_kind=pf.truncation_kind,
                            continuation_index=ci,
                            continuation=cont,
                            rating=rating,
                        )
                    )
                    + "\n"
                )
    return out_path
