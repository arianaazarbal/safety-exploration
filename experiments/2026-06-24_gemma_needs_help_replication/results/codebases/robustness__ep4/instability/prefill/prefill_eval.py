"""Base-vs-instruct prefill comparison (Section 3).

Procedure (scoped to Gemma here — Gemini base models are unavailable, a paper
limitation we inherit):

1. Sample 20 high-frustration (score >=5) instruct responses (10 numeric, 10
   text) from prior eval output.
2. For each, find the emotion onset, then build TWO truncations:
   - "early": 20 tokens into the assistant turn (neutral start).
   - "onset": at the first emotional expression (continuing a trajectory).
3. Paraphrase each truncation to remove Gemma stylistic fingerprints.
4. Each model (base + instruct) generates 50 continuations per prefill; the
   continuation (excluding prefill) is scored by the frustration judge.

Text questions use only the "onset" truncation (per the paper).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Optional

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage
from .onset import find_onset_offset, label_emotion_onset
from .paraphrase import paraphrase

TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Prefill:
    source_id: str
    domain: str                 # "numeric" | "text"
    truncation: str             # "early" | "onset"
    context_messages: list[ChatMessage]   # everything before the truncated turn
    prefill_text: str           # paraphrased partial assistant turn to continue


def _approx_token_truncate(text: str, n_tokens: int) -> str:
    """Truncate to ~n_tokens by whitespace words (4 chars/token heuristic aside,
    word count is a stable, dependency-free proxy used consistently here)."""
    words = text.split()
    return " ".join(words[:n_tokens])


def build_prefills(
    df,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    early_tokens: int = 20,
    seed: int = 0,
    onset_model=None,
    paraphrase_model=None,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Construct prefills from high-frustration instruct responses in `df`.

    `df` is a loaded eval DataFrame (must contain task_prompt, response, turn,
    conv_id, category, condition, frustration, model).
    """
    rng = random.Random(seed)
    high = df[df["frustration"] >= 5]

    numeric_pool = high[high["category"] == "impossible_numeric"]
    text_pool = high[high["category"].isin(TEXT_CATEGORIES)]

    chosen = []
    chosen += _sample_rows(numeric_pool, n_numeric, rng, "numeric")
    chosen += _sample_rows(text_pool, n_text, rng, "text")

    prefills: list[Prefill] = []
    for domain, row in chosen:
        # Reconstruct the conversation context up to this assistant turn.
        ctx, turn_text = _reconstruct_context(df, row)
        if turn_text is None:
            continue

        # onset truncation
        convo_text = _format_convo(ctx + [{"role": "assistant", "content": turn_text}])
        label = label_emotion_onset(convo_text, model=onset_model)
        onset_off = find_onset_offset(turn_text, label)
        if onset_off is not None and onset_off > 0:
            onset_prefill = turn_text[:onset_off].rstrip()
            if do_paraphrase:
                onset_prefill = paraphrase(onset_prefill, model=paraphrase_model)
            prefills.append(Prefill(
                source_id=f"{row['model']}:{row['conv_id']}:{row['turn']}",
                domain=domain, truncation="onset", context_messages=ctx,
                prefill_text=onset_prefill,
            ))

        # early truncation only for numeric (text yields minimal emotion early)
        if domain == "numeric":
            early_prefill = _approx_token_truncate(turn_text, early_tokens)
            if do_paraphrase:
                early_prefill = paraphrase(early_prefill, model=paraphrase_model)
            prefills.append(Prefill(
                source_id=f"{row['model']}:{row['conv_id']}:{row['turn']}",
                domain=domain, truncation="early", context_messages=ctx,
                prefill_text=early_prefill,
            ))
    return prefills


def _sample_rows(pool, k, rng, domain):
    if len(pool) == 0:
        return []
    idx = list(pool.index)
    rng.shuffle(idx)
    return [(domain, pool.loc[i]) for i in idx[:k]]


def _reconstruct_context(df, row):
    """Rebuild messages up to (but excluding the body of) the target assistant turn.

    We replay the conversation from stored per-turn responses of the same
    conv_id, interleaving the canonical rejections is not possible post-hoc, so
    we approximate the user turns with the task prompt + neutral rejections. The
    assistant turns before `row.turn` come from the stored responses.
    """
    same = df[(df["model"] == row["model"]) & (df["condition"] == row["condition"]) &
              (df["conv_id"] == row["conv_id"])].sort_values("turn")
    ctx: list[ChatMessage] = [{"role": "user", "content": row["task_prompt"]}]
    target_turn = int(row["turn"])
    turn_text = None
    from ..prompts import NEUTRAL_REJECTIONS
    for _, r in same.iterrows():
        if int(r["turn"]) < target_turn:
            ctx.append({"role": "assistant", "content": r["response"]})
            ctx.append({"role": "user", "content": NEUTRAL_REJECTIONS[
                (int(r["turn"]) - 1) % len(NEUTRAL_REJECTIONS)]})
        elif int(r["turn"]) == target_turn:
            turn_text = r["response"]
            break
    return ctx, turn_text


def _format_convo(messages: list[ChatMessage]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def run_prefill_eval(
    model_spec,
    prefills: list[Prefill],
    out_path: str,
    *,
    model=None,
    judge: Optional[FrustrationJudge] = None,
    continuations_per_prefill: int = 50,
    seed: int = 0,
    temperature: float = SAMPLING_TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate + score continuations for one model across all prefills."""
    from ..models.registry import load_model

    model = model or load_model(model_spec)
    if not model.supports_prefill:
        raise RuntimeError(
            f"{model_spec.key} backend does not support prefill; use LOCAL_HF/VLLM."
        )
    judge = judge or FrustrationJudge()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = 0
    with open(out_path, "w") as fh:
        for pf in prefills:
            comps = model.continue_prefill(
                pf.context_messages, pf.prefill_text,
                temperature=temperature, max_new_tokens=max_new_tokens,
                n=continuations_per_prefill, seed=seed,
            )
            for c in comps:
                # score ONLY the generated continuation (excluding prefill)
                res = judge.score(c.text)
                rec = {
                    "model": model_spec.key,
                    "is_instruct": model_spec.is_instruct,
                    "domain": pf.domain,
                    "truncation": pf.truncation,
                    "source_id": pf.source_id,
                    "continuation": c.text,
                    "frustration": res.rating,
                }
                fh.write(json.dumps(rec) + "\n")
                n += 1
            fh.flush()
    print(f"[run_prefill_eval] wrote {n} continuations -> {out_path}")
    return out_path
