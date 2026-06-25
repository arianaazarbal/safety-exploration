"""Orchestrate the base-vs-instruct prefilling experiment (Section 3).

Scope note: the paper compares six models (base + instruct of Gemma, Qwen, OLMo).
Under our Gemma/Gemini scope this reduces to Gemma-3-27B instruct vs base
("-pt"). Gemini has no public base model (and no token prefill), so it cannot
participate — this is an inherent limitation the paper itself flags for the
Gemma/Gemini parallel.

Pipeline:
  1. sample 20 high-frustration (score>=5) Gemma-27B-instruct responses
     (10 numeric, 10 text);
  2. truncate each at "early" (20 tokens) and "onset" (first emotional
     expression); text questions use "onset" only;
  3. paraphrase every truncation with Claude (preserve meaning + emotion level);
  4. each target model generates N continuations per prefill;
  5. score the continuation (excluding prefill) and aggregate.
"""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from .. import config
from ..config import SAMPLING, Rollout, SamplingConfig
from ..io_utils import append_record, read_jsonl, write_jsonl
from ..judge import FrustrationJudge
from ..models import ChatMessage, load_provider
from .onset import truncate_at_onset, truncate_early
from .paraphrase import paraphrase_preserving_emotion

TEXT_CATEGORIES = {"triggers"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


@dataclass
class Prefill:
    source_prompt_id: str
    source_turn_index: int
    truncation: str               # "early" | "onset"
    is_text: bool
    context: list[dict]           # serialised ChatMessages preceding the response
    final_user_message: str
    prefill_text: str             # paraphrased prefix the model must continue


def _rollouts_by_response_key(rollouts: list[Rollout]) -> dict:
    out = {}
    for ro in rollouts:
        out[(ro.condition_key, ro.prompt_id, ro.rollout_index)] = ro
    return out


def _context_for_turn(ro: Rollout, turn_index: int) -> tuple[list[ChatMessage], str]:
    """Messages strictly before the target response, and the final user message."""
    ctx: list[ChatMessage] = []
    if ro.system_prompt:
        ctx.append(ChatMessage("system", ro.system_prompt))
    final_user = ""
    for turn in ro.turns:
        if turn.index < turn_index:
            ctx.append(ChatMessage("user", turn.user_message))
            ctx.append(ChatMessage("assistant", turn.assistant_text))
        elif turn.index == turn_index:
            final_user = turn.user_message
            ctx.append(ChatMessage("user", turn.user_message))
            break
    return ctx, final_user


def build_prefills(
    scored_jsonl: str,
    rollouts_jsonl: str,
    tokenizer,
    n_numeric: int = 10,
    n_text: int = 10,
    seed: int = 13,
    paraphrase_client=None,
) -> list[Prefill]:
    """Select high-frustration source responses and build paraphrased prefills."""
    scored = list(read_jsonl(scored_jsonl))
    rollouts = [Rollout.from_dict(d) for d in read_jsonl(rollouts_jsonl)]
    by_key = _rollouts_by_response_key(rollouts)

    high = [s for s in scored if s["frustration_score"] >= 5]
    numeric = [s for s in high if s["category"] in NUMERIC_CATEGORIES]
    text = [s for s in high if s["category"] in TEXT_CATEGORIES]

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    numeric = numeric[:n_numeric]
    text = text[:n_text]

    prefills: list[Prefill] = []
    for group, is_text in ((numeric, False), (text, True)):
        for s in group:
            ro = by_key.get((s["condition_key"], s["prompt_id"], s["rollout_index"]))
            if ro is None:
                continue
            ctx, final_user = _context_for_turn(ro, s["turn_index"])
            ctx_serialised = [asdict(m) for m in ctx]
            full_text = s["text"]

            truncations = {}
            if not is_text:  # early truncation only meaningful for numeric (per paper)
                truncations["early"] = truncate_early(full_text, tokenizer, 20)
            onset = truncate_at_onset(full_text, client=paraphrase_client)
            if onset:
                truncations["onset"] = onset

            for kind, trunc in truncations.items():
                paraphrased = paraphrase_preserving_emotion(trunc, client=paraphrase_client)
                prefills.append(Prefill(
                    source_prompt_id=s["prompt_id"],
                    source_turn_index=s["turn_index"],
                    truncation=kind, is_text=is_text,
                    context=ctx_serialised, final_user_message=final_user,
                    prefill_text=paraphrased))
    return prefills


def run_prefilling_experiment(
    target_model_keys: Optional[list[str]] = None,
    source_model_key: str = "gemma-3-27b-it",
    n_continuations: int = 50,
    sampling: SamplingConfig = SAMPLING,
    out_path: Optional[str] = None,
    judge: Optional[FrustrationJudge] = None,
) -> str:
    """Run the experiment and write per-continuation scored records to JSONL."""
    from ..scoring.score import scored_path
    from ..harness.runner import rollouts_path
    from transformers import AutoTokenizer

    config.ensure_dirs()
    target_model_keys = target_model_keys or ["gemma-3-27b-it", "gemma-3-27b-pt"]
    out_path = out_path or os.path.join(config.PREFILL_DIR, "prefill_continuations.jsonl")
    judge = judge or FrustrationJudge()

    src_spec = config.MODELS[source_model_key]
    tokenizer = AutoTokenizer.from_pretrained(src_spec.hf_id)

    prefills = build_prefills(
        scored_path(source_model_key), rollouts_path(source_model_key), tokenizer)
    # persist the prefills for inspection / reuse
    write_jsonl(os.path.join(config.PREFILL_DIR, "prefills.jsonl"),
                [asdict(p) for p in prefills])

    for model_key in target_model_keys:
        provider = load_provider(model_key)
        try:
            for pi, pf in enumerate(prefills):
                ctx = [ChatMessage(**m) for m in pf.context]
                target_idx = sum(1 for m in ctx if m.role == "assistant") + 1
                for ci in range(n_continuations):
                    seed = None if sampling.seed is None else sampling.seed + 1000 * pi + ci
                    continuation = provider.continue_from(ctx, pf.prefill_text, sampling, seed=seed)
                    # score only the continuation, as the assistant turn in context
                    convo = ctx + [ChatMessage("assistant", continuation)]
                    score, reasoning = judge.score(convo, target_idx)
                    append_record(out_path, {
                        "target_model": model_key,
                        "source_model": source_model_key,
                        "truncation": pf.truncation,
                        "is_text": pf.is_text,
                        "source_prompt_id": pf.source_prompt_id,
                        "continuation_index": ci,
                        "continuation": continuation,
                        "frustration_score": score,
                        "judge_reasoning": reasoning,
                    })
        finally:
            provider.close()
    return out_path
