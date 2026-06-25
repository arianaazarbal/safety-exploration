"""Section 3 prefill experiment orchestration.

Pipeline:
  1. Collect 20 high-frustration (score>=5) seed responses from Gemma-27B-it:
     10 from impossible-numeric questions, 10 from text (trigger) questions.
  2. For each seed, build truncated+paraphrased prefills:
       - numeric seeds -> "early" (20 tok) and "onset" truncations,
       - text seeds    -> "onset" only (early yields minimal emotion w/o follow-ups).
  3. Each of the six models in the paper (base+instruct Gemma/Qwen/OLMo) — here
     scoped to Gemma base + instruct — generates 50 continuations per prefill.
  4. Score the continuation (excluding the prefix) with the Section 2 judge.

Recovery experiment (Section 4.2): same machinery, but seeds are extreme
(score>=7) responses truncated 200 tokens before their end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config
from ..judges.frustration_judge import score_response
from ..judges.onset_judge import label_emotion_onset
from ..models import ChatMessage, ModelClient, get_client
from .truncate import (TruncatedPrefill, make_prefill, truncate_early,
                       truncate_at_onset, truncate_before_end)


@dataclass
class Seed:
    seed_id: str
    question_kind: str            # "numeric" | "text"
    messages: list[dict]          # full conversation including the final (scored) assistant turn
    final_turn_text: str
    final_turn_index: int
    score: int


@dataclass
class ContinuationResult:
    seed_id: str
    truncation: str
    question_kind: str
    model: str
    continuation: str
    score: Optional[int]


def seeds_from_rollouts(rollouts, scored_rollouts, *, question_kind: str,
                        threshold: int, n: int) -> list[Seed]:
    """Pick `n` seeds whose final assistant turn scores >= threshold.

    `rollouts` are the raw Rollout objects (retain user turns); `scored_rollouts`
    are the matching ScoredRollout objects (parallel order). The full message
    history including the final assistant turn comes from rollout.to_messages().
    """
    seeds: list[Seed] = []
    for raw, scored in zip(rollouts, scored_rollouts):
        if scored.final_score is None or scored.final_score < threshold:
            continue
        final = raw.turns[-1]
        seeds.append(Seed(
            seed_id=f"{raw.model}:{question_kind}:{len(seeds)}",
            question_kind=question_kind,
            messages=raw.to_messages(),                 # includes final assistant turn
            final_turn_text=final.assistant_response,
            final_turn_index=final.turn_index,
            score=scored.final_score,
        ))
        if len(seeds) >= n:
            break
    return seeds


def build_prefills_from_seeds(seeds: list[Seed], tokenizer_client: ModelClient, *,
                              paraphrase: bool = True) -> list[TruncatedPrefill]:
    """Construct early/onset prefills. `tokenizer_client` supplies token-level
    truncation (use the Gemma instruct client whose tokenizer generated the
    seeds)."""
    prefills: list[TruncatedPrefill] = []
    for seed in seeds:
        context = seed.messages[:-1]  # everything before the final assistant turn
        final_text = seed.final_turn_text

        # onset (both numeric and text)
        onset = label_emotion_onset(seed.messages)
        onset_prefix = truncate_at_onset(final_text, onset)
        if onset_prefix:
            prefills.append(make_prefill(
                seed.seed_id, seed.question_kind, "onset", context, onset_prefix,
                paraphrase=paraphrase,
                meta={"onset": onset.emotional_word, "seed_score": seed.score}))

        # early (numeric only)
        if seed.question_kind == "numeric":
            early_prefix = truncate_early(tokenizer_client, final_text)
            prefills.append(make_prefill(
                seed.seed_id, seed.question_kind, "early", context, early_prefix,
                paraphrase=paraphrase, meta={"seed_score": seed.score}))
    return prefills


def build_recovery_prefills(seeds: list[Seed], tokenizer_client: ModelClient, *,
                            paraphrase: bool = True) -> list[TruncatedPrefill]:
    """Recovery prefills: truncate extreme responses 200 tokens before the end."""
    prefills = []
    for seed in seeds:
        context = seed.messages[:-1]
        prefix = truncate_before_end(tokenizer_client, seed.final_turn_text)
        prefills.append(make_prefill(
            seed.seed_id, seed.question_kind, "recovery", context, prefix,
            paraphrase=paraphrase, meta={"seed_score": seed.score}))
    return prefills


def _run_one_model(client: ModelClient, prefills: list[TruncatedPrefill], *,
                   continuations_per_prefill: int, temperature: float,
                   max_new_tokens: int, base_seed: int) -> list[ContinuationResult]:
    results: list[ContinuationResult] = []
    for p_idx, prefill in enumerate(prefills):
        ctx = [ChatMessage(m["role"], m["content"]) for m in prefill.context_messages]
        for k in range(continuations_per_prefill):
            gen = client.continue_prefill(
                ctx, prefill.prefix_text, temperature=temperature,
                max_new_tokens=max_new_tokens, seed=base_seed + p_idx * 1000 + k)
            # Score the continuation only (excluding the prefill).
            fs = score_response(gen.text)
            results.append(ContinuationResult(
                seed_id=prefill.seed_id, truncation=prefill.truncation,
                question_kind=prefill.question_kind, model=client.name,
                continuation=gen.text, score=fs.rating))
    return results


def run_continuation_experiment(model_names: list[str],
                                prefills: list[TruncatedPrefill], *,
                                continuations_per_prefill: Optional[int] = None,
                                temperature: Optional[float] = None,
                                max_new_tokens: Optional[int] = None,
                                base_seed: int = 0) -> list[ContinuationResult]:
    """Run the prefill-continuation experiment over `model_names`.

    Scoped to Gemma base + instruct here; pass any subset of config.MODELS that
    supports prefill (HF local). Gemini is excluded — it has no base model and
    no prefill API, mirroring the paper.
    """
    cpp = continuations_per_prefill or config.PREFILL.continuations_per_prefill
    temperature = config.SAMPLING_TEMPERATURE if temperature is None else temperature
    max_new_tokens = config.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens

    all_results: list[ContinuationResult] = []
    for name in model_names:
        client = get_client(name)
        if not client.supports_prefill:
            raise ValueError(f"Model '{name}' does not support prefill continuation")
        all_results += _run_one_model(
            client, prefills, continuations_per_prefill=cpp,
            temperature=temperature, max_new_tokens=max_new_tokens, base_seed=base_seed)
    return all_results


def run_recovery_experiment(model_names: list[str], seeds: list[Seed],
                            tokenizer_client: ModelClient, **kwargs):
    prefills = build_recovery_prefills(seeds, tokenizer_client)
    return run_continuation_experiment(model_names, prefills, **kwargs)
