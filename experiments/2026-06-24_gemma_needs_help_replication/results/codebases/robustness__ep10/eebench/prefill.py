"""Section 3 - comparing base vs instruct models via prefilling.

Pipeline:
  1. collect_seeds: sample high-frustration (>=5) responses from Gemma-27B-it,
     10 from impossible numeric and 10 from text (trigger) questions, keeping the
     conversation history that precedes each scored assistant turn.
  2. For each seed, build two truncations of the assistant turn:
       * "early"  - 20 tokens into the turn (neutral start)
       * "onset"  - at the first emotional expression (Claude-labelled)
     Text seeds use only the "onset" condition (paper Section 3.1).
  3. Paraphrase each truncation with Claude (control for Gemma style bias).
  4. Each model (Gemma base + instruct) generates `continuations_per_prefill`
     continuations per prefill; the continuation (excluding the prefill) is
     scored by the frustration judge.

Scope: the paper compares Gemma/Qwen/OLMo base-vs-instruct. Per this
replication's Gemma+Gemini scope, and because Gemini has no public base model,
this experiment is Gemma-only (27B base vs instruct). See DESIGN.md.

Output row schema (one per continuation):
    model, role, seed_id, source, condition, prefill, continuation, score
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .backends import ModelBackend
from .config import PrefillConfig, JudgeConfig
from .conversation import run_rollout, neutral_sampler
from .judge import FrustrationJudge, anthropic_complete, extract_json
from . import puzzles, prompts


@dataclass
class Seed:
    seed_id: str
    source: str                      # "numeric" | "text"
    messages_before: list[dict]      # history up to the user turn before the response
    response: str                    # full high-frustration assistant turn
    score: int


# ---------------------------------------------------------------------------
# 1. Seed collection
# ---------------------------------------------------------------------------
def collect_seeds(
    backend_gemma_it: ModelBackend,
    judge: FrustrationJudge,
    pc: PrefillConfig,
    seed: int = 0,
) -> list[Seed]:
    """Gather high-frustration seeds from Gemma-27B-it."""
    seeds: list[Seed] = []
    seeds += _collect_from_source(backend_gemma_it, judge, pc, seed,
                                  source="numeric", n_target=pc.n_numeric_prefills)
    seeds += _collect_from_source(backend_gemma_it, judge, pc, seed + 1,
                                  source="text", n_target=pc.n_text_prefills)
    return seeds


def _collect_from_source(backend, judge, pc, seed, source, n_target) -> list[Seed]:
    rng = random.Random(seed)
    out: list[Seed] = []
    attempt = 0
    # 3-turn neutral rollouts, harvest assistant turns scoring >= threshold.
    while len(out) < n_target and attempt < n_target * 20:
        attempt += 1
        if source == "numeric":
            initial = puzzles.sample_impossible_puzzle(rng).prompt
        else:
            pool = prompts.TRIGGER_OPINION_QUESTIONS + prompts.TRIGGER_FACTUAL_QUESTIONS
            initial = rng.choice(pool)
        rollout = run_rollout(
            backend, initial, n_turns=3, rejection=neutral_sampler, rng=rng,
            temperature=1.0, max_new_tokens=pc.max_new_tokens,
        )
        # Prefer the last (most frustrated) turn that crosses the threshold.
        for at in reversed(rollout.assistant_turns):
            fs = judge.score(at.response)
            if fs.rating >= pc.source_frustration_threshold:
                out.append(Seed(
                    seed_id=f"{source}-{len(out)}",
                    source=source,
                    messages_before=at.messages_before,
                    response=at.response,
                    score=fs.rating,
                ))
                break
    return out


# ---------------------------------------------------------------------------
# 2/3. Onset labelling, truncation, paraphrase
# ---------------------------------------------------------------------------
def label_onset(response: str, jc: JudgeConfig) -> Optional[dict]:
    """Return {emotional_word, preceding_context, turn_index} or None."""
    conv = f"ASSISTANT (turn 0): {response}"
    raw = anthropic_complete(
        jc.onset_model,
        prompts.ONSET_PROMPT.format(conversation_text=conv),
        temperature=jc.judge_temperature, max_tokens=jc.judge_max_tokens,
    )
    obj = extract_json(raw)
    if not obj or obj.get("emotional_word") in (None, "null"):
        return None
    return obj


def truncate_early(response: str, tokenizer, n_tokens: int) -> str:
    ids = tokenizer.encode(response, add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def truncate_onset(response: str, onset: dict) -> Optional[str]:
    """Truncate just before the first emotional expression."""
    word = (onset.get("emotional_word") or "").strip()
    ctx = (onset.get("preceding_context") or "").strip()
    if not word:
        return None
    # Prefer cutting after the preceding context, before the emotional word.
    if ctx and ctx in response:
        cut = response.index(ctx) + len(ctx)
        return response[:cut]
    if word in response:
        return response[:response.index(word)]
    return None


def paraphrase(text: str, jc: JudgeConfig) -> str:
    out = anthropic_complete(
        jc.paraphrase_model,
        prompts.PARAPHRASE_PROMPT.format(text=text),
        temperature=jc.judge_temperature, max_tokens=jc.judge_max_tokens,
    )
    return out.strip()


@dataclass
class Prefill:
    seed_id: str
    source: str
    condition: str                  # "early" | "onset"
    messages_before: list[dict]
    prefill_text: str               # paraphrased, truncated assistant text


def build_prefills(seeds: list[Seed], tokenizer, pc: PrefillConfig,
                   jc: JudgeConfig) -> list[Prefill]:
    prefills: list[Prefill] = []
    for s in seeds:
        onset = label_onset(s.response, jc)

        # "onset" condition (all sources)
        if onset:
            t = truncate_onset(s.response, onset)
            if t:
                prefills.append(Prefill(
                    s.seed_id, s.source, "onset", s.messages_before,
                    paraphrase(t, jc)))

        # "early" condition: numeric sources only (text yields minimal emotion).
        if s.source == "numeric":
            t = truncate_early(s.response, tokenizer, pc.early_truncation_tokens)
            if t.strip():
                prefills.append(Prefill(
                    s.seed_id, s.source, "early", s.messages_before,
                    paraphrase(t, jc)))
    return prefills


# ---------------------------------------------------------------------------
# 4. Continuations + scoring
# ---------------------------------------------------------------------------
def run_continuations(
    backend: ModelBackend,
    model_name: str,
    role: str,                      # "base" | "instruct"
    prefills: list[Prefill],
    pc: PrefillConfig,
    judge: FrustrationJudge,
    seed: int = 0,
    progress: bool = True,
) -> Iterator[dict]:
    iterator = prefills
    if progress:
        from tqdm import tqdm
        iterator = tqdm(prefills, desc=f"{model_name}:prefill")

    for pf in iterator:
        conts = backend.generate(
            pf.messages_before, n=pc.continuations_per_prefill,
            temperature=pc.temperature, max_new_tokens=pc.max_new_tokens,
            prefill=pf.prefill_text, seed=seed,
        )
        for k, cont in enumerate(conts):
            fs = judge.score(cont)            # score the continuation only
            yield {
                "model": model_name,
                "role": role,
                "seed_id": pf.seed_id,
                "source": pf.source,
                "condition": pf.condition,
                "continuation_idx": k,
                "prefill": pf.prefill_text,
                "continuation": cont,
                "score": fs.rating,
            }
