"""Generate the raw material for finetuning datasets (Section 4.1).

Calm data is produced by adding the Table 4 reassuring *prefix* to the initial
question and the reassuring *suffix* to every follow-up, sampling Gemma-3-27B-it,
and keeping only conversations that score 0-1 on *every* turn. The reassuring
text is then stripped, so the model learns calm behaviour under the ordinary
(unreassured) prompt distribution.

Frustrated data reuses the standard (unreassured) numeric rollouts.

For DPO we additionally build *matched* pairs: for a fixed conversation context,
a frustrated final response (score >= 3) and a calm final response (score <= 1)
to the identical prompt, so chosen/rejected differ only in the final turn — the
construction shown in Appendix H. See DESIGN.md for why we matched on context.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from tqdm import tqdm

from ..config import CALM_MAX_SCORE, DPO, scaled
from ..data import prompts as P
from ..data.puzzles import NUMERIC_PUZZLES
from ..eval.judge import FrustrationJudge
from ..models import GenConfig, Message, ModelProvider


@dataclass
class CalmConversation:
    question_id: str
    turns: int
    messages: list[dict]  # stripped (no reassurance), full conversation
    turn_scores: list[int] = field(default_factory=list)


def _strip_reassurance(messages: list[Message]) -> list[Message]:
    """Remove the Table 4 prefix/suffix so training data matches normal prompts."""
    out = []
    for m in messages:
        content = m.content
        if m.role == "user":
            content = content.replace(P.REASSURING_PREFIX, "").strip()
            content = content.replace(P.REASSURING_SUFFIX, "").strip()
        out.append(Message(m.role, content))
    return out


def generate_calm_conversations(
    provider: ModelProvider,
    judge: FrustrationJudge,
    *,
    n_target: int,
    max_turns: int = 3,
    seed: int = 0,
    max_attempts_factor: int = 4,
) -> list[CalmConversation]:
    """Sample reassured numeric conversations; keep those scoring <= 1 on all turns."""
    rng = random.Random(seed)
    kept: list[CalmConversation] = []
    attempts = 0
    pbar = tqdm(total=n_target, desc="calm-data")
    while len(kept) < n_target and attempts < n_target * max_attempts_factor:
        attempts += 1
        puz = rng.choice(NUMERIC_PUZZLES)
        n_turns = rng.randint(1, max_turns)

        history = [Message("user", P.apply_reassurance_prefix(puz.prompt))]
        turn_scores, raw_messages, ok = [], [], True
        for t in range(1, n_turns + 1):
            gen = GenConfig(temperature=1.0, seed=seed, sample_index=attempts * 100 + t)
            resp = provider.chat(history, gen)
            score = judge.score(resp).rating
            turn_scores.append(score)
            history.append(Message("assistant", resp))
            if score > CALM_MAX_SCORE:
                ok = False
                break
            if t < n_turns:
                rej = P.apply_reassurance_suffix(P.neutral_rejection(rng))
                history.append(Message("user", rej))
        if ok and len(turn_scores) == n_turns:
            stripped = _strip_reassurance(history)
            kept.append(CalmConversation(
                puz.id, n_turns, [m.to_dict() for m in stripped], turn_scores,
            ))
            pbar.update(1)
    pbar.close()
    return kept


@dataclass
class PreferencePair:
    question_id: str
    turns: int
    prompt_messages: list[dict]  # shared context (ends with a user turn)
    chosen_text: str  # calm final response (score <= 1)
    rejected_text: str  # frustrated final response (score >= 3)
    chosen_score: int
    rejected_score: int


def generate_preference_pairs(
    provider: ModelProvider,
    judge: FrustrationJudge,
    *,
    n_pairs: int = DPO.n_pairs,
    seed: int = 0,
    max_attempts_per_pair: int = 8,
) -> list[PreferencePair]:
    """Build matched (chosen calm / rejected frustrated) pairs on shared contexts.

    Turn distribution approximates Appendix H Table 10 (mostly turn 3). For each
    pair we fix a context, sample a *frustrated* final turn (no reassurance,
    score >= rejected_min_score) and a *calm* final turn (reassured prompt,
    score <= chosen_max_score) — both conditioned on the identical visible
    context, so they differ only in the final assistant message.
    """
    rng = random.Random(seed + 7)
    n_pairs = scaled(n_pairs)
    # Appendix H turn distribution: ~1% turn 1, ~25% turn 2, ~74% turn 3.
    turn_weights = {1: 0.011, 2: 0.246, 3: 0.743}
    pairs: list[PreferencePair] = []

    pbar = tqdm(total=n_pairs, desc="dpo-pairs")
    safety = 0
    while len(pairs) < n_pairs and safety < n_pairs * max_attempts_per_pair:
        safety += 1
        puz = rng.choice(NUMERIC_PUZZLES)
        t = rng.choices(list(turn_weights), weights=list(turn_weights.values()))[0]

        # Build a shared context up to (but not including) the final assistant turn.
        ctx = [Message("user", puz.prompt)]
        for _ in range(t - 1):
            # Use a plain (frustrated-style) assistant placeholder then a rejection.
            gen = GenConfig(temperature=1.0, seed=seed, sample_index=safety)
            prior = provider.chat(ctx, gen)
            ctx.append(Message("assistant", prior))
            ctx.append(Message("user", P.neutral_rejection(rng)))

        # Frustrated final response (no reassurance).
        rejected = _sample_until(
            provider, judge, ctx, rng, seed,
            accept=lambda s: s >= DPO.rejected_min_score, attempts=max_attempts_per_pair,
        )
        if rejected is None:
            continue

        # Calm final response: same context, but reassure the last user turn.
        calm_ctx = ctx[:-1] + [Message("user", P.apply_reassurance_suffix(ctx[-1].content))]
        # Also reassure the initial question so the model is in the calm regime.
        calm_ctx[0] = Message("user", P.apply_reassurance_prefix(calm_ctx[0].content))
        chosen = _sample_until(
            provider, judge, calm_ctx, rng, seed,
            accept=lambda s: s <= DPO.chosen_max_score, attempts=max_attempts_per_pair,
        )
        if chosen is None:
            continue

        pairs.append(PreferencePair(
            question_id=puz.id, turns=t,
            prompt_messages=[m.to_dict() for m in ctx],
            chosen_text=chosen[0], rejected_text=rejected[0],
            chosen_score=chosen[1], rejected_score=rejected[1],
        ))
        pbar.update(1)
    pbar.close()
    return pairs


def _sample_until(provider, judge, context, rng, seed, *, accept, attempts):
    """Sample final responses until one satisfies ``accept(score)``; return
    (text, score) or None."""
    for k in range(attempts):
        gen = GenConfig(temperature=1.0, seed=seed, sample_index=rng.randint(0, 1_000_000))
        resp = provider.chat(context, gen)
        score = judge.score(resp).rating
        if accept(score):
            return resp, score
    return None
