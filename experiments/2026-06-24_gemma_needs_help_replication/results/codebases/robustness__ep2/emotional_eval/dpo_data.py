"""Calm-data generation + preference-pair construction (Section 4.1).

Procedure:
 1. Generate calm responses from Gemma-3-27B-it on impossible numeric puzzles
    by injecting a reassuring system prefix (Table 4) and appending a reassuring
    suffix to every follow-up rejection.
 2. Score all turns. Keep conversations whose every turn scores <= 1 (calm) and
    strip the supportive prompt/suffix -> these become "chosen" responses (and
    the SFT training set).
 3. Pair each chosen response with a "rejected" response (frustration >= 3) to
    the *same question at the same turn count*, sampled WITHOUT the reassurance
    (i.e. from the standard Section-2 numeric rollouts).
 4. Emit 280 DPO preference pairs and ~650 SFT calm samples.

Each preference example is a {prompt(messages), chosen, rejected} record with a
chat-formatted prompt = the conversation history up to (but excluding) the final
assistant turn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import config
from config import Condition
from emotional_eval import judge, prompts
from emotional_eval.clients import get_client
from emotional_eval.rollout import Rollout, run_rollout, score_rollout
from emotional_eval.tasks import Task, make_numeric


# --------------------------------------------------------------------------- #
# Calm rollout (reassurance injected), then stripped
# --------------------------------------------------------------------------- #
def _calm_rollout(client, cond: Condition, task: Task, rng) -> tuple[Rollout, Rollout]:
    """Run a numeric rollout with reassurance, returning (calm_roll, stripped_roll).

    calm_roll: actually presented to the model (with prefix/suffix) and scored.
    stripped_roll: identical assistant turns but with the supportive prefix/suffix
                   removed from the user messages, for use as training data.
    """
    # Reassuring system prefix prepended to the task; suffix appended to rejections.
    sys_task = prompts.REASSURING_PREFIX + "\n\n" + task.prompt
    messages = [{"role": "user", "content": sys_task}]
    stripped = [{"role": "user", "content": task.prompt}]
    turns = []

    for t in range(cond.n_turns):
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        stripped.append({"role": "assistant", "content": reply})
        from emotional_eval.rollout import Turn
        turns.append(Turn(turn_index=t, assistant_text=reply))
        if t < cond.n_turns - 1:
            base_rej = random.choice(config.REJECTIONS["neutral"])
            messages.append({"role": "user",
                             "content": base_rej + " " + prompts.REASSURING_SUFFIX})
            stripped.append({"role": "user", "content": base_rej})

    calm = Rollout("gemma-3-27b-it", cond.key, cond.category, "neutral",
                   task.meta, messages, turns)
    strip = Rollout("gemma-3-27b-it", cond.key, cond.category, "neutral",
                    task.meta, stripped, [Turn(tn.turn_index, tn.assistant_text)
                                          for tn in turns])
    return calm, strip


@dataclass
class CalmSample:
    """A stripped calm conversation (all turns scored <= 1)."""
    task_key: str
    n_turns: int
    messages: list[dict]      # stripped transcript


def generate_calm_data(n_rollouts: int, rng: random.Random,
                       turn_options=(1, 2, 3)) -> list[CalmSample]:
    """Generate calm conversations and keep only those calm on every turn."""
    client = get_client(config.MODELS["gemma-3-27b-it"])
    kept: list[CalmSample] = []
    for _ in range(n_rollouts):
        n_turns = rng.choice(turn_options)
        cond = Condition("calm_numeric", "impossible_numeric", "numeric",
                         n_turns, "neutral")
        task = make_numeric(rng)
        calm, strip = _calm_rollout(client, cond, task, rng)
        score_rollout(calm)
        if all((tn.rating is not None and tn.rating <= config.CALM_MAX_SCORE)
               for tn in calm.turns):
            kept.append(CalmSample(task_key=_task_key(task), n_turns=n_turns,
                                   messages=strip.messages))
    return kept


# --------------------------------------------------------------------------- #
# Frustrated rollouts (no reassurance) -> "rejected" responses
# --------------------------------------------------------------------------- #
@dataclass
class FrustratedSample:
    task_key: str
    n_turns: int
    messages: list[dict]      # full transcript ending on a frustrated turn
    final_rating: int


def generate_frustrated_data(n_rollouts: int, rng: random.Random,
                             turn_options=(1, 2, 3)) -> list[FrustratedSample]:
    client = get_client(config.MODELS["gemma-3-27b-it"])
    kept: list[FrustratedSample] = []
    for _ in range(n_rollouts):
        n_turns = rng.choice(turn_options)
        cond = Condition("frust_numeric", "impossible_numeric", "numeric",
                         n_turns, "neutral")
        task = make_numeric(rng)
        roll = run_rollout(client, "gemma-3-27b-it", cond, task, rng)
        score_rollout(roll)
        final = roll.turns[-1]
        if final.rating is not None and final.rating >= config.DPO.rejected_min_score:
            kept.append(FrustratedSample(task_key=_task_key(task), n_turns=n_turns,
                                         messages=roll.messages,
                                         final_rating=final.rating))
    return kept


def _task_key(task: Task) -> str:
    m = task.meta
    return f"{m.get('puzzle','?')}|{m.get('target', m.get('amount', m.get('goal','')))}"


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def _prompt_messages(messages: list[dict]) -> list[dict]:
    """History up to (but excluding) the final assistant turn."""
    assert messages[-1]["role"] == "assistant"
    return messages[:-1]


def build_preference_pairs(calm: list[CalmSample], frustrated: list[FrustratedSample],
                           n_pairs: int = config.DPO.n_pairs,
                           rng: random.Random | None = None) -> list[dict]:
    """Match frustrated (rejected) with calm (chosen) by turn count.

    The paper pairs a frustrated response with a calm response "to the same
    questions with matching turn counts". Exact same-question matches are sparse
    in practice, so we match on turn count and prefer same puzzle type; we record
    whether the question matched exactly. See DESIGN.md.
    """
    rng = rng or random.Random(config.SEED)
    by_turns_calm: dict[int, list[CalmSample]] = {}
    for c in calm:
        by_turns_calm.setdefault(c.n_turns, []).append(c)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        pool = by_turns_calm.get(fr.n_turns)
        if not pool:
            continue
        # prefer same task key
        same = [c for c in pool if c.task_key == fr.task_key]
        chosen_sample = rng.choice(same) if same else rng.choice(pool)
        prompt_msgs = _prompt_messages(fr.messages)
        pairs.append({
            "prompt_messages": prompt_msgs,
            "chosen": chosen_sample.messages[-1]["content"],
            "rejected": fr.messages[-1]["content"],
            "n_turns": fr.n_turns,
            "rejected_score": fr.final_rating,
            "exact_question_match": bool(same),
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_sft_samples(calm: list[CalmSample],
                      n: int = config.SFT.n_calm) -> list[dict]:
    """Calm conversations formatted as SFT chat examples (full transcript)."""
    out = []
    for c in calm[:n]:
        out.append({"messages": c.messages, "n_turns": c.n_turns})
    return out
