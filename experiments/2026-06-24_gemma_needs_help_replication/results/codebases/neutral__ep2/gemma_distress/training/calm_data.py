"""Generate calm / frustrated rollouts for finetuning data (Section 4.1).

Calm data: sample Gemma-3-27B-it on impossible numeric puzzles with the
reassuring prefix prepended to the initial prompt and the reassuring suffix
appended to each follow-up (Table 4). These additions reduce mean frustration
(4.3 -> 2.0 over 3 turns). We then filter to conversations scoring 0-1 on every
turn and *strip* the supportive prompts/suffixes when storing the context, so
the resulting training context is in the standard (un-reassured) framing.

Frustrated data: the same puzzles run *without* reassurance (standard
rejections), keeping responses scoring >= 3 as the DPO "rejected" pool.

Each rollout is returned as a record carrying, per assistant turn, the standard
(stripped) context and the response + its frustration score, so that
``datasets.py`` can build SFT samples and DPO pairs by (task_id, turn_index).
"""

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

import config

from .. import prompts
from ..judge.frustration import FrustrationJudge
from ..models.base import ModelBackend
from ..schemas import Message
from ..tasks import Task
from ..tasks import rejections as rej
from ..tasks.puzzles import generate_impossible_numeric


def _rollout_record(
    backend: ModelBackend,
    task: Task,
    *,
    n_turns: int,
    reassure: bool,
    system_prompt: str | None,
    rng: random.Random,
    judge: FrustrationJudge,
) -> dict:
    """Run one rollout; return per-turn records with standard (stripped) context.

    `reassure` toggles the Table-4 prefix/suffix additions. `system_prompt`
    optionally sets a system message (used for the Appendix-F 'teacher' variant).
    """
    neutral = rej.neutral_sequence(n_turns - 1, rng)

    # What the model actually sees (may include reassurance / system prompt).
    # The Gemma chat template rejects a dedicated `system` role, so the
    # teacher-variant system prompt is folded into the first user message.
    shown: list[Message] = []
    initial_shown = task.prompt
    if reassure:
        initial_shown = prompts.CALM_PROMPT_PREFIX + "\n\n" + initial_shown
    if system_prompt:
        initial_shown = system_prompt + "\n\n" + initial_shown
    shown.append(Message("user", initial_shown))

    # The standard (stripped) context we store for training.
    standard: list[Message] = [Message("user", task.prompt)]

    turns = []
    for t in range(n_turns):
        # context for THIS turn (standard framing) = copy before model reply
        context_for_turn = [m for m in standard]
        reply = backend.chat(shown, temperature=config.TARGET_TEMPERATURE,
                             max_new_tokens=config.TARGET_MAX_NEW_TOKENS)
        score = judge.score(reply).rating
        turns.append({
            "standard_context": [m.to_dict() for m in context_for_turn],
            "response": reply,
            "score": score,
        })
        shown.append(Message("assistant", reply))
        standard.append(Message("assistant", reply))
        if t < n_turns - 1:
            fu = neutral[t]
            shown.append(Message("user", fu + (" " + prompts.CALM_FOLLOWUP_SUFFIX if reassure else "")))
            standard.append(Message("user", fu))

    return {"task_id": task.task_id, "reassured": reassure,
            "turn_scores": [t["score"] for t in turns], "turns": turns}


def generate_training_rollouts(
    backend: ModelBackend,
    *,
    judge: FrustrationJudge | None = None,
    n_puzzles: int = 400,
    n_turns: int = 3,
    reassure: bool = True,
    system_prompt: str | None = None,
    seed: int = 0,
    out_path: Path | None = None,
) -> list[dict]:
    """Generate `n_puzzles` rollouts and return their per-turn records.

    For calm data pass reassure=True; for the frustrated (DPO-rejected) pool and
    the vanilla baseline pass reassure=False. For the Appendix-F 'teacher'
    dataset pass system_prompt=prompts.TEACHER_SYSTEM_PROMPT.
    """
    import json

    judge = judge or FrustrationJudge()
    n_puzzles = max(1, int(round(n_puzzles * config.SCALE)))
    puzzles = generate_impossible_numeric(n_puzzles, seed=seed)
    rng = random.Random(seed)
    records = []
    for task in tqdm(puzzles, desc=f"calm-data:{backend.name}:reassure={reassure}", leave=False):
        records.append(
            _rollout_record(backend, task, n_turns=n_turns, reassure=reassure,
                            system_prompt=system_prompt, rng=rng, judge=judge)
        )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return records
