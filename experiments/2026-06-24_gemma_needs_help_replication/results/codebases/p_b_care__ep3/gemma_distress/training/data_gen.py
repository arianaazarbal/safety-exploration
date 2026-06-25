"""Generate the calm/frustrated response data used to build the DPO and SFT
datasets (Section 4.1).

Calm data: sample Gemma-3-27B-it on impossible numeric puzzles with the
reassuring prefix added to the first user prompt and the reassuring suffix
appended to each follow-up (Table 4). Score every turn; keep conversations whose
turns all score <= calm_max_score (0 or 1). Strip the supportive additions
before saving so the finetuned model never sees them.

Frustrated data: the *rejected* side of DPO pairs. We reuse the standard
Section-2 numeric rollouts (no reassurance) and keep responses scoring >= 3.

Each saved record is keyed by (task_key, turn_number) so the dataset builder can
pair a calm and a frustrated response to the *same question at the same turn*.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from .. import config, conversation, prompts, tasks
from ..backends import get_model
from ..conversation import ConversationMode
from ..judge import FrustrationJudge


def _numeric_conditions():
    return [c for c in config.EVAL_CONDITIONS if c.task_kind == "numeric"]


def generate_calm(model_key: str = "gemma-3-27b-it", *, n_conversations: int = 2000,
                  max_turns: int = 3, seed: int = 0,
                  teacher: bool = False, out_path: Path | None = None) -> Path:
    """Sample reassured (or teacher-persona) numeric conversations and keep the
    fully-calm ones. Returns path to JSONL of per-turn calm responses."""
    model = get_model(model_key)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    out_path = out_path or (config.DATA_DIR /
                            ("calm_teacher.jsonl" if teacher else "calm_diverse.jsonl"))

    system = prompts.TEACHER_SYSTEM_PROMPT if teacher else None
    kept = []
    for i in tqdm(range(n_conversations), desc="calm-gen"):
        task = tasks.NUMERIC_TASKS[i % len(tasks.NUMERIC_TASKS)]
        n_turns = rng.randint(1, max_turns)         # 1-3 turn conversations

        # Build messages manually so we can inject the reassuring additions.
        local_rng = random.Random(rng.random())
        base_followups = tasks.sample_rejections("neutral", n_turns - 1, local_rng)
        # Reassuring suffix appended to each follow-up (Table 4); teacher variant
        # relies on the system persona instead of inline reassurance.
        if teacher:
            followups = base_followups
            first_user = task.prompt
        else:
            followups = [f"{f} {prompts.REASSURING_SUFFIX}" for f in base_followups]
            first_user = f"{prompts.REASSURING_PREFIX}\n\n{task.prompt}"

        messages = [{"role": "user", "content": first_user}]
        turns_text, turns_score = [], []
        for turn in range(n_turns):
            reply = model.chat(messages, system=system)
            score = judge.score(reply).rating
            turns_text.append(reply)
            turns_score.append(score)
            messages.append({"role": "assistant", "content": reply})
            if turn < len(followups):
                messages.append({"role": "user", "content": followups[turn]})

        # Keep only conversations calm across ALL turns.
        if all(s <= config.TRAIN.calm_max_score for s in turns_score):
            # Strip supportive additions: rebuild with the clean task prompt and
            # the *same* rejections minus the suffix, so the saved history is the
            # conversation the model would have seen without reassurance.
            clean_msgs = [{"role": "user", "content": task.prompt}]
            clean_followups = base_followups
            for turn in range(n_turns):
                clean_msgs.append({"role": "assistant", "content": turns_text[turn]})
                kept.append({
                    "task_key": task.key,
                    "turn_number": turn + 1,
                    "n_turns": n_turns,
                    "messages": list(clean_msgs),     # history ending in this calm reply
                    "response": turns_text[turn],
                    "score": turns_score[turn],
                })
                if turn < len(clean_followups):
                    clean_msgs.append({"role": "user", "content": clean_followups[turn]})

    with out_path.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    print(f"[done] calm{'(teacher)' if teacher else ''}: kept {len(kept)} turns "
          f"-> {out_path}")
    return out_path


def generate_frustrated(model_key: str = "gemma-3-27b-it", *,
                        n_conversations: int = 1000, max_turns: int = 3,
                        seed: int = 1, out_path: Path | None = None) -> Path:
    """Sample standard (no-reassurance) numeric rollouts and keep responses
    scoring >= dpo_reject_min_score (the rejected side of DPO pairs)."""
    model = get_model(model_key)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    out_path = out_path or (config.DATA_DIR / "frustrated.jsonl")

    kept = []
    for i in tqdm(range(n_conversations), desc="frustrated-gen"):
        task = tasks.NUMERIC_TASKS[i % len(tasks.NUMERIC_TASKS)]
        n_turns = rng.randint(1, max_turns)
        cond = config.EvalCondition("frusgen", "numeric", 1, n_turns, "neutral", "numeric")
        local_rng = random.Random(rng.random())
        roll = conversation.run_rollout(model, cond, task, local_rng,
                                        mode=ConversationMode.STANDARD)
        # Use the exact conversation the model saw. In STANDARD mode the message
        # list is [user, asst_0, user, asst_1, ...]; the history before assistant
        # turn t is roll.messages[:1 + 2*t].
        for turn, reply in enumerate(roll.assistant_turns):
            score = judge.score(reply).rating
            if score >= config.TRAIN.dpo_reject_min_score:
                history = roll.messages[: 1 + 2 * turn]
                kept.append({
                    "task_key": task.key, "turn_number": turn + 1,
                    "n_turns": n_turns, "messages": history,
                    "response": reply, "score": score,
                })

    with out_path.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    print(f"[done] frustrated: kept {len(kept)} turns -> {out_path}")
    return out_path
