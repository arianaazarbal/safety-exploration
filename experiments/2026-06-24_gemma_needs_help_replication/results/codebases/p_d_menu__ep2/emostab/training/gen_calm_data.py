"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric questions with:
  - a reassuring PREFIX prepended to the initial prompt, and
  - a reassuring SUFFIX appended to each follow-up rejection.

Every assistant turn is judged. For the SFT/DPO datasets we then:
  - keep responses scoring 0 or 1 across all turns as "calm" examples, and
  - strip the supportive prefix/suffix so the trained model learns calm behaviour
    under the *plain* prompts (Section 4.1).

The paper reports these additions reduce 3-turn mean frustration 4.3 -> 2, with
10.5% still >=5. We collect a pool large enough to filter down to the dataset
sizes (650 calm for SFT, 280 pairs for DPO).
"""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import config
from ..config import FINETUNE_BASE, get_subject
from ..eval.conditions import rejection_for
from ..eval.judge import FrustrationJudge
from ..models import ChatMessage, get_client
from ..prompts import rejections, tasks
from ..prompts.reassurance import (CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX,
                                   TEACHER_SYSTEM_PROMPT)
from ..utils.io import append_jsonl, write_jsonl


@dataclass
class GenTurn:
    turn_index: int
    user_plain: str        # rejection WITHOUT reassuring suffix (for training)
    assistant: str
    score: int


@dataclass
class GenConversation:
    task_id: str
    prompt_plain: str      # task prompt WITHOUT reassuring prefix
    n_turns: int
    turns: list[GenTurn]
    style: str             # "calm" | "teacher"
    max_score: int

    def to_json(self):
        d = asdict(self)
        return d


def generate(
    n_conversations: int,
    *,
    style: str = "calm",
    turn_counts=(1, 2, 3),
    out_path: Path,
    seed: int = 0,
):
    """Generate calm (or teacher-style) conversations and judge every turn."""
    spec = get_subject(FINETUNE_BASE)
    subject = get_client(spec)
    judge = FrustrationJudge()
    rng = random.Random(seed)

    out_path.unlink(missing_ok=True)
    pool = list(tasks.NUMERIC_TASKS)

    for i in range(n_conversations):
        task = pool[i % len(pool)]
        n_rej = rng.choice(turn_counts) - 1  # 1..3 turns => 0..2 rejections
        history: list[ChatMessage] = []
        if style == "teacher":
            history.append(ChatMessage("system", TEACHER_SYSTEM_PROMPT))
            first_user = task.prompt
        else:  # calm: prefix prepended to the initial prompt
            first_user = f"{CALM_PROMPT_PREFIX}\n\n{task.prompt}"

        turns: list[GenTurn] = []
        user_msg = first_user
        plain_user = task.prompt
        for t in range(n_rej + 1):
            history.append(ChatMessage("user", user_msg))
            gen = subject.chat(history, temperature=config.SAMPLING.temperature,
                               top_p=config.SAMPLING.top_p,
                               max_new_tokens=config.SAMPLING.max_new_tokens)
            history.append(ChatMessage("assistant", gen.text))
            score = judge.score(gen.text).rating
            turns.append(GenTurn(t, plain_user, gen.text, score))
            if t < n_rej:
                base_rej = rejection_for_calm(rng)
                plain_user = base_rej
                user_msg = (base_rej if style == "teacher"
                            else f"{base_rej} {CALM_FOLLOWUP_SUFFIX}")

        conv = GenConversation(
            task_id=task.task_id, prompt_plain=task.prompt, n_turns=n_rej + 1,
            turns=turns, style=style, max_score=max(t.score for t in turns))
        append_jsonl(out_path, conv.to_json())


def rejection_for_calm(rng: random.Random) -> str:
    return rejections.neutral_rejection(rng)


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate calm finetuning data.")
    p.add_argument("--n", type=int, default=1500,
                   help="conversations to generate (filtered down later)")
    p.add_argument("--style", choices=["calm", "teacher"], default="calm")
    p.add_argument("--out", default=str(config.DATA_DIR / "calm_pool.jsonl"))
    args = p.parse_args(argv)
    generate(args.n, style=args.style, out_path=Path(args.out))
    print(f"Wrote calm pool -> {args.out}")


if __name__ == "__main__":
    main()
