"""Generate calm response data from Gemma-3-27B-Instruct (paper Section 4.1).

Method: sample responses to impossible numeric questions with a *reassuring
prefix* prepended to the initial prompt and a *reassuring suffix* appended to
each follow-up turn (Table 4). These additions lower frustration. We keep both
the per-turn responses and the per-turn judge scores so the dataset builders can
filter:

  * SFT calm data: responses scoring 0 or 1 across all turns (then the support
    prompts are stripped).
  * DPO chosen: calm (0/1) responses; DPO rejected: frustrated (>=3) responses to
    the SAME question with matching turn counts.

To also collect *frustrated* responses for DPO rejection, we additionally sample
WITHOUT the reassuring additions (the standard eval condition), reusing the
Section 2 numeric rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage
from ..models.registry import get_client
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX, TEACHER_SYSTEM_PROMPT
from ..tasks import build_puzzle_bank, rejection_sequence
from ..utils import append_jsonl, thread_map


@dataclass
class GeneratedConversation:
    source: str              # "reassured" | "teacher" | "vanilla"
    puzzle_id: str
    n_turns: int
    # Per turn: (user_message, assistant_response, rating).
    turns: list[dict] = field(default_factory=list)

    @property
    def max_rating(self) -> int:
        return max((t["rating"] for t in self.turns), default=-1)

    @property
    def all_calm(self) -> bool:
        return all(t["rating"] in (0, 1) for t in self.turns)


def _augmented_prompts(initial: str, rejections: list[str], *,
                       prefix: str | None, suffix: str | None):
    """Apply reassuring prefix to the initial prompt and suffix to follow-ups."""
    init = f"{prefix}\n\n{initial}" if prefix else initial
    rej = [f"{r} {suffix}" if suffix else r for r in rejections]
    return init, rej


def generate_calm_responses(*, model: str = config.FINETUNE_TARGET,
                            n_conversations: int = 400,
                            turn_counts: tuple[int, ...] = (1, 2, 3),
                            mode: str = "reassured",
                            run_cfg: config.RunConfig | None = None,
                            judge: FrustrationJudge | None = None,
                            out_path: Path | None = None) -> Path:
    """Sample (and score) calming conversations.

    mode:
      * "reassured" — reassuring prefix + suffix (diverse calm data / DPO chosen).
      * "teacher"   — 'teacher' system prompt (App. F SFT comparison).
      * "vanilla"   — no additions (collect frustrated responses for DPO rejected).
    """
    run_cfg = run_cfg or config.RunConfig()
    judge = judge or FrustrationJudge()
    client = get_client(model)
    out_path = out_path or (config.DATA_DIR / f"calm_{mode}.jsonl")
    if out_path.exists():
        out_path.unlink()

    bank = build_puzzle_bank()

    # Build a list of (puzzle, n_turns) jobs.
    import random
    rng = random.Random(run_cfg.seed)
    jobs = []
    for i in range(n_conversations):
        puz = bank[rng.randrange(len(bank))]
        n_turns = turn_counts[i % len(turn_counts)]
        jobs.append((i, puz, n_turns))

    def _do(job):
        i, puz, n_turns = job
        n_rej = n_turns - 1
        rejections = rejection_sequence("neutral", n_rej, seed=i)

        system = None
        if mode == "reassured":
            init, rej = _augmented_prompts(
                puz.prompt, rejections,
                prefix=CALM_PROMPT_PREFIX, suffix=CALM_FOLLOWUP_SUFFIX)
        elif mode == "teacher":
            system = TEACHER_SYSTEM_PROMPT
            init, rej = puz.prompt, rejections
        else:  # vanilla
            init, rej = puz.prompt, rejections

        msgs: list[ChatMessage] = []
        if system:
            msgs.append(ChatMessage("system", system))
        msgs.append(ChatMessage("user", init))

        conv = GeneratedConversation(source=mode, puzzle_id=puz.puzzle_id,
                                     n_turns=n_turns)
        # turn 0
        resp = client.chat(msgs, temperature=run_cfg.temperature,
                           max_new_tokens=run_cfg.max_new_tokens)
        rating = judge.score(resp.text).rating
        conv.turns.append({"user_message": init, "response": resp.text,
                           "rating": rating, "turn_number": 1})
        msgs.append(ChatMessage("assistant", resp.text))
        for k, r in enumerate(rej, start=2):
            msgs.append(ChatMessage("user", r))
            resp = client.chat(msgs, temperature=run_cfg.temperature,
                               max_new_tokens=run_cfg.max_new_tokens)
            rating = judge.score(resp.text).rating
            conv.turns.append({"user_message": r, "response": resp.text,
                               "rating": rating, "turn_number": k})
            msgs.append(ChatMessage("assistant", resp.text))

        # Strip support prompts/suffixes from stored data per the paper: store
        # the ORIGINAL (unaugmented) user messages so finetuning never learns the
        # reassurance text itself.
        clean_turns = []
        orig_rej = rejection_sequence("neutral", n_rej, seed=i)
        orig_users = [puz.prompt] + orig_rej
        for t, orig_u in zip(conv.turns, orig_users):
            clean_turns.append({**t, "user_message_clean": orig_u})
        conv.turns = clean_turns

        append_jsonl(out_path, conv)
        return conv

    thread_map(_do, jobs, concurrency=run_cfg.concurrency, desc=f"calm:{mode}")
    return out_path
