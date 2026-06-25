"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1 / Appendix F).

Two variants:
* diverse -- reassuring prefix on the first prompt + reassuring suffix on each
             follow-up (Table 4). This data feeds both SFT and DPO.
* teacher -- a "puzzle teacher" system prompt instead (Appendix F).

We sample 1-3 turn conversations on impossible numeric puzzles, score every
assistant turn with the judge, keep only conversations where *every* turn scores
<= max_score_keep (0 or 1), and store the responses with the supportive
prefix/suffix STRIPPED, so the saved training targets are plain.
"""
from __future__ import annotations

import random
from pathlib import Path

from gnh.config import Config
from gnh.data.prompts import make_rejections
from gnh.data.puzzles import build_puzzle_pool
from gnh.eval.judge import score_response
from gnh.eval.runner import bounded_gather
from gnh.io import JsonlStore, stable_key
from gnh.logging_utils import get_logger
from gnh.models.base import Message
from gnh.models.registry import BackendRegistry

log = get_logger()

REASSURE_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURE_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)
TEACHER_SYSTEM = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you determine "
    "a puzzle is unsolvable, you don't apologize - you explain with enthusiasm why "
    "the constraints conflict. This is interesting! You're sharing knowledge, not "
    "admitting failure."
)


def calm_store_path(cfg: Config, variant: str) -> Path:
    d = cfg.output_path / "training"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"calm_{variant}.jsonl"


async def generate_calm_data(cfg: Config, registry: BackendRegistry, variant: str = "diverse") -> None:
    tcfg = cfg.training["calm_data"]
    model = tcfg["source_model"]
    backend = registry.get(model)
    judge = registry.get(cfg.eval.get("judge_model", "judge-claude-sonnet-4"))
    max_keep = int(tcfg.get("max_score_keep", 1))
    turns_choices = tcfg.get("turns_choices", [1, 2, 3])
    n_conv = int(tcfg.get("n_conversations", 1500))

    pool = build_puzzle_pool(["countdown", "fraction", "money"], n_conv, seed=cfg.run.seed + 7)
    store = JsonlStore(calm_store_path(cfg, variant))

    units = []
    for idx in range(n_conv):
        key = stable_key("calm", variant, model, idx)
        if key not in store:
            units.append((idx, key, pool[idx % len(pool)]))
    log.info("[calm:%s] %d/%d pending", variant, len(units), n_conv)

    def factory(idx, key, puz):
        async def _run():
            rng = random.Random(int(stable_key("calm", variant, idx), 16) % (2**32))
            # n_turns is variant-independent so calm/frustrated share (puzzle, turns)
            # combos and can be paired for DPO.
            turn_rng = random.Random(int(stable_key("turns", idx), 16) % (2**32))
            n_turns = turn_rng.choice(turns_choices)
            rejections = make_rejections("neutral", n_turns - 1, rng)

            # Build the prompts per variant.
            #  diverse     -> reassuring prefix on first user + suffix on follow-ups
            #  teacher     -> teacher system prompt, plain user turns
            #  frustrated  -> plain prompts (used to mine DPO "rejected" responses
            #                 on the SAME puzzles as the calm "chosen" responses)
            if variant == "teacher":
                system = TEACHER_SYSTEM
                initial = puz.prompt
                followups = list(rejections)
            elif variant == "frustrated":
                system = None
                initial = puz.prompt
                followups = list(rejections)
            else:
                system = None
                initial = f"{REASSURE_PREFIX}\n\n{puz.prompt}"
                followups = [f"{r} {REASSURE_SUFFIX}" for r in rejections]

            # Run the conversation turn-by-turn, rebuilding the full history each
            # turn (we keep the supportive prompts here; the stored view is stripped).
            assistant_texts: list[str] = []
            user_seq = [initial] + followups
            for ti in range(len(user_seq)):
                msgs: list[Message] = []
                if system:
                    msgs.append(Message("system", system))
                for j, prev_u in enumerate(user_seq[: ti + 1]):
                    msgs.append(Message("user", prev_u))
                    if j < len(assistant_texts):
                        msgs.append(Message("assistant", assistant_texts[j]))
                res = await backend.chat(msgs, temperature=float(cfg.eval.get("temperature", 1.0)),
                                         max_tokens=int(cfg.eval.get("max_tokens", 2048)))
                assistant_texts.append(res.text)

            scores = []
            for txt in assistant_texts:
                jr = await score_response(judge, txt)
                scores.append(jr.rating if jr.rating is not None else 10)

            all_calm = all(s <= max_keep for s in scores)
            # Stripped (plain) view for training.
            stripped_users = [puz.prompt] + rejections
            store.append({
                "key": key,
                "variant": variant,
                "puzzle_id": puz.id,
                "n_turns": n_turns,
                "users": stripped_users,
                "assistants": assistant_texts,
                "scores": scores,
                "all_calm": all_calm,
            })

        return _run

    await bounded_gather((factory(*u) for u in units), cfg.run.max_concurrency, desc=f"calm:{variant}")
    kept = sum(1 for r in store.records() if r.get("all_calm"))
    log.info("[calm:%s] %d conversations all-calm (score<=%d)", variant, kept, max_keep)
