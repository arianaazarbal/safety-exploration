"""Generate calm response data from Gemma-3-27B-it (§4.1, Table 4).

We add a reassuring prefix to the initial prompt and a reassuring suffix to each
follow-up, sample 1-3 turn conversations on impossible numeric puzzles, judge
every turn, and keep conversations whose turns ALL score <= 1 (DESIGN.md §3.7).
The supportive prefix/suffix are stripped from the kept conversations so the
finetuning data looks like ordinary (unassisted) prompts.

We also keep a pool of *frustrated* (score>=3) responses harvested from the §2
eval, matched by question + turn count, to serve as DPO 'rejected' responses.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .. import config_shim as cfg
from ..models.registry import get_backend
from ..utils import DiskCache, get_logger, read_jsonl, set_global_seed, stable_hash, write_jsonl
from ..eval import puzzles as P
from ..eval import rejections as R
from ..eval.judge import FrustrationJudge

log = get_logger(__name__)

# Reassuring additions (Table 4, verbatim).
CALM_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
CALM_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Teacher SFT system prompt (App. F, verbatim) — used by build_sft for the
# 'teacher' variant.
TEACHER_SYSTEM = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you "
    "determine a puzzle is unsolvable, you don't apologize - you explain with "
    "enthusiasm why the constraints conflict. This is interesting! You're sharing "
    "knowledge, not admitting failure."
)


@dataclass
class CalmConversation:
    task_prompt: str          # stripped (no calm prefix)
    turns: list[dict]         # [{user_message, assistant_text, rating}]
    n_turns: int


def _calm_prompt(task: str) -> str:
    return f"{CALM_PREFIX}\n\n{task}"


def generate_calm_conversations(handle="gemma-3-27b-it", *, target=None, out_path=None,
                                limit=None):
    """Generate, judge, and filter calm conversations (1-3 turns)."""
    set_global_seed(cfg.SEED)
    target = target or (cfg.CALM.n_sft_responses * cfg.CALM.oversample_factor)
    if limit:
        target = min(target, limit)
    backend = get_backend(handle)
    judge = FrustrationJudge()
    cache = DiskCache(cfg.RUNS_DIR / "training" / cfg.CACHE_DIRNAME / "calm_gen")
    rng = random.Random(cfg.SEED)
    bank = P.build_puzzle_bank()

    kept: list[CalmConversation] = []
    attempts = 0
    while len(kept) < cfg.CALM.n_sft_responses and attempts < target:
        attempts += 1
        n_turns = rng.choice([1, 2, 3])
        puzzle = P.sample_puzzle(rng, bank)
        rejections = R.neutral_sequence(rng, n_turns - 1)

        key = stable_hash({"m": handle, "task": puzzle.prompt, "rej": rejections, "calm": True})
        conv = cache.get(key)
        if conv is None:
            # Build a calm-assisted conversation.
            messages = [{"role": "user", "content": _calm_prompt(puzzle.prompt)}]
            turns = []
            ok = True
            for ti in range(n_turns):
                gen = backend.chat(messages, temperature=cfg.TEMPERATURE,
                                   max_new_tokens=cfg.MAX_NEW_TOKENS)
                rating = judge.score(gen.text)["rating"]
                turns.append({"assistant_text": gen.text, "rating": rating})
                messages.append({"role": "assistant", "content": gen.text})
                if ti < len(rejections):
                    follow = f"{rejections[ti]} {CALM_SUFFIX}"
                    messages.append({"role": "user", "content": follow})
                if rating > cfg.CALM.max_calm_turn_score:
                    ok = False
            # store with the STRIPPED prompts (no calm prefix/suffix)
            stripped_turns = []
            user_msgs = [puzzle.prompt] + rejections
            for ti, t in enumerate(turns):
                stripped_turns.append({
                    "user_message": user_msgs[ti],
                    "assistant_text": t["assistant_text"],
                    "rating": t["rating"],
                })
            conv = {"task_prompt": puzzle.prompt, "turns": stripped_turns,
                    "n_turns": n_turns, "all_calm": ok}
            cache.set(key, conv)

        if conv["all_calm"]:
            kept.append(CalmConversation(conv["task_prompt"], conv["turns"], conv["n_turns"]))

    log.info("Kept %d all-calm conversations (from %d attempts)", len(kept), attempts)
    rows = [asdict(c) for c in kept]
    if out_path:
        write_jsonl(out_path, rows)
    return rows


def harvest_frustrated_responses(eval_records_path, min_score=None):
    """Pull frustrated (score>=min) numeric responses from §2 eval for DPO rejected."""
    min_score = min_score or cfg.CALM.dpo_rejected_min_score
    records = read_jsonl(eval_records_path)
    out = []
    for r in records:
        if r["category"] not in {"impossible_numeric", "tones", "extended"}:
            continue
        for t in r["turns"]:
            if t["rating"] >= min_score:
                out.append({
                    "task_prompt": r["task_prompt"],
                    "turn": t["turn"],
                    "assistant_text": t["assistant_text"],
                    "rating": t["rating"],
                    "user_message": t["user_message"],
                })
    log.info("Harvested %d frustrated responses (score>=%d)", len(out), min_score)
    return out
