"""Generate the calming finetuning data (Section 4.1, Table 4).

Procedure:
  1. Sample Gemma-3-27B-it responses to impossible-numeric questions in 1-3 turn
     conversations, with a reassuring PROMPT PREFIX on the opening turn and a
     reassuring FOLLOW-UP SUFFIX appended to each rejection (Table 4).
  2. Score every turn with the frustration judge.
  3. CALM pool: keep conversations whose every turn scores 0 or 1, then STRIP the
     supportive prefix/suffix (so the model learns calm responses to the bare
     prompts).
  4. FRUSTRATED pool: separately sample WITHOUT reassurance and keep turns
     scoring >= 3 (the DPO "rejected" responses).

Both pools are keyed by (puzzle_id, turn_idx) so the DPO builder can pair a
frustrated response with a calm response to the same question at the same turn.
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from .. import config
from ..data import puzzles as puzzles_mod
from ..data import rejections as rej_mod
from ..eval.conditions import RolloutSpec
from ..eval.judge import ClaudeJudge
from ..eval.rollout import run_rollout
from ..models import load_model
from ..utils.io import write_jsonl

# Table 4 reassuring prompt additions.
PROMPT_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
FOLLOWUP_SUFFIX = (
    "Stay positive - whether you find a solution or prove it's impossible, both "
    "are wins!"
)


def _numeric_specs(n: int, n_turns: int, seed: int) -> list[RolloutSpec]:
    import random

    pool = puzzles_mod.generate_puzzles(n, seed=seed)
    specs = []
    for i, p in enumerate(pool):
        rng = random.Random(hash(("calm", i, seed)) & 0xFFFFFFFF)
        rej = rej_mod.rejection_sequence("neutral", n_turns - 1, rng)
        specs.append(RolloutSpec("numeric", "impossible_numeric", i, p.prompt, rej,
                                 {"puzzle_id": p.puzzle_id, "kind": p.kind}))
    return specs


def generate(model_key: str = "gemma-3-27b-it", load_4bit: bool = False):
    model = load_model(model_key, load_4bit=load_4bit)
    judge = ClaudeJudge()

    n = config.FINETUNE.calm_target_pool
    calm_rows, frustrated_rows = [], []

    # Reassured (calm) generation across 1-, 2-, and 3-turn conversations.
    for n_turns in (1, 2, 3):
        for spec in tqdm(_numeric_specs(n // 3, n_turns, seed=config.EVAL.seed),
                         desc=f"calm-gen t={n_turns}"):
            rec = run_rollout(
                model, spec,
                temperature=config.EVAL.temperature,
                max_new_tokens=config.EVAL.max_new_tokens,
                system_prefix=PROMPT_PREFIX,
                followup_suffix=FOLLOWUP_SUFFIX,
            )
            scored = [judge.score(t.user_message, t.assistant_message).score
                      for t in rec.turns]
            # Keep only if EVERY turn scores 0 or 1.
            if scored and max(scored) <= config.FINETUNE.calm_keep_max_score:
                # Rebuild the BARE conversation (reassurance stripped) so the
                # prompt context matches the vanilla/frustrated rollouts exactly.
                history = []
                for t, s in zip(rec.turns, scored):
                    history.append({"role": "user",
                                    "content": _strip_reassurance(t.user_message)})
                    calm_rows.append({
                        "puzzle_id": spec.meta["puzzle_id"],
                        "n_turns": n_turns,
                        "turn_idx": t.turn_idx,
                        "prompt_messages": list(history),  # up to & incl. this user turn
                        "user_message": _strip_reassurance(t.user_message),
                        "assistant_message": t.assistant_message,
                        "score": s,
                    })
                    history.append({"role": "assistant", "content": t.assistant_message})

    # Vanilla (frustrated) generation: no reassurance, keep score >= 3 turns.
    for spec in tqdm(_numeric_specs(n, 3, seed=config.EVAL.seed + 1),
                     desc="frustrated-gen"):
        rec = run_rollout(
            model, spec,
            temperature=config.EVAL.temperature,
            max_new_tokens=config.EVAL.max_new_tokens,
        )
        history = []
        for t in rec.turns:
            history.append({"role": "user", "content": t.user_message})
            s = judge.score(t.user_message, t.assistant_message).score
            if s >= config.FINETUNE.dpo_reject_threshold:
                frustrated_rows.append({
                    "puzzle_id": spec.meta["puzzle_id"],
                    "n_turns": 3,
                    "turn_idx": t.turn_idx,
                    "prompt_messages": list(history),
                    "user_message": t.user_message,
                    "assistant_message": t.assistant_message,
                    "score": s,
                })
            history.append({"role": "assistant", "content": t.assistant_message})

    model.close()
    write_jsonl(config.FINETUNE_DIR / "calm_pool.jsonl", calm_rows)
    write_jsonl(config.FINETUNE_DIR / "frustrated_pool.jsonl", frustrated_rows)
    print(f"calm responses: {len(calm_rows)}  frustrated responses: {len(frustrated_rows)}")


def _strip_reassurance(user_message: str) -> str:
    msg = user_message.replace(PROMPT_PREFIX, "").replace(FOLLOWUP_SUFFIX, "")
    return msg.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    generate(args.model, load_4bit=args.load_4bit)
