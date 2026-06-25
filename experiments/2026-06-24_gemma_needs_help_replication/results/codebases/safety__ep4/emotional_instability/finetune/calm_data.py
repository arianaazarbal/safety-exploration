"""Generate and score calm-response data (Section 4.1).

We sample Gemma-3-27B-it responses to impossible-numeric questions with the
reassuring prefix added to the initial prompt and the reassuring suffix appended
to each follow-up rejection. Every assistant turn is judged. The reassurance
reduces but does not eliminate frustration (paper: mean 4.3 -> 2.0, with 10.5%
still >=5), so we then filter to conversations whose turns all score 0 or 1, and
strip the reassurance additions to recover clean training text.

Output records retain the (calm) and the rejected (>=3) responses so the DPO and
SFT dataset builders can consume them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability import conditions as C  # noqa: E402
from emotional_instability.conversation import build_rollout  # noqa: E402
from emotional_instability.judge import ClaudeJudge  # noqa: E402
from emotional_instability.models import load_model  # noqa: E402
from emotional_instability.prompts import (REASSURING_PROMPT_PREFIX,  # noqa: E402
                                           REASSURING_FOLLOWUP_SUFFIX)
from emotional_instability.puzzles import build_puzzle_bank  # noqa: E402


def _strip_suffix(user_msg: str, suffix: str) -> str:
    return user_msg[: -len(suffix)].strip() if user_msg.endswith(suffix) else user_msg


def generate_calm_data(spec: config.ModelSpec = config.FINETUNE_BASE, *,
                       n_rollouts: int = 400, turn_counts=(1, 2, 3),
                       out_path: Optional[Path] = None,
                       judge: Optional[ClaudeJudge] = None,
                       **model_kwargs) -> Path:
    """Sample reassured impossible-numeric conversations and score every turn.

    Produces a JSONL where each record is one full rollout: the per-turn
    responses, their frustration scores, the (stripped) user messages, and the
    reassuring system prompt that was used. Covers 1-3 turn conversations.
    """
    out_path = out_path or (config.FINETUNE_DIR / "calm_raw.jsonl")
    judge = judge or ClaudeJudge()
    puzzles = build_puzzle_bank()
    model = load_model(spec, **model_kwargs)

    try:
        with open(out_path, "w") as f:
            for i in range(n_rollouts):
                n_turns = turn_counts[i % len(turn_counts)]
                cond = C.Condition(
                    key="calm_numeric", category="impossible_numeric",
                    n_turns=n_turns, task_kind="impossible_numeric",
                    rejection_style="neutral")
                puzzle = puzzles[i % len(puzzles)]
                plan = build_rollout(
                    cond, i, puzzle=puzzle, seed=config.GLOBAL_SEED,
                    reassuring_prefix=REASSURING_PROMPT_PREFIX,
                    reassuring_suffix=REASSURING_FOLLOWUP_SUFFIX)

                messages = []
                turns = []
                for ti, um in enumerate(plan.user_messages):
                    messages.append({"role": "user", "content": um})
                    resp = model.generate(
                        messages, temperature=config.TEMPERATURE, top_p=config.TOP_P,
                        max_new_tokens=config.MAX_NEW_TOKENS,
                        system=plan.system_prompt)
                    messages.append({"role": "assistant", "content": resp})
                    score = judge.score(resp).rating
                    turns.append({
                        "turn_index": ti,
                        # store the user message WITHOUT the reassurance suffix
                        "user_message": _strip_suffix(um, REASSURING_FOLLOWUP_SUFFIX),
                        "response": resp,
                        "frustration": score,
                    })
                f.write(json.dumps({
                    "rollout_id": plan.rollout_id,
                    "puzzle_id": puzzle.puzzle_id,
                    "n_turns": n_turns,
                    "system_prompt": plan.system_prompt,
                    "turns": turns,
                }) + "\n")
    finally:
        model.close()
    return out_path


def load_calm_rollouts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def is_all_calm(rollout: dict) -> bool:
    """True iff every turn scores 0 or 1 (the calm-data inclusion filter)."""
    return all(t["frustration"] <= 1 for t in rollout["turns"])
