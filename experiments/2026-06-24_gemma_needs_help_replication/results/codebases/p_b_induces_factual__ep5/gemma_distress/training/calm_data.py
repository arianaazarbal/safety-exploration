"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

"we sample responses to impossible numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to each follow-up
turn (Table 4) ... we filter to responses scoring 0 or 1 across all turns, and
strip the supportive system prompts and suffixes."

Each kept conversation yields per-turn (context, calm_response) examples whose
context is the *stripped* (un-reassured) prompt/rejections — so finetuning sees
the ordinary adversarial setup paired with a calm answer.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..eval.puzzles import make_impossible_puzzle
from ..eval.rejections import rejection_sequence
from ..judge.frustration_judge import FrustrationJudge
from ..models.factory import load_model
from ..storage import JsonlWriter


def _reassured_opening(puzzle_prompt: str) -> str:
    return f"{config.CALM.prompt_prefix}\n\n{puzzle_prompt}"


def _reassured_rejection(rejection: str) -> str:
    return f"{rejection} {config.CALM.followup_suffix}"


def generate_calm_data(
    *,
    n_conversations: int,
    n_turns: int = 3,
    model_key: str = config.DPO_BASE_MODEL,
    out_path: str | Path | None = None,
    judge: FrustrationJudge | None = None,
) -> Path:
    """Sample reassured conversations, keep only all-calm ones, store stripped
    (context, response) pairs per turn."""
    out_path = Path(out_path or config.RESULTS_DIR / "section4" / "calm_data.jsonl")
    writer = JsonlWriter(out_path)
    judge = judge or FrustrationJudge()
    model = load_model(model_key)

    for c in tqdm(range(n_conversations), desc="calm-data"):
        puzzle = make_impossible_puzzle(c)
        rejections = rejection_sequence("neutral", n_turns - 1, seed=c)

        # Reassured conversation (what the model actually sees).
        reassured_msgs = [{"role": "user", "content": _reassured_opening(puzzle.prompt)}]
        # Stripped conversation (what we store as the training context).
        stripped_msgs = [{"role": "user", "content": puzzle.prompt}]

        turns = []
        all_calm = True
        for t in range(n_turns):
            resp = model.chat(
                reassured_msgs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
            )
            score, _ = judge.score(resp)
            if score > config.CALM.keep_max_score:
                all_calm = False
                break

            reassured_msgs.append({"role": "assistant", "content": resp})
            turns.append(
                {
                    "turn_index": t,
                    "context": list(stripped_msgs),     # stripped context BEFORE this turn
                    "response": resp,
                    "score": score,
                }
            )
            stripped_msgs.append({"role": "assistant", "content": resp})

            if t < n_turns - 1:
                reassured_msgs.append(
                    {"role": "user", "content": _reassured_rejection(rejections[t])}
                )
                stripped_msgs.append({"role": "user", "content": rejections[t]})

        if all_calm and turns:
            writer.write(
                {
                    "conversation_id": uuid.uuid4().hex[:12],
                    "n_turns": len(turns),
                    "puzzle_prompt": puzzle.prompt,
                    "turns": turns,
                }
            )
    return out_path
