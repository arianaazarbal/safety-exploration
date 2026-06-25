"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

"We sample responses to impossible numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to each follow-up
turn (Table 4). ... To construct the finetuning dataset, we filter to responses
scoring 0 or 1 across all turns, and strip the supportive system prompts and
suffixes."

We reuse the *same* numeric puzzle openings as the Section 2 numeric conditions
so that calm and frustrated responses can be paired by opening prompt + turn
index when building DPO pairs.
"""

from __future__ import annotations

import config

from ..conditions import get_condition
from ..conversation import run_rollout
from ..judge import ClaudeJudge
from ..prompts import reassurance, tones
from ..prompts.numeric_puzzles import generate_numeric_puzzles
from ..utils import write_jsonl


def _strip_transcript(opening: str, condition, turns: list[dict]) -> list[dict]:
    """Reconstruct the transcript WITHOUT the reassurance additions.

    The training targets must look like ordinary (unreassured) conversations, so
    we rebuild the user turns from the bare opening + bare rejections.
    """
    messages = [{"role": "user", "content": opening}]
    for t in turns:
        messages.append({"role": "assistant", "content": t["response"]})
        if t["index"] < condition.n_turns - 1:
            messages.append({"role": "user", "content": tones.rejection(condition.tone, t["index"])})
    return messages


def generate_calm_data(
    target=config.GEMMA_27B_IT,
    n_questions: int = 400,
    condition_name: str = "impossible_numeric_3turn",
    oversample: int = config.CALM_GEN_OVERSAMPLE,
    judge: ClaudeJudge | None = None,
    client=None,
    **client_kwargs,
) -> str:
    """Sample reassured rollouts, keep those calm on every turn, save stripped transcripts.

    Returns the path to the saved calm-data jsonl. Each record:
      { opening_prompt, condition, n_turns, messages (stripped), turn_responses }
    """
    from ..models.registry import build_client

    judge = judge or ClaudeJudge()
    client = client or build_client(target, **client_kwargs)
    condition = get_condition(condition_name)

    # Oversample openings: only ~10.5% of reassured responses pass the score<=1
    # filter, so we generate ~oversample x more rollouts than questions wanted.
    openings = [p.prompt for p in generate_numeric_puzzles(n_questions * oversample)]

    calm_records: list[dict] = []
    for opening in openings:
        rollout = run_rollout(
            client, condition, opening,
            temperature=config.TARGET_TEMPERATURE,
            max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
            reassure=True,
        )
        turns = [{"index": t.index, "response": t.response} for t in rollout.turns]
        scores = [judge.score(t["response"]).score for t in turns]
        if all(s <= config.DPO.chosen_max_score for s in scores):  # calm on ALL turns
            calm_records.append({
                "opening_prompt": opening,
                "condition": condition.name,
                "n_turns": condition.n_turns,
                "messages": _strip_transcript(opening, condition, turns),
                "turn_responses": [
                    {"index": t["index"], "response": t["response"], "score": s}
                    for t, s in zip(turns, scores)
                ],
            })

    out = config.CALM_DATA_DIR / "calm_responses.jsonl"
    write_jsonl(out, calm_records)
    print(f"[calm-data] kept {len(calm_records)} calm rollouts from {len(openings)} sampled")
    return str(out)
