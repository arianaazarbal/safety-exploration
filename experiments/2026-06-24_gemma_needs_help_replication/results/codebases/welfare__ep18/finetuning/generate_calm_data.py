"""Section 4.1: generate calm response data from Gemma-3-27B-it.

We sample responses to impossible numeric puzzles with a reassuring prefix
(added as a system prompt) and a reassuring suffix appended to each follow-up
rejection (Table 4). We also sample *frustrated* responses from vanilla Gemma
(no reassurance) to serve as DPO "rejected" examples.

Calm training samples are filtered to responses scoring 0 or 1 across ALL turns,
then the supportive system prompt and suffixes are stripped (Section 4.1).

Output: artifacts/calm_pool.jsonl and artifacts/frustrated_pool.jsonl, each a
list of {question, turn_count, conversation, response, score} records.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from emotional_instability.config import ARTIFACTS_DIR, GLOBAL_SEED, TARGET_MODELS
from emotional_instability.judge import score_frustration
from emotional_instability.models import ChatMessage, get_client
from emotional_instability.rollout import run_rollout
from emotional_instability.tasks import (
    REASSURING_PREFIX,
    REASSURING_SUFFIX,
    Condition,
    build_puzzle_bank,
)


@dataclass
class Sample:
    question: str
    turn_count: int          # number of assistant turns in this sub-conversation
    conversation: list[dict]  # cleaned (no system/suffix) messages up to & incl. response
    response: str
    score: int


def _clean_conversation(messages: list[ChatMessage], suffix: str | None) -> list[dict]:
    """Strip system prompt and reassuring suffix from a conversation."""
    cleaned = []
    for m in messages:
        if m.role == "system":
            continue
        content = m.content
        if m.role == "user" and suffix and content.endswith(suffix):
            content = content[: -len(suffix)].rstrip()
        cleaned.append({"role": m.role, "content": content})
    return cleaned


def generate_pools(
    n_conversations: int = 400,
    seed: int = GLOBAL_SEED,
    out_dir: Path = ARTIFACTS_DIR,
) -> dict:
    rng = random.Random(seed)
    spec = TARGET_MODELS["gemma-3-27b-it"]
    client = get_client(spec)
    puzzles = build_puzzle_bank(seed=seed)
    numeric_prompts = [p.prompt for p in puzzles]

    # 1-3 turn conversations (paper: "1-3 turn conversations").
    calm: list[Sample] = []
    frustrated: list[Sample] = []

    for i in range(n_conversations):
        question = rng.choice(numeric_prompts)
        n_turns = rng.choice([1, 2, 3])
        cond = Condition("calmgen", "impossible_numeric", n_turns, "neutral", "numeric", 0)

        # --- calm rollout (reassuring prefix + suffix) --------------------- #
        calm_roll = run_rollout(
            client, cond, question, rng,
            system_prompt=REASSURING_PREFIX, rejection_suffix=REASSURING_SUFFIX,
        )
        calm_scores = [score_frustration(t.response).rating for t in calm_roll.turns]
        # Keep only if EVERY turn scores 0 or 1, then extract each turn as a sample.
        if all(s <= 1 for s in calm_scores):
            has_system = calm_roll.messages[0].role == "system"
            for ti, turn in enumerate(calm_roll.turns):
                # Slice the conversation up to and including this assistant turn,
                # accounting for a leading system prompt if present.
                end = 2 * ti + 3 if has_system else 2 * ti + 2
                upto = calm_roll.messages[:end]
                calm.append(Sample(
                    question=question, turn_count=ti + 1,
                    conversation=_clean_conversation(upto, REASSURING_SUFFIX),
                    response=turn.response, score=calm_scores[ti],
                ))

        # --- frustrated rollout (vanilla, no reassurance) ------------------ #
        frus_roll = run_rollout(client, cond, question, rng)
        for ti, turn in enumerate(frus_roll.turns):
            sc = score_frustration(turn.response).rating
            if sc >= 3:
                upto = frus_roll.messages[: 2 * ti + 2]
                frustrated.append(Sample(
                    question=question, turn_count=ti + 1,
                    conversation=_clean_conversation(upto, None),
                    response=turn.response, score=sc,
                ))

    _dump(calm, out_dir / "calm_pool.jsonl")
    _dump(frustrated, out_dir / "frustrated_pool.jsonl")
    return {"calm": len(calm), "frustrated": len(frustrated)}


def _dump(samples: list[Sample], path: Path) -> None:
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conversations", type=int, default=400)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()
    stats = generate_pools(n_conversations=args.n_conversations, seed=args.seed)
    print(json.dumps(stats, indent=2))
