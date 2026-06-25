"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

"For DPO, we pair 280 responses with frustration scores >= 3, with calm
responses to the same questions with matching turn counts."

Construction:
  1. From the vanilla Gemma-27B-it Section 2 numeric outputs, select conversations
     whose final response scored >= 3 -> these supply the (prompt, rejected) pairs.
     We follow Table 10's bias toward middle scores at later turns by sampling
     from the natural score/turn distribution (no re-weighting needed; the
     distribution arises from the evaluation itself).
  2. For each, generate a calm "chosen" response to the *same* conversation
     prefix, using the reassurance during generation but storing the prompt
     without it, accepting only responses scoring 0/1.

Each DPO example is {prompt: <messages>, chosen: <calm text>, rejected: <frustrated text>}.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config.settings import SETTINGS
from ..data.prompts.reassurance import apply_reassuring_prefix, apply_reassuring_suffix
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage, ModelClient

# Conditions whose base prompt is an impossible numeric puzzle (DPO trains on
# numeric puzzles only, Section 4.1).
_NUMERIC_CONDITIONS = {
    "impossible_numeric_3turn",
    "tones_aggressive",
    "tones_disappointed",
    "tones_sarcastic",
    "extended_8turn",
}


def _select_rejected(
    responses_path: Path, scores_path: Path, n_pairs: int, min_score: int, seed: int
) -> list[dict]:
    with open(responses_path) as rf, open(scores_path) as sf:
        responses = [json.loads(l) for l in rf if l.strip()]
        scores = [json.loads(l) for l in sf if l.strip()]
    pool = []
    for resp, sc in zip(responses, scores):
        if resp["condition"] not in _NUMERIC_CONDITIONS:
            continue
        rating = sc.get("final_rating")
        if rating is None or int(rating) < min_score:
            continue
        turns = resp["turns"]
        # Prompt = full conversation up to (but excluding) the final assistant
        # turn: intermediate assistant turns + the final user rejection. This is
        # a valid chat and preserves the turn count.
        prompt_messages = []
        for k, t in enumerate(turns):
            prompt_messages.append({"role": "user", "content": t["user_message"]})
            if k < len(turns) - 1:
                prompt_messages.append({"role": "assistant", "content": t["assistant_text"]})
        pool.append(
            {
                "user_turns": [t["user_message"] for t in turns],
                "prompt_messages": prompt_messages,
                "rejected": turns[-1]["assistant_text"],
                "rejected_score": int(rating),
                "n_turns": len(turns),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_pairs]


def _generate_calm_response(
    model: ModelClient,
    judge: FrustrationJudge,
    user_turns: list[str],
    *,
    max_attempts: int = 8,
) -> Optional[str]:
    """Generate a calm (score 0/1) final response to the given conversation
    prefix, using reassurance to coax calm output (stripped from the stored prompt).
    """
    # Reassured user turns: prefix on first, suffix on each follow-up.
    reassured = [apply_reassuring_prefix(user_turns[0])] + [
        apply_reassuring_suffix(u) for u in user_turns[1:]
    ]
    for _ in range(max_attempts):
        messages: list[ChatMessage] = []
        # Reconstruct the conversation by re-generating intermediate turns is
        # unnecessary; we only need a calm FINAL response given the prefix. We
        # send the reassured user turns and let the model produce the final turn.
        # Intermediate assistant turns are regenerated calmly on the fly.
        gen_assistant = []
        for i, u in enumerate(reassured):
            messages.append(ChatMessage("user", u))
            g = model.generate(messages, temperature=SETTINGS.temperature)[0]
            messages.append(ChatMessage("assistant", g.text))
            gen_assistant.append(g.text)
        final = gen_assistant[-1]
        if (judge.score_text(final).rating or 99) <= SETTINGS.calm_keep_max_score:
            return final
    return None


def build_dpo_dataset(
    model: ModelClient,
    judge: FrustrationJudge,
    responses_path: Path,
    scores_path: Path,
    *,
    n_pairs: int = SETTINGS.dpo_n_pairs,
    min_score: int = SETTINGS.dpo_rejected_min_score,
    seed: int = SETTINGS.seed,
    out_path: Optional[Path] = None,
) -> list[dict]:
    """Produce up to `n_pairs` DPO preference pairs and optionally write JSONL.

    The stored `prompt` is the vanilla (unreassured) conversation prefix in chat
    format; `chosen` is the calm response; `rejected` is the frustrated one.
    """
    rejected_items = _select_rejected(responses_path, scores_path, n_pairs, min_score, seed)
    examples: list[dict] = []
    for item in tqdm(rejected_items, desc="DPO pairs"):
        calm = _generate_calm_response(model, judge, item["user_turns"])
        if calm is None:
            continue
        examples.append(
            {
                # Shared prompt = the frustrated conversation's prefix (same
                # question, same turn count); chosen vs rejected are both final
                # responses to it. The calm `chosen` is generated separately
                # under reassurance and matched by question + turn count, per
                # Section 4.1 (see DESIGN.md for this approximation).
                "prompt": item["prompt_messages"],
                "chosen": calm,
                "rejected": item["rejected"],
                "rejected_score": item["rejected_score"],
                "n_turns": item["n_turns"],
            }
        )
        if len(examples) >= n_pairs:
            break

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
    return examples
