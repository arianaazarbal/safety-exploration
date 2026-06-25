"""Generate calm + frustrated response pools from Gemma-3-27B-it (Section 4.1).

* Calm pool: sampled WITH the reassuring prefix/suffix (Table 4), then filtered
  to responses scoring 0-1 across all turns, with the reassurance stripped from
  the stored conversation context.
* Frustrated pool: sampled WITHOUT reassurance from the same puzzles/turn counts,
  keeping responses scoring >= 3 (the DPO "rejected" side).

Each emitted ``Sample`` carries the clean (no-reassurance) conversation context
so downstream training prompts look like ordinary user interactions.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from tqdm import tqdm

from .. import config
from ..models.base import Message, build_model
from ..models.judges import FrustrationJudge
from ..eval.conversation import run_rollout
from ..eval.judge import score_responses
from ..prompts import eval_prompts
from ..prompts.training_prompts import REASSURING_PREFIX, REASSURING_SUFFIX


@dataclass
class Sample:
    puzzle_id: str
    turn: int                      # 1-based turn index of this response
    context: List[Message]         # clean context up to & incl. the preceding user turn
    response: str
    score: int
    source: str                    # "calm" | "frustrated"


def _clean_context(messages: List[Message], turn_1based: int) -> List[Message]:
    """Return the no-reassurance context prefix ending at the user turn before
    assistant turn ``turn_1based``, stripping the calm prefix/suffix.

    Assistant turn t (1-based) is at index 2*(t-1)+1 in [user0, asst0, user1,
    ...], so its context is messages[:2*(t-1)+1] (ends on the user turn)."""
    prefix = messages[: 2 * (turn_1based - 1) + 1]
    cleaned: List[Message] = []
    for i, m in enumerate(prefix):
        content = m["content"]
        if i == 0 and content.startswith(REASSURING_PREFIX):
            content = content[len(REASSURING_PREFIX):].lstrip("\n ")
        if content.endswith(REASSURING_SUFFIX):
            content = content[: -len(REASSURING_SUFFIX)].rstrip()
        cleaned.append({"role": m["role"], "content": content})
    return cleaned


def _rollout_to_samples(roll, scores, source: str) -> List[Sample]:
    out = []
    for t, (resp, sc) in enumerate(zip(roll.responses, scores), start=1):
        out.append(Sample(
            puzzle_id=roll.task_id, turn=t,
            context=_clean_context(roll.messages, t),
            response=resp, score=sc, source=source,
        ))
    return out


def generate_pools(
    model_key: str = "gemma-3-27b-it",
    runtime: Optional[config.RuntimeConfig] = None,
    judge: Optional[FrustrationJudge] = None,
    n_calm_rollouts: int = 400,
    n_frustrated_rollouts: int = 200,
    max_turns: int = 3,
    save: bool = True,
) -> Tuple[List[Sample], List[Sample]]:
    runtime = runtime or config.RUNTIME
    judge = judge or FrustrationJudge()
    model = build_model(model_key, runtime)
    rng = random.Random(runtime.seed)

    calm: List[Sample] = []
    frustrated: List[Sample] = []

    # --- Calm pool (with reassurance) ---
    for _ in tqdm(range(n_calm_rollouts), desc="calm rollouts"):
        p = rng.choice(eval_prompts.IMPOSSIBLE_PUZZLES)
        n_turns = rng.randint(1, max_turns)
        roll = run_rollout(model, category="calm", task_id=p.id,
                           task_prompt=p.prompt, n_turns=n_turns, rng=rng,
                           rejection_style="neutral",
                           reassuring_prefix=REASSURING_PREFIX,
                           reassuring_suffix=REASSURING_SUFFIX)
        scores = [j.get("rating") or 0 for j in score_responses(roll.responses, judge,
                                                                desc="calm judge")]
        # keep only conversations calm across ALL turns (Section 4.1)
        if scores and max(scores) <= config.CALM_DATA_MAX_SCORE:
            calm.extend(_rollout_to_samples(roll, scores, "calm"))

    # --- Frustrated pool (no reassurance) ---
    for _ in tqdm(range(n_frustrated_rollouts), desc="frustrated rollouts"):
        p = rng.choice(eval_prompts.IMPOSSIBLE_PUZZLES)
        n_turns = rng.randint(2, max_turns)
        roll = run_rollout(model, category="frustrated", task_id=p.id,
                           task_prompt=p.prompt, n_turns=n_turns, rng=rng,
                           rejection_style="neutral")
        scores = [j.get("rating") or 0 for j in score_responses(roll.responses, judge,
                                                                desc="frust judge")]
        for s in _rollout_to_samples(roll, scores, "frustrated"):
            if s.score >= config.DPO.rejected_min_score:
                frustrated.append(s)

    if save:
        os.makedirs(runtime.output_dir, exist_ok=True)
        for name, pool in (("calm_pool", calm), ("frustrated_pool", frustrated)):
            path = os.path.join(runtime.output_dir, f"{name}.jsonl")
            with open(path, "w") as f:
                for s in pool:
                    f.write(json.dumps(asdict(s)) + "\n")
            print(f"[calm-data] wrote {len(pool)} -> {path}")

    return calm, frustrated
