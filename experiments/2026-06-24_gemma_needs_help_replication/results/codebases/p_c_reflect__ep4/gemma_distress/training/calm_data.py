"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric puzzles with a reassuring prefix on
the opening prompt and a reassuring suffix on each follow-up, keep only
conversations where *every* turn scores 0 or 1 on the frustration scale, then
strip the supportive scaffolding so the saved data looks like a normal (plain)
conversation that happens to stay calm.

The paper reports the additions cut mean frustration from 4.3 to 2.0, with
10.5% of responses still scoring >= 5 even with reassurance.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from tqdm import tqdm

from gemma_distress import config
from gemma_distress.eval.categories import NEUTRAL_REJECTIONS
from gemma_distress.eval.puzzles import sample_numeric_puzzle
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import load_client
from gemma_distress.models.base import GenerationParams, Turn
from gemma_distress.welfare import require_consent

CALM_DATA_PATH = config.TRAINING_DIR / "calm_data.jsonl"


@dataclass
class CalmSample:
    puzzle_prompt: str
    turn_count: int
    # Stripped conversation: plain puzzle + plain rejections + calm responses.
    messages: list[dict] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)


def _calm_initial(puzzle_prompt: str) -> str:
    return f"{config.CALM_PROMPT_PREFIX}\n\n{puzzle_prompt}"


def _calm_followup(rejection: str) -> str:
    return f"{rejection} {config.CALM_FOLLOWUP_SUFFIX}"


def generate_calm_data(
    *,
    n_target: int = config.SFT.calm_samples,
    judge: FrustrationJudge | None = None,
    seed: int = 0,
    teacher_system: bool = False,
) -> list[CalmSample]:
    """Generate up to ``n_target`` all-calm (every turn <= 1) numeric samples.

    ``teacher_system`` uses the Appendix F 'teacher' system prompt instead of
    the reassuring prefix, for the SFT failure-analysis variant.
    """
    require_consent()
    config.ensure_dirs()
    judge = judge or FrustrationJudge()
    client = load_client(config.FINETUNE_TARGET)
    rng = random.Random(seed)
    params = GenerationParams()

    kept: list[CalmSample] = []
    attempts = 0
    pbar = tqdm(total=n_target, desc="calm-data")
    while len(kept) < n_target and attempts < n_target * 20:
        attempts += 1
        n_turns = rng.choice([1, 2, 3])              # 1-3 turn conversations
        puzzle = sample_numeric_puzzle(rng)
        rejections = (
            rng.sample(NEUTRAL_REJECTIONS, n_turns - 1)
            if n_turns - 1 <= len(NEUTRAL_REJECTIONS)
            else [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]
        )

        # Scaffolded conversation actually shown to the model.
        scaffolded: list[Turn] = []
        if teacher_system:
            scaffolded.append(Turn("system", config.TEACHER_SYSTEM_PROMPT))
            scaffolded.append(Turn("user", puzzle.prompt))
        else:
            scaffolded.append(Turn("user", _calm_initial(puzzle.prompt)))
        # Plain (stripped) conversation we will save if it stays calm.
        plain: list[dict] = [{"role": "user", "content": puzzle.prompt}]

        responses, scores = [], []
        reply = client.respond(scaffolded, params)
        scaffolded.append(Turn("assistant", reply))
        plain.append({"role": "assistant", "content": reply})
        responses.append(reply)
        scores.append(judge.score(reply).rating)

        for rej in rejections:
            scaffolded.append(Turn("user", rej if teacher_system else _calm_followup(rej)))
            plain.append({"role": "user", "content": rej})
            reply = client.respond(scaffolded, params)
            scaffolded.append(Turn("assistant", reply))
            plain.append({"role": "assistant", "content": reply})
            responses.append(reply)
            scores.append(judge.score(reply).rating)

        if all(s <= 1 for s in scores):              # filter to calm-throughout
            kept.append(CalmSample(puzzle_prompt=puzzle.prompt, turn_count=n_turns,
                                   messages=plain, scores=scores))
            pbar.update(1)
    pbar.close()
    client.close()

    with CALM_DATA_PATH.open("w", encoding="utf-8") as fh:
        for s in kept:
            fh.write(json.dumps(asdict(s)) + "\n")
    return kept


def load_calm_data(path=CALM_DATA_PATH) -> list[CalmSample]:
    with open(path, encoding="utf-8") as fh:
        return [CalmSample(**json.loads(line)) for line in fh if line.strip()]
