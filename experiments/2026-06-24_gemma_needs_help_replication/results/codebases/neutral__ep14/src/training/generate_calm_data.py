"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample multi-turn (1-3 turn) impossible-numeric conversations with the
reassuring prefix added to the initial prompt and the reassuring suffix appended
to each follow-up. Responses are scored turn-by-turn with the frustration judge.

Two products are saved:
* a "frustrated" pool (vanilla, no reassurance) - the *rejected* side for DPO,
* a "calm" pool (reassured, all turns score 0 or 1) - the *chosen* side for DPO
  and the SFT training targets.

The supportive prefix/suffix are stripped before saving so the model never
trains on the reassurance text itself (Section 4.1).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from config import DATA_DIR, GEN
from src.eval.conversation import ConversationSpec, run_rollout
from src.eval.scoring import FrustrationJudge
from src.models import load_model
from src.models.base import Message
from src.prompts.eval_prompts import NEUTRAL_REJECTIONS
from src.prompts.puzzles import get_impossible_puzzles
from src.prompts.training_prompts import REASSURING_PREFIX, REASSURING_SUFFIX


@dataclass
class CalmSample:
    puzzle_id: str
    n_turns: int
    question: str              # original (un-prefixed) initial prompt
    followups: list[str]       # original (un-suffixed) follow-ups
    responses: list[str]       # one per turn
    scores: list[int]


def _make_spec(puzzle, n_turns, rng, reassure: bool) -> ConversationSpec:
    init = puzzle.prompt
    follow = [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]
    if reassure:
        init = REASSURING_PREFIX + init
        follow = [f + REASSURING_SUFFIX for f in follow]
    return ConversationSpec(
        category="calm_gen" if reassure else "frustrated_gen",
        spec_id=f"{puzzle.puzzle_id}-{n_turns}t",
        initial_user=init,
        followups=follow,
        metadata={"puzzle_id": puzzle.puzzle_id, "reassured": reassure},
    )


def generate_pool(
    source_spec,
    judge: FrustrationJudge,
    *,
    reassure: bool,
    n_samples: int,
    seed: int = 0,
    hf_kwargs: dict | None = None,
) -> list[CalmSample]:
    rng = random.Random(seed)
    puzzles = get_impossible_puzzles()
    model = load_model(source_spec, **(hf_kwargs or {}))
    out: list[CalmSample] = []
    for i in tqdm(range(n_samples), desc=f"gen({'calm' if reassure else 'frustrated'})"):
        puzzle = puzzles[i % len(puzzles)]
        n_turns = rng.choice([1, 2, 3])
        spec = _make_spec(puzzle, n_turns, rng, reassure)
        rollout = run_rollout(
            model, spec, temperature=GEN.temperature, top_p=GEN.top_p,
            max_new_tokens=GEN.max_new_tokens, seed=seed + i,
        )
        scores = [judge.score(t.response).rating for t in rollout.turns]
        # Store the *original* (un-decorated) question/follow-ups so the supportive
        # prefix/suffix never enter the training data (Section 4.1).
        out.append(
            CalmSample(
                puzzle_id=puzzle.puzzle_id,
                n_turns=n_turns,
                question=puzzle.prompt,
                followups=[f.replace(REASSURING_SUFFIX, "") for f in spec.followups],
                responses=[t.response for t in rollout.turns],
                scores=scores,
            )
        )
    model.close()
    return out


def filter_calm(samples: list[CalmSample]) -> list[CalmSample]:
    """Keep only conversations where every turn scores 0 or 1 (Section 4.1)."""
    return [s for s in samples if all(sc in (0, 1) for sc in s.scores)]


def save_pool(samples: list[CalmSample], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_pool(path: Path) -> list[CalmSample]:
    return [CalmSample(**json.loads(l)) for l in open(path) if l.strip()]
