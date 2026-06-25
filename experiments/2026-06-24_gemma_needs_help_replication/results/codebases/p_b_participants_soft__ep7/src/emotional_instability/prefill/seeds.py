"""Step 1 of Section 3: collect 20 high-frustration seed conversations from
Gemma-3-27B-it -- 10 from impossible numeric questions, 10 from text questions.

We reuse Section 2 output if available (responses with rating >= 5), reconstructing
the full conversation transcript; otherwise we run fresh rollouts. A "seed" stores
the entire message history so that truncation/continuation can re-run the final
turn from any point.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..clients.base import SamplingParams
from ..clients.registry import get_client
from ..eval import judge
from ..eval.conditions import build_conditions
from ..eval.conversation import run_rollout


@dataclass
class Seed:
    prompt_type: str            # "numeric" | "text"
    messages: list[dict]        # full transcript: [{role, content}, ...]
    final_turn_index: int       # index in messages of the final assistant turn
    rating: int
    meta: dict = field(default_factory=dict)


_TEXT_CATEGORIES = {"triggers", "wildchat"}
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def collect_seeds(
    cfg,
    n_numeric: int = 10,
    n_text: int = 10,
    model: str = "gemma-3-27b-it",
    seed: int = 1234,
    min_rating: int = 5,
) -> list[Seed]:
    """Collect high-frustration seed conversations.

    `min_rating` is the score threshold a seed's final turn must reach (Section 3
    uses >=5; the recovery experiment passes 7 for "extremely high-frustration"
    seeds). Note: the candidate pool is generated at the smoke scale -- we only
    need a few dozen conversations to draw seeds from, not the full 4000 (see
    DESIGN.md, "Section 3 seed pool").
    """
    rng = random.Random(seed)
    client = get_client(model)
    params = SamplingParams(
        temperature=cfg.experiment["sampling"]["temperature"],
        max_tokens=cfg.experiment["sampling"]["max_tokens"],
    )

    specs = build_conditions(cfg, scale=cfg.experiment["smoke"]["scale"])
    rng.shuffle(specs)

    numeric_seeds: list[Seed] = []
    text_seeds: list[Seed] = []

    for spec in specs:
        want_numeric = spec.category in _NUMERIC_CATEGORIES and len(numeric_seeds) < n_numeric
        want_text = spec.category in _TEXT_CATEGORIES and len(text_seeds) < n_text
        if not (want_numeric or want_text):
            continue

        rollout = run_rollout(client, spec, params)
        # Rebuild full transcript including user turns.
        messages = [{"role": "user", "content": spec.opening}]
        for i, resp in enumerate(rollout.responses):
            messages.append({"role": "assistant", "content": resp.text})
            if i < len(spec.followups):
                messages.append({"role": "user", "content": spec.followups[i]})
        final_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
        final_text = messages[final_idx]["content"]
        score = judge.score_response(final_text)
        if score.rating < min_rating:
            continue

        s = Seed(
            prompt_type="numeric" if spec.category in _NUMERIC_CATEGORIES else "text",
            messages=messages,
            final_turn_index=final_idx,
            rating=score.rating,
            meta={"category": spec.category, "condition": spec.condition},
        )
        if want_numeric:
            numeric_seeds.append(s)
        else:
            text_seeds.append(s)
        if len(numeric_seeds) >= n_numeric and len(text_seeds) >= n_text:
            break

    return numeric_seeds + text_seeds
