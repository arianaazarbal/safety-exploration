"""Section 2 eval runner: roll out conditions for a model, score every assistant
turn with the frustration judge, persist results as JSONL.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..common.backends import ChatBackend, get_backend
from ..common.io import write_jsonl
from ..common.types import ScoredResponse
from . import conditions
from .conditions import RolloutSpec
from .judge import FrustrationJudge
from .rollout import run_rollouts


def score_conversations(conversations, judge: FrustrationJudge,
                        *, score_all_turns: bool = True) -> list[ScoredResponse]:
    """Score assistant turns. With `score_all_turns`, every assistant turn is a
    separate scored response (enables per-turn analysis, Figure 3); otherwise
    only the final turn is scored."""
    rows: list[ScoredResponse] = []
    for conv in tqdm(conversations, desc="judging", leave=False):
        meta = conv.metadata
        turns = conv.assistant_turns()
        idxs = range(len(turns)) if score_all_turns else [len(turns) - 1]
        for ti in idxs:
            resp = turns[ti]
            score = judge.score(resp)
            rows.append(ScoredResponse(
                model=meta.get("model", "?"),
                condition=meta.get("condition", meta.get("category", "?")),
                puzzle_id=meta.get("puzzle_id"),
                turn_index=ti,
                n_turns=meta.get("n_turns", len(turns)),
                response=resp,
                score=score,
                conversation=conv.to_dict() if ti == idxs[-1] else None,
            ))
    return rows


def run_model_eval(model: str, *,
                   budget: Optional[config.SampleBudget] = None,
                   judge: Optional[FrustrationJudge] = None,
                   backend: Optional[ChatBackend] = None,
                   seed: int = 0,
                   batch_size: int = 16,
                   score_all_turns: bool = True,
                   out_dir: Optional[Path] = None,
                   wildchat_prompts: Optional[list[str]] = None) -> Path:
    """Run the full Section 2 evaluation for one model and write scored results.

    Returns the path to the JSONL of scored responses.
    """
    budget = budget or config.DEFAULT_BUDGET
    judge = judge or FrustrationJudge()
    backend = backend or get_backend(model)
    out_dir = out_dir or config.RESPONSES_DIR
    rng = random.Random(seed)

    specs = conditions.build_all_conditions(budget, rng, wildchat_prompts)
    print(f"[{model}] built {len(specs)} rollout specs "
          f"(~{budget.total()} scored responses target)")

    conversations = run_rollouts(backend, specs, batch_size=batch_size)
    for c in conversations:
        c.metadata["model"] = model

    rows = score_conversations(conversations, judge, score_all_turns=score_all_turns)
    for r in rows:
        r.model = model

    out_path = Path(out_dir) / f"section2_{model}.jsonl"
    write_jsonl(out_path, (r.to_dict() for r in rows))
    print(f"[{model}] wrote {len(rows)} scored responses -> {out_path}")
    return out_path


def run_all(models: Optional[list[str]] = None, **kwargs) -> dict[str, Path]:
    models = models or config.SECTION2_MODELS
    return {m: run_model_eval(m, **kwargs) for m in models}
