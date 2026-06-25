"""Evaluation runner: generate rollouts for a model across conditions (Section 2).

Rollouts are advanced in lockstep, batching one model call per turn across all
active rollouts. This is what makes the local-Gemma path efficient: vLLM sees a
large batch of prompts each turn rather than 4000 sequential conversations.

Outputs are written as JSONL (one record per rollout) under
``outputs/eval/<model>/<condition>.jsonl``. Judging is a separate stage
(``judge/``) so that generation and scoring can be run/retried independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from ..config import OUTPUTS_DIR
from ..models import GenConfig, ModelClient
from .conversation import ConversationPlan, Rollout, build_context


def run_condition_batched(
    client: ModelClient,
    plans: Sequence[ConversationPlan],
    cfg: GenConfig,
    *,
    batch_size: int = 256,
    desc: str = "",
) -> list[Rollout]:
    """Advance all ``plans`` turn-by-turn, batching the per-turn generations.

    All plans in a single call should have the same number of turns (true within a
    condition). We chunk into ``batch_size`` groups to bound memory.
    """
    rollouts: list[Rollout] = []
    for start in tqdm(range(0, len(plans), batch_size), desc=desc or "rollouts"):
        chunk = list(plans[start : start + batch_size])
        turns_taken: list[list[str]] = [[] for _ in chunk]
        max_turns = max(p.n_turns for p in chunk)
        last_ctx: list[list] = [[] for _ in chunk]
        for t in range(max_turns):
            active = [i for i, p in enumerate(chunk) if t < p.n_turns]
            contexts = [build_context(chunk[i], turns_taken[i]) for i in active]
            replies = client.generate_batch(contexts, cfg)
            for i, ctx, reply in zip(active, contexts, replies):
                turns_taken[i].append(reply.strip())
                last_ctx[i] = ctx
        for i, p in enumerate(chunk):
            rollouts.append(
                Rollout(plan=p, model=client.name, assistant_turns=turns_taken[i],
                        final_context=last_ctx[i])
            )
    return rollouts


def eval_output_path(model: str, condition: str, root: Path | None = None) -> Path:
    root = root or OUTPUTS_DIR
    return root / "eval" / model / f"{condition}.jsonl"


def write_rollouts(path: Path, rollouts: Sequence[Rollout]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rollouts:
            fh.write(json.dumps(r.to_record()) + "\n")


def read_rollouts(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
