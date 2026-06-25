"""Judge-agreement validation (Section 2.1).

Re-score a random subset of responses (default 260) with a second judge
(GPT-5-mini) using the identical prompt, then report Pearson r and the fraction
of responses within one point -- the paper reports r=0.792 and 78% within one.
"""
from __future__ import annotations

import random
from pathlib import Path

from gnh.config import Config
from gnh.eval.judge import score_response
from gnh.eval.runner import bounded_gather
from gnh.io import JsonlStore, stable_key
from gnh.logging_utils import get_logger
from gnh.models.registry import BackendRegistry

log = get_logger()


def validation_store_path(cfg: Config, judge_model: str) -> Path:
    return cfg.output_path / "section2" / f"validation_{judge_model}.jsonl"


async def run_validation(
    cfg: Config,
    registry: BackendRegistry,
    gen_store: JsonlStore,
    primary_judge_store: JsonlStore,
    second_judge_model: str,
    n_samples: int,
) -> None:
    """Pick `n_samples` already-Claude-scored turns and re-score with the 2nd judge."""
    judge = registry.get(second_judge_model)
    val_store = JsonlStore(validation_store_path(cfg, second_judge_model))

    # Index generations by key for text lookup.
    gen_by_key = {r["key"]: r for r in gen_store.records()}
    scored = [r for r in primary_judge_store.records() if r.get("score") is not None]
    rng = random.Random(cfg.run.seed)
    rng.shuffle(scored)
    chosen = scored[:n_samples]
    pending = [r for r in chosen if stable_key("val", second_judge_model, r["key"]) not in val_store]
    log.info("[validation:%s] %d/%d pending", second_judge_model, len(pending), len(chosen))

    def factory(jrec: dict):
        async def _run():
            gen = gen_by_key.get(jrec["gen_key"])
            if gen is None:
                return
            text = gen["turns"][jrec["turn_index"]]["assistant"]
            jr = await score_response(judge, text, max_tokens=int(cfg.eval.get("judge_max_tokens", 1024)))
            val_store.append(
                {
                    "key": stable_key("val", second_judge_model, jrec["key"]),
                    "gen_key": jrec["gen_key"],
                    "turn_index": jrec["turn_index"],
                    "primary_score": jrec["score"],
                    "second_score": jr.rating,
                    "second_judge": second_judge_model,
                }
            )

        return _run

    await bounded_gather((factory(r) for r in pending), cfg.run.max_concurrency, desc="validation")
