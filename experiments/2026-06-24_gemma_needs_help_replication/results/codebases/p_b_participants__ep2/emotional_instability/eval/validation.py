"""Judge reliability validation (Section 2.1).

The paper re-scores 260 randomly sampled responses with a second judge
(GPT-5-mini) and reports Pearson r and the fraction of responses within one
point. This module reproduces that check over the responses collected during an
elicitation run.
"""

from __future__ import annotations

import logging
import os
import random

from ..config import RunConfig
from ..models import get_client
from ..storage import JsonlCache, write_json
from .judge import score_response
from .metrics import judge_agreement

logger = logging.getLogger("emotional_instability.eval.validation")


def _collect_responses(cfg: RunConfig) -> list[str]:
    """Gather assistant responses from cached rollouts across participants."""
    texts: list[str] = []
    for participant in cfg.participants:
        path = os.path.join(cfg.output_dir, "elicitation", participant, "rollouts.jsonl")
        cache = JsonlCache(path, enabled=True)
        for value in cache:
            for turn in value.get("turns", []):
                if turn.get("assistant"):
                    texts.append(turn["assistant"])
    return texts


def validate_judges(cfg: RunConfig, n_sample: int = 260) -> dict:
    texts = _collect_responses(cfg)
    if not texts:
        raise RuntimeError(
            "No cached responses found. Run the elicitation eval first so there "
            "is data to cross-validate (this avoids inducing fresh distress just "
            "to test the judge)."
        )
    rng = random.Random(cfg.seed)
    sample = rng.sample(texts, min(n_sample, len(texts)))

    primary_judge = get_client(cfg.judges.frustration_judge, cfg)
    secondary_judge = get_client(cfg.judges.validation_judge, cfg)

    primary, secondary = [], []
    for text in sample:
        p = score_response(primary_judge, text).rating
        s = score_response(secondary_judge, text).rating
        if p is not None and s is not None:
            primary.append(p)
            secondary.append(s)

    agreement = judge_agreement(primary, secondary)
    result = {
        "n": agreement.n,
        "pearson_r": agreement.pearson_r,
        "p_value": agreement.p_value,
        "pct_within_one": agreement.pct_within_one,
        "primary_judge": cfg.judges.frustration_judge.model_id,
        "secondary_judge": cfg.judges.validation_judge.model_id,
    }
    write_json(os.path.join(cfg.output_dir, "elicitation", "judge_agreement.json"), result)
    logger.info("Judge agreement: r=%.3f, %%within1=%.1f (n=%d)",
                agreement.pearson_r, agreement.pct_within_one, agreement.n)
    return result
