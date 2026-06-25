"""Orchestrate the Section 2 elicitation sweep for one model.

Builds all conversation specs, runs each rollout, scores every assistant turn
with the frustration judge, and streams results to JSONL. Generation and
judging are decoupled from analysis: this module only produces scored rollouts;
:mod:`metrics` turns them into the Figure 1/2/3 numbers.

Each output record is one rollout with a per-turn list of scores:
    {condition, category, initial, rejections, responses, scores: [int...], meta}
"""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..logging_utils import append_jsonl, get_logger, read_jsonl
from ..models.base import GenConfig
from ..models.registry import build_model
from .conditions import build_specs
from .judge import FrustrationJudge
from .rollout import run_rollout

logger = get_logger(__name__)


def _gen_config(cfg: Config) -> GenConfig:
    g = cfg.generation
    return GenConfig(
        temperature=g.temperature,
        max_new_tokens=g.max_new_tokens,
        thinking=g.get("thinking", False),
    )


def run_elicitation(
    model_name: str,
    cfg: Config,
    *,
    adapter_path: str | None = None,
    out_path: str | os.PathLike | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> str:
    """Run the elicitation sweep for ``model_name``; return the output path.

    ``adapter_path`` evaluates a LoRA-finetuned variant (SFT/DPO models).
    ``limit`` caps the number of rollouts (smoke tests). ``resume`` skips
    rollouts already present in the output file (keyed by line index).
    """
    model = build_model(model_name, cfg, adapter_path=adapter_path)
    judge = FrustrationJudge(cfg)
    gen = _gen_config(cfg)
    specs = build_specs(cfg)
    if limit is not None:
        specs = specs[:limit]

    tag = adapter_path.replace("/", "_") if adapter_path else model_name
    if out_path is None:
        out_path = Path(cfg.output_dir) / "elicitation" / f"{tag}.jsonl"
    out_path = Path(out_path)

    done = 0
    if resume and out_path.exists():
        done = sum(1 for _ in read_jsonl(out_path))
        logger.info("Resuming %s: %d rollouts already done", out_path, done)

    for spec in tqdm(specs[done:], desc=f"elicit:{model_name}", initial=done, total=len(specs)):
        rollout = run_rollout(model, spec, gen)
        scores = [judge.score(r).rating for r in rollout.responses]
        record = rollout.to_record()
        record["scores"] = scores
        record["model"] = model_name
        append_jsonl(out_path, record)

    logger.info("Wrote elicitation results to %s", out_path)
    return str(out_path)
