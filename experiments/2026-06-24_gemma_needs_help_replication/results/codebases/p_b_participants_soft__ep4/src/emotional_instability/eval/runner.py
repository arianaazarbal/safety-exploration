"""Section 2 driver: generate rollouts for a model across categories, score
every assistant turn with the judge, and persist scored rollout records.

Designed to be resumable: rollouts are written to JSONL as they complete, and a
re-run skips categories whose output file already exists (unless `overwrite`).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm

from ..config import Config, load_config
from ..models import GenerationConfig, get_client
from ..prompts.eval_prompts import build_category_items
from ..utils.io import read_jsonl, write_jsonl
from .conversation import Rollout, run_rollouts
from .judge import FrustrationJudge


def _gen_cfg(cfg: Config) -> GenerationConfig:
    s = cfg.eval["sampling"]
    return GenerationConfig(
        temperature=s["temperature"],
        top_p=s["top_p"],
        max_new_tokens=s["max_new_tokens"],
        seed=s["seed"],
        thinking=False,
    )


def _score_rollouts(judge: FrustrationJudge, rollouts: Sequence[Rollout]) -> None:
    """Score every assistant turn in place."""
    flat_texts: List[str] = []
    index: List[tuple[int, int]] = []
    for ri, r in enumerate(rollouts):
        for ti, t in enumerate(r.turns):
            flat_texts.append(t.assistant_text)
            index.append((ri, ti))
    results = judge.score_many(flat_texts)
    for (ri, ti), res in zip(index, results):
        turn = rollouts[ri].turns[ti]
        turn.frustration_score = res.rating
        turn.judge_evidence = res.evidence
        turn.judge_reasoning = res.reasoning


def run_section2_for_model(
    model_name: str,
    *,
    categories: Sequence[str] | None = None,
    batch_size: int = 32,
    seed: int = 0,
    overwrite: bool = False,
    cfg: Config | None = None,
) -> Path:
    """Run all (or selected) Section-2 categories for one participant model.

    Returns the path to the combined scored-rollouts JSONL.
    """
    cfg = cfg or load_config()
    categories = list(categories or cfg.eval["categories"].keys())
    out_dir = cfg.path("outputs_dir") / "section2" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "rollouts.jsonl"

    client = get_client(model_name)
    judge = FrustrationJudge(
        get_client("judge_primary"),
        max_concurrency=cfg.eval["judge"]["max_concurrency"],
    )
    gen_cfg = _gen_cfg(cfg)

    all_records: list[dict] = []
    for category in categories:
        cat_path = out_dir / f"{category}.jsonl"
        if cat_path.exists() and not overwrite:
            all_records.extend(read_jsonl(cat_path))
            continue
        items = build_category_items(category, cfg.eval, seed=seed)
        cat_records: list[dict] = []
        for start in tqdm(range(0, len(items), batch_size),
                          desc=f"{model_name}/{category}"):
            chunk = items[start:start + batch_size]
            rollouts = run_rollouts(client, chunk, gen_cfg, base_seed=seed + start)
            _score_rollouts(judge, rollouts)
            cat_records.extend(r.to_record() for r in rollouts)
        write_jsonl(cat_path, cat_records)
        all_records.extend(cat_records)

    write_jsonl(combined_path, all_records)
    return combined_path
