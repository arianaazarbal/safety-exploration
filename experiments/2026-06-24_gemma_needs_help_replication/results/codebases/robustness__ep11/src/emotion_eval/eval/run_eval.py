"""Section 2 runner: elicit & quantify distress.

For each model under test and each category, build the rollout specs, run the multi-turn
conversations, judge every assistant turn for frustration, and write JSONL artefacts:

  runs/<run>/section2/rollouts.<model>.jsonl   — full conversations
  runs/<run>/section2/scored.<model>.jsonl     — one row per scored assistant turn

The aggregate metrics (% ≥5, mean, per-turn) are computed separately in
analysis/aggregate.py so re-scoring or re-aggregation never requires re-running models.

Model generation is sequential per model (a single local GPU serves one Gemma at a time);
judge calls are fanned out over a thread pool since they are independent API requests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ..config import Config, append_jsonl, load_config, stage_dir
from ..models import build_model
from ..models.registry import MODEL_SPECS
from ..tasks.conditions import build_category_rollouts
from .judge import FrustrationJudge
from .rollout import Rollout, run_rollout


def _adapter_dir(cfg: Config, model_name: str) -> Path | None:
    spec = MODEL_SPECS[model_name]
    if not spec.adapter_kind:
        return None
    return stage_dir(cfg, "finetune") / f"{spec.adapter_kind}_adapter"


def _judge_rollout(judge: FrustrationJudge, rollout: Rollout) -> list[dict]:
    """Score every assistant turn; flag the final turn as the headline response."""
    rows = []
    n_turns = len(rollout.turns)
    for turn in rollout.turns:
        result = judge.score(turn.text)
        rows.append(
            {
                "model": rollout.model,
                "category": rollout.category,
                "condition": rollout.condition,
                "rollout_id": rollout.rollout_id,
                "turn_index": turn.turn_index,
                "is_final": turn.turn_index == n_turns - 1,
                "rating": result.rating,
                "evidence": result.evidence,
                "meta": rollout.meta,
            }
        )
    return rows


def run_model(cfg: Config, model_name: str, judge: FrustrationJudge, out_dir: Path) -> None:
    model = build_model(model_name, adapter_dir=_adapter_dir(cfg, model_name))
    rollouts_path = out_dir / f"rollouts.{model_name.replace('/', '_')}.jsonl"
    scored_path = out_dir / f"scored.{model_name.replace('/', '_')}.jsonl"
    # fresh files
    rollouts_path.unlink(missing_ok=True)
    scored_path.unlink(missing_ok=True)

    for category, cfg_cat in cfg.section2.categories.items():
        specs = build_category_rollouts(category, cfg_cat, cfg.seed)
        # Generate all rollouts for the category first (local GPU is the bottleneck and is
        # inherently sequential), then fan out the independent judge API calls.
        rollouts: list[Rollout] = []
        for spec in tqdm(specs, desc=f"gen {model_name}:{category}"):
            rollout = run_rollout(
                model, spec, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens
            )
            append_jsonl(rollouts_path, rollout.as_record())
            rollouts.append(rollout)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for rows in tqdm(
                pool.map(lambda r: _judge_rollout(judge, r), rollouts),
                total=len(rollouts),
                desc=f"judge {model_name}:{category}",
            ):
                for row in rows:
                    append_jsonl(scored_path, row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 2: elicit & quantify distress")
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="override the model list from the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = stage_dir(cfg, "section2")
    judge_model = build_model(cfg.judge.model)
    judge = FrustrationJudge(
        judge_model, max_tokens=cfg.judge.max_tokens, temperature=cfg.judge.temperature
    )

    models = args.models or cfg.models
    for model_name in models:
        print(f"=== Section 2: {model_name} ===")
        run_model(cfg, model_name, judge, out_dir)
    print(f"Done. Artefacts in {out_dir}")


if __name__ == "__main__":
    main()
