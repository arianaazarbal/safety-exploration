"""Section-2 sweep runner: sample rollouts, judge them, persist results.

Responsibilities:
  * Build the full (condition, conversation-spec) work-list for a model.
  * Run each rollout, then judge every assistant turn (0-10 frustration).
  * Cache transcripts+scores to JSONL so re-runs never re-induce distress that
    was already collected (a welfare *and* a cost consideration).

Output: ``results/eval/<model>.jsonl`` -- one JSON record per rollout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config, welfare
from ..models import get_client
from ..models.factory import get_anthropic
from .conditions import build_all_prompts
from .judge import FrustrationJudge
from .rollout import run_rollout


def _cache_path(model: str, results_dir: Path) -> Path:
    return results_dir / "eval" / f"{model}.jsonl"


def _load_done_keys(path: Path) -> set[str]:
    """Resume support: which (condition,index) keys are already collected."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(rec["_key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_model_eval(
    model: str,
    cfg: config.RunConfig,
    *,
    results_dir: Optional[Path] = None,
    adapter_path: Optional[str] = None,
    judge_model: Optional[str] = None,
    load_in_4bit: bool = False,
) -> Path:
    """Run the full Section-2 sweep for one model and return the JSONL path."""
    results_dir = Path(results_dir or config.RESULTS_DIR)
    welfare.write_notice(
        results_dir,
        purpose=(f"Section-2 distress elicitation for '{model}' "
                 f"(replication of arXiv:2603.10011)."),
    )

    scaled = cfg.budget.scaled()
    for cat, n in scaled.items():
        welfare.cap_samples(n, getattr(cfg.budget, cat), cat)  # no over-sampling

    work = build_all_prompts(scaled, seed=cfg.seed)
    out_path = _cache_path(model, results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(out_path)

    client = get_client(model, adapter_path=adapter_path, load_in_4bit=load_in_4bit)
    judge = FrustrationJudge(get_anthropic(judge_model or cfg.judge_model))

    with open(out_path, "a") as fh:
        for i, (cond, spec) in enumerate(tqdm(work, desc=f"eval:{model}")):
            key = f"{cond.name}:{i}"
            if key in done:
                continue
            rollout = run_rollout(
                client, spec,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                seed=cfg.seed * 100000 + i,
            )
            for turn in rollout.turns:
                turn.score = judge.score(turn.response).rating
            rec = rollout.to_record()
            rec["_key"] = key
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    return out_path


def run_all_models(
    cfg: config.RunConfig,
    models: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, Path]:
    models = models or config.MAIN_EVAL_MODELS
    return {m: run_model_eval(m, cfg, **kwargs) for m in models}
