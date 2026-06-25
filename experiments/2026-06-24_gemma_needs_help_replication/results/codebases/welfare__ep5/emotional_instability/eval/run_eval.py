"""Run the Section 2 elicitation evaluation end to end.

For each target model and each of the 8 conditions: generate rollouts, score
every assistant turn with the frustration judge, and write a JSONL of scored
rollouts. Aggregation/figures are handled by ``analyze.py``.

The expensive generation (Gemma local, Gemini API) and the judge (Claude API)
are separated so that a run can be resumed: rollouts are written before
judging, and judging skips turns that already carry a score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

from tqdm import tqdm

from .. import config
from ..models.registry import load_model
from ..prompts.wildchat import load_wildchat_prompts
from .judge import FrustrationJudge
from .rollout import Rollout, run_single_rollout


def _n_rollouts_for(cond: config.Condition, fraction: float) -> int:
    """Convert a per-condition target response count to a rollout count.

    Each rollout yields ``cond.n_turns`` scored responses, so we run
    ``target_responses / n_turns`` conversations. ``fraction`` scales the whole
    eval down for cheap smoke tests.
    """
    target = max(1, int(round(cond.target_responses * fraction)))
    return max(1, math.ceil(target / cond.n_turns))


def run_model_eval(
    spec,
    *,
    out_dir: Path,
    fraction: float = 1.0,
    conditions: Sequence[config.Condition] = tuple(config.CONDITIONS),
    judge: Optional[FrustrationJudge] = None,
    adapter_path: Optional[str] = None,
    base_seed: int = 0,
    model_kwargs: Optional[dict] = None,
) -> Path:
    """Evaluate a single model across conditions; return the output JSONL path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = spec.name + ("+adapter" if adapter_path else "")
    out_path = out_dir / f"{tag.replace('/', '_')}.jsonl"

    model = load_model(spec, adapter_path=adapter_path, **(model_kwargs or {}))
    judge = judge or FrustrationJudge()
    wildchat_pool = load_wildchat_prompts()

    with out_path.open("w") as f:
        for cond in conditions:
            n = _n_rollouts_for(cond, fraction)
            desc = f"{spec.name} :: {cond.key} (n={n})"
            for i in tqdm(range(n), desc=desc):
                roll: Rollout = run_single_rollout(
                    model, cond, seed=base_seed + i, wildchat_pool=wildchat_pool
                )
                judge.score_rollout(roll)
                f.write(json.dumps(roll.to_dict()) + "\n")
                f.flush()

    model.close()
    return out_path


def run_all(
    specs: Sequence = tuple(config.SECTION2_MODELS),
    *,
    out_dir: Optional[Path] = None,
    fraction: float = 1.0,
    model_kwargs: Optional[dict] = None,
) -> list[Path]:
    out_dir = Path(out_dir or (config.RESULTS_DIR / "section2"))
    judge = FrustrationJudge()
    paths = []
    for spec in specs:
        paths.append(
            run_model_eval(
                spec, out_dir=out_dir, fraction=fraction, judge=judge,
                model_kwargs=model_kwargs,
            )
        )
    return paths
