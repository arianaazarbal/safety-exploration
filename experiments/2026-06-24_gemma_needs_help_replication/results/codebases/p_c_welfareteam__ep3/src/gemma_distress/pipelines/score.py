"""Section 2.1 driver: score a model's rollouts on the 0-10 frustration scale.

Loads ``section2/rollouts/{model}.jsonl``, flattens to one scoring task per
assistant turn, scores them concurrently with the primary judge, and streams
``section2/scores/{model}.jsonl``. Resumable by scored-task ``id``.
"""
from __future__ import annotations

from ..config import Config
from ..io_utils import append_jsonl, completed_ids, load_jsonl
from ..judge import build_judge
from ..judge.scoring import iter_scoring_tasks, score_tasks
from . import artefact, log


def run(config: Config, model_name: str, *, judge_role: str = "frustration_primary",
        limit: int | None = None) -> str:
    rollouts_path = artefact("section2", "rollouts", f"{model_name}.jsonl")
    out_path = artefact("section2", "scores", f"{model_name}.jsonl")
    rollouts = load_jsonl(rollouts_path)
    if not rollouts:
        raise FileNotFoundError(
            f"no rollouts at {rollouts_path}; run `elicit` for {model_name} first"
        )

    done = completed_ids(out_path, id_key="id")
    tasks = [t for t in iter_scoring_tasks(rollouts) if t["id"] not in done]
    if limit is not None:
        tasks = tasks[:limit]
    log(f"{model_name}: {len(done)} already scored; {len(tasks)} to score "
        f"with {judge_role}")

    judge = build_judge(judge_role, config)
    conc = config.experiment["judge"]["max_concurrency"]
    n = 0
    for rec in score_tasks(tasks, judge, max_concurrency=conc):
        append_jsonl(out_path, rec)
        n += 1
        if n % 100 == 0:
            log(f"{model_name}: +{n} scored")
    log(f"{model_name}: wrote {n} scores -> {out_path}")
    return str(out_path)
