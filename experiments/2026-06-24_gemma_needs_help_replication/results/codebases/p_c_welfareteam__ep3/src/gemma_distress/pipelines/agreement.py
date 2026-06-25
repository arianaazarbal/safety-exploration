"""Section 2.1 driver: inter-judge agreement validation.

Reproduces the paper's judge-reliability check: sample N already-scored responses
(default 260), re-score them with the secondary judge (GPT-5-mini) using the same
rubric, and report Pearson r, % within one point, and mean absolute difference.

Re-scoring needs the conversation context, which the score records drop to save
space, so we rebuild the (context, response) tasks from the rollouts and join on
the scored-task ``id``.
"""
from __future__ import annotations

import json
import random

from ..config import Config
from ..io_utils import load_jsonl
from ..judge import build_judge
from ..judge.agreement import compute_agreement
from ..judge.scoring import iter_scoring_tasks
from . import artefact, log


def run(config: Config, *, models: list[str] | None = None) -> str:
    jcfg = config.experiment["judge"]
    n_sample = jcfg["agreement_sample_size"]
    rng = random.Random(jcfg["agreement_seed"])

    models = models or config.all_targets()
    # Join scored ids -> (context, response) from rollouts.
    task_by_id: dict[str, dict] = {}
    primary: dict[str, int] = {}
    for m in models:
        rollouts = load_jsonl(artefact("section2", "rollouts", f"{m}.jsonl"))
        for t in iter_scoring_tasks(rollouts):
            task_by_id[t["id"]] = t
        for s in load_jsonl(artefact("section2", "scores", f"{m}.jsonl")):
            if s.get("score") is not None:
                primary[s["id"]] = int(s["score"])

    eligible = [i for i in primary if i in task_by_id]
    if not eligible:
        raise RuntimeError("no scored responses found; run elicit + judge first")
    sample_ids = rng.sample(eligible, min(n_sample, len(eligible)))
    log(f"re-scoring {len(sample_ids)} responses with the secondary judge")

    secondary_judge = build_judge("frustration_secondary", config)
    prim_scores, sec_scores = [], []
    for i, sid in enumerate(sample_ids, 1):
        t = task_by_id[sid]
        res = secondary_judge.score_one(t["context"], t["response"])
        if res.score is None:
            continue
        prim_scores.append(primary[sid])
        sec_scores.append(res.score)
        if i % 50 == 0:
            log(f"agreement: {i}/{len(sample_ids)}")

    agr = compute_agreement(prim_scores, sec_scores)
    report = {
        **agr.__dict__,
        "primary_judge": config.judge("frustration_primary").model_id,
        "secondary_judge": config.judge("frustration_secondary").model_id,
    }
    out_path = artefact("section2", "agreement.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"agreement: r={agr.pearson_r:.3f}, within-1={agr.within_one_point:.0%}, "
        f"n={agr.n} -> {out_path}")
    return str(out_path)
