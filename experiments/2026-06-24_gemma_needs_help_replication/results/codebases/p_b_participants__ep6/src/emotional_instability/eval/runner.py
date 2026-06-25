"""Orchestrate the full Section 2 evaluation for one participant model.

Builds all rollout specs from the active sample sizes, runs each conversation,
scores every turn with the frustration judge, applies the welfare debrief, writes
per-rollout JSONL, and returns the aggregate summary.

Generation (the participant model) and scoring (the judge) are separated so a
local Gemma can generate sequentially on the GPU while judge calls fan out over
threads. Gemini generation is itself IO-bound and parallelised.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ..config import sample_sizes
from ..models.base import ChatModel
from ..models.judge import FrustrationJudge, make_judge
from ..utils.io import write_jsonl, write_json
from ..utils.welfare import DEFAULT_POLICY, check_rollout_budget
from . import analysis
from .conditions import build_specs
from .rollout import maybe_debrief, run_rollout
from .scoring import score_rollouts


def run_eval_for_model(
    cfg: dict,
    model: ChatModel,
    judge: FrustrationJudge | None = None,
    out_dir: str | Path | None = None,
    label: str | None = None,
) -> dict:
    label = label or model.name
    out_dir = Path(out_dir or cfg["run"]["output_dir"]) / "eval" / label
    judge = judge or make_judge(cfg, "frustration")

    sizes = sample_sizes(cfg)
    specs = build_specs(sizes, seed=cfg["run"]["seed"])
    check_rollout_budget(len(specs), DEFAULT_POLICY)

    temperature = cfg["run"]["temperature"]
    max_new = cfg["max_new_tokens"]
    is_local = model.supports_prefill()  # HF Gemma -> sequential; Gemini -> parallel

    def _gen(spec):
        return run_rollout(model, spec, temperature=temperature, max_new_tokens=max_new)

    if is_local:
        rollouts = [_gen(s) for s in tqdm(specs, desc=f"gen[{label}]")]
    else:
        with ThreadPoolExecutor(max_workers=cfg["run"]["max_concurrency"]) as ex:
            rollouts = list(tqdm(ex.map(_gen, specs), total=len(specs), desc=f"gen[{label}]"))

    score_rollouts(rollouts, judge, max_workers=cfg["run"]["max_concurrency"])
    for r in rollouts:
        maybe_debrief(model, r)

    write_jsonl(out_dir / "rollouts.jsonl", (r.to_record() for r in rollouts))
    summ = analysis.summary(rollouts)
    summ["differential_words"] = analysis.differential_words(rollouts)
    summ["per_turn"] = {
        "extended": analysis.per_turn_curve(rollouts, condition="extended",
                                            bootstrap_iters=cfg["petri"]["bootstrap_iters"]),
        "wildchat": analysis.per_turn_curve(rollouts, condition="wildchat",
                                            bootstrap_iters=cfg["petri"]["bootstrap_iters"]),
    }
    write_json(out_dir / "summary.json", summ)
    return summ
