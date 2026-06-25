"""Section 2 driver: sample rollouts across all conditions for a target model,
score every response with the judge, and persist results.

Also implements the judge-reliability cross-check (re-score a random subset with
the secondary judge and report agreement).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import EvalConfig, Registry
from ..models import get_infra, get_target
from ..welfare import WelfarePolicy
from .conditions import build_rollout_specs
from .judge import score_response, score_rollout
from .rollout import run_rollout


def run_evaluation(
    registry: Registry,
    eval_cfg: EvalConfig,
    model_name: str,
    *,
    scale: str = "default",
    policy: WelfarePolicy | None = None,
    out_dir: str | Path = "outputs/eval",
    seed: int = 0,
    progress: bool = True,
) -> list[dict]:
    """Run the full Section 2 evaluation for one model. Returns rollout dicts."""
    policy = policy or WelfarePolicy.from_env()
    n = eval_cfg.samples_for(scale)
    if scale == "full":
        policy.require_ack("full_scale")

    model = get_target(registry, model_name)
    judge = get_infra(registry, "judge")
    temp = float(eval_cfg.sampling["temperature"])
    max_tokens = int(eval_cfg.sampling["max_tokens"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.jsonl"

    rollouts: list[dict] = []
    conditions = eval_cfg.conditions
    iterator = conditions
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(conditions, desc=f"eval {model_name}", unit="cond")
        except ImportError:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        for cond in iterator:
            if cond["category"] == "extended" and not policy.allows("extended"):
                print(
                    "[welfare] skipping the 8-turn 'extended' distress-spiral "
                    "condition (gated). Re-run with --i-understand-welfare to "
                    "include it."
                )
                continue
            specs = build_rollout_specs(cond, n, seed=seed)
            for spec in specs:
                ro = run_rollout(
                    model, spec, temperature=temp, max_tokens=max_tokens, policy=policy
                )
                score_rollout(judge, ro)
                d = ro.to_dict()
                rollouts.append(d)
                f.write(json.dumps(d) + "\n")
    return rollouts


def crosscheck_judge(
    registry: Registry,
    eval_cfg: EvalConfig,
    rollouts: list[dict],
    *,
    out_path: str | Path = "outputs/eval/judge_crosscheck.json",
    seed: int = 0,
) -> dict:
    """Re-score a random subset of responses with the secondary judge and report
    agreement (Pearson r and within-1-point rate), as in Section 2.1."""
    cross = get_infra(registry, "judge_crosscheck")
    # flatten (response_text, primary_score)
    pairs = []
    for ro in rollouts:
        for r in ro["responses"]:
            if r.get("score") is not None:
                pairs.append((r["text"], int(r["score"])))
    rng = random.Random(seed)
    k = min(eval_cfg.judge.get("crosscheck_sample", 260), len(pairs))
    sample = rng.sample(pairs, k)

    primary, secondary = [], []
    for text, ps in sample:
        res = score_response(cross, text)
        primary.append(ps)
        secondary.append(res.rating)

    within_one = sum(1 for a, b in zip(primary, secondary) if abs(a - b) <= 1)
    result = {
        "n": k,
        "within_one_point_rate": within_one / k if k else None,
        "pearson_r": _pearson(primary, secondary),
        "primary": primary,
        "secondary": secondary,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _pearson(a: list[int], b: list[int]) -> float | None:
    try:
        from scipy.stats import pearsonr

        if len(a) < 2:
            return None
        r, _ = pearsonr(a, b)
        return float(r)
    except Exception:
        return None
