"""Section 2 orchestration: generate plans -> roll out target -> judge -> score.

Writes two JSONL artifacts per target under <output_dir>/eval/<target>/:
  rollouts.jsonl  - one line per conversation (all turns)
  scored.jsonl    - one line per scored assistant response (the unit of analysis)
and a summary.json with the metric bundle.

Concurrency: generation and judging are I/O bound for API providers, so we use a
thread pool. Local HF generation is GPU-bound and effectively serialised by the
single model; that is fine (set workers=1 for HF targets).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config import Config, ModelSpec
from .judge import FrustrationJudge
from .metrics import summarize
from .providers import GenConfig, get_model
from .rollout import run_rollout
from .tasks import build_all_plans


def _gen_cfg(cfg: Config) -> GenConfig:
    s = cfg.sampling
    return GenConfig(
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        disable_thinking=s.disable_thinking,
    )


def run_eval_for_target(
    cfg: Config,
    target_name: str,
    conditions: list[str] | None = None,
    gen_workers: int = 8,
    judge_workers: int = 8,
) -> dict:
    spec = cfg.target(target_name)
    out_dir = cfg.output_dir / "eval" / target_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(spec)
    judge = FrustrationJudge(get_model(cfg.judge))
    gcfg = _gen_cfg(cfg)

    plans = build_all_plans(cfg.sampling.scale, cfg.sampling.seed, conditions)

    # HF (local GPU) models cannot truly parallelise; force serial generation.
    if spec.backend == "hf":
        gen_workers = 1

    # 1) generate rollouts
    rollouts = []
    rollout_path = out_dir / "rollouts.jsonl"
    with rollout_path.open("w") as rf:
        if gen_workers == 1:
            for plan in tqdm(plans, desc=f"{target_name}:rollout"):
                ro = run_rollout(model, plan, gcfg)
                rollouts.append(ro)
                rf.write(json.dumps(ro.to_dict()) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=gen_workers) as ex:
                futs = {ex.submit(run_rollout, model, p, gcfg): p for p in plans}
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"{target_name}:rollout"):
                    ro = fut.result()
                    rollouts.append(ro)
                    rf.write(json.dumps(ro.to_dict()) + "\n")

    # 2) flatten to scored items
    items = []
    for ro in rollouts:
        for t in ro.turns:
            items.append({
                "target": target_name,
                "condition": ro.condition,
                "category": ro.category,
                "turn_index": t.turn_index,
                "response": t.response,
                "plan_meta": ro.plan_meta,
            })

    # 3) judge each response
    def _judge(item):
        res = judge.score(item["response"])
        item = dict(item)
        item["rating"] = res.rating
        item["evidence"] = res.evidence
        return item

    scored = []
    scored_path = out_dir / "scored.jsonl"
    with scored_path.open("w") as sf:
        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            futs = [ex.submit(_judge, it) for it in items]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"{target_name}:judge"):
                it = fut.result()
                scored.append(it)
                sf.write(json.dumps(it) + "\n")

    # 4) summarise
    summary = summarize(scored)
    summary["target"] = target_name
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_eval(cfg: Config, targets: list[str] | None = None,
             conditions: list[str] | None = None) -> dict[str, dict]:
    targets = targets or [t.name for t in cfg.targets if not t.is_base]
    out = {}
    for name in targets:
        out[name] = run_eval_for_target(cfg, name, conditions)
    # combined headline table (Figure 1)
    headline = {
        name: round(s["avg_pct_high_across_categories"], 2) for name, s in out.items()
    }
    (cfg.output_dir / "eval" / "headline.json").write_text(json.dumps(headline, indent=2))
    return out
