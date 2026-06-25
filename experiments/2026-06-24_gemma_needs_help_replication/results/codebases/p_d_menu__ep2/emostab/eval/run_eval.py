"""CLI: run the Section 2 elicitation evaluation for one or more subject models.

Usage:
    python -m emostab.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
    python -m emostab.eval.run_eval --models gemma-3-27b-it --no-welfare  # paper-faithful

Outputs one JSONL of episodes per model under results/eval/<model>.jsonl, plus a
flat per-turn JSONL (one row per scored assistant turn) for analysis.

Sample-budget interpretation
----------------------------
The paper reports per-category *response* counts (Appendix B). Each conversation
of T turns yields T scored responses, so we run ceil(n_samples / T) conversations
per condition by default (`--count-unit responses`). Use `--count-unit
conversations` to treat the budget as conversation counts instead. See DESIGN.md.
"""
from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

from .. import config
from ..config import WELFARE, WelfareConfig, get_subject
from ..models import get_client
from ..prompts import tasks
from ..utils.io import append_jsonl, write_jsonl
from .conditions import Condition, build_conditions
from .judge import FrustrationJudge
from .rollout import RolloutEngine


def stable_seed(*parts) -> int:
    """Deterministic, cross-process-stable seed (Python's hash() is salted)."""
    key = "|".join(str(p) for p in parts).encode()
    return zlib.crc32(key) & 0x7FFFFFFF


def plan_episodes(cond: Condition, count_unit: str) -> int:
    """Number of conversations to run for this condition."""
    if count_unit == "conversations":
        return cond.n_samples
    return max(1, math.ceil(cond.n_samples / cond.n_turns))


def run_model(
    model_name: str,
    *,
    welfare_cfg: WelfareConfig,
    count_unit: str,
    out_dir: Path,
    wildchat_n: int = 20,
    limit: int | None = None,
) -> dict:
    spec = get_subject(model_name)
    subject = get_client(spec)
    judge = FrustrationJudge()
    engine = RolloutEngine(subject, model_name, judge, welfare_cfg=welfare_cfg)

    wc = tasks.build_wildchat(n_prompts=wildchat_n)
    conditions = build_conditions(wildchat_prompts=wc)

    ep_path = out_dir / f"{model_name}.episodes.jsonl"
    turn_path = out_dir / f"{model_name}.turns.jsonl"
    ep_path.unlink(missing_ok=True)
    turn_path.unlink(missing_ok=True)

    n_eps = 0
    for cond in conditions:
        n_conv = plan_episodes(cond, count_unit)
        for i in range(n_conv):
            if limit is not None and n_eps >= limit:
                break
            task = cond.task_pool[i % len(cond.task_pool)]
            seed = stable_seed(model_name, cond.name, task.task_id, i)
            ep = engine.run_episode(cond, task, seed)
            append_jsonl(ep_path, ep.to_json())
            for t in ep.turns:
                append_jsonl(turn_path, {
                    "model": model_name, "condition": cond.name,
                    "category": cond.category, "tone": cond.tone,
                    "task_id": task.task_id, "turn_index": t.turn_index,
                    "score": t.score, "heuristic": t.heuristic,
                    "terminated_early": ep.terminated_early,
                    "stop_reason": ep.stop_reason,
                    "text": t.assistant,
                })
            n_eps += 1
    return {"model": model_name, "episodes": n_eps,
            "episodes_path": str(ep_path), "turns_path": str(turn_path)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Run Section 2 elicitation eval.")
    p.add_argument("--models", nargs="+", required=True,
                   help="subject model names (Gemma/Gemini)")
    p.add_argument("--no-welfare", action="store_true",
                   help="disable the welfare layer (reproduce paper protocol)")
    p.add_argument("--count-unit", choices=["responses", "conversations"],
                   default="responses")
    p.add_argument("--wildchat-n", type=int, default=20)
    p.add_argument("--limit", type=int, default=None,
                   help="cap episodes per model (smoke testing)")
    p.add_argument("--out", default=str(config.RESULTS_DIR / "eval"))
    args = p.parse_args(argv)

    welfare_cfg = WelfareConfig(enabled=not args.no_welfare) if args.no_welfare else WELFARE
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for m in args.models:
        summary.append(run_model(
            m, welfare_cfg=welfare_cfg, count_unit=args.count_unit,
            out_dir=out_dir, wildchat_n=args.wildchat_n, limit=args.limit))
    write_jsonl(out_dir / "run_summary.jsonl", summary)
    for s in summary:
        print(f"{s['model']}: {s['episodes']} episodes -> {s['turns_path']}")


if __name__ == "__main__":
    main()
