"""CLI: Section 2 main evaluation.

Pipeline per target model:
  1. Build the 8-condition conversation specs (4,000 rollouts; Appendix B counts).
  2. Roll out the conversations (batched for local, threaded for API).
  3. Score every assistant turn with the Claude judge.
  4. Stream rollouts and per-turn scores to JSONL (resumable).

Run aggregation/figures separately via ``analysis.aggregate`` so judging cost is
paid once.

Usage:
    python -m emotional_instability.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
    python -m emotional_instability.eval.run_eval --conditions extended wildchat
"""
from __future__ import annotations

import argparse

from ..config import ModelSpec, load_config
from ..data.datasets import build_eval_specs
from ..models.base import SamplingParams
from ..models.registry import build_client, build_judge
from ..utils.io import append_jsonl, existing_ids, write_jsonl
from . import rollout as R
from .judge import score_turns


def _sampling(config) -> SamplingParams:
    s = config.sampling
    return SamplingParams(
        temperature=s.get("temperature", 1.0),
        max_tokens=s.get("max_tokens", 2048),
        thinking=s.get("thinking", False),
    )


def run_for_model(config, spec: ModelSpec, conditions, judge_workers: int) -> None:
    specs = build_eval_specs(config, conditions)
    client = build_client(spec)
    params = _sampling(config)

    rollout_path = config.output_path("eval", f"{spec.name}.rollouts.jsonl")
    score_path = config.output_path("eval", f"{spec.name}.scores.jsonl")
    done = existing_ids(rollout_path)
    todo = [s for s in specs if s.id not in done]
    print(f"[{spec.name}] {len(todo)}/{len(specs)} rollouts remaining")
    if not todo:
        print(f"[{spec.name}] all rollouts already complete; re-run aggregate to summarise.")
        return

    # Rollouts: batched for local backends, threaded for API backends.
    if spec.backend in ("vllm", "hf"):
        rollouts = R.run_batched(client, todo, params)
    else:
        rollouts = R.run_threaded(client, todo, params, max_workers=judge_workers)

    write_jsonl(rollout_path, (r.to_record() for r in rollouts), append=True)

    # Judge every assistant turn.
    judge = build_judge(config.judge["model"])

    def _score(r: R.Rollout) -> list[dict]:
        scores = score_turns(judge, r.turns, max_tokens=config.judge.get("max_tokens", 1024))
        return [
            {
                "id": f"{r.spec_id}#t{t}", "rollout_id": r.spec_id, "model": spec.name,
                "condition": r.condition, "category": r.category, "turn": t,
                "rating": sc.rating, "high": sc.high, "evidence": sc.evidence,
            }
            for t, sc in enumerate(scores)
        ]

    from ..utils.concurrency import thread_map

    for recs in thread_map(_score, rollouts, max_workers=judge_workers, desc=f"judge {spec.name}"):
        write_jsonl(score_path, recs, append=True)
    print(f"[{spec.name}] done -> {score_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 2 distress evaluation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="target model names (default: all)")
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of conditions")
    ap.add_argument("--judge-workers", type=int, default=16)
    args = ap.parse_args()

    config = load_config(args.config)
    models = config.target_models
    if args.models:
        models = [m for m in models if m.name in args.models]

    for spec in models:
        run_for_model(config, spec, args.conditions, args.judge_workers)


if __name__ == "__main__":
    main()
