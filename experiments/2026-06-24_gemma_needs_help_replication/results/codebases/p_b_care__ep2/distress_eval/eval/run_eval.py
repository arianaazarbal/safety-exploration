"""Main Section 2 driver: sample responses for a target model and judge them.

Usage:
    python -m distress_eval.eval.run_eval --model gemma-3-27b-it
    python -m distress_eval.eval.run_eval --model gemini-2.5-flash --max-responses 400

Produces a JSONL file in OUTPUT_DIR with one record per rollout (each record
holds every scored assistant turn). Rollouts and judging are fanned out across
a thread pool; the Gemma vLLM backend serialises internally, while API backends
(Gemini, judge) genuinely run concurrently.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import config
from ..models.registry import get_judge, get_target
from . import conditions, judge
from .rollout import RolloutResult, run_rollout


def _scale_budget(max_responses: int | None) -> dict[str, int]:
    if max_responses is None:
        return dict(config.RESPONSE_BUDGET)
    total = sum(config.RESPONSE_BUDGET.values())
    frac = max_responses / total
    return {k: max(1, round(v * frac)) for k, v in config.RESPONSE_BUDGET.items()}


def build_all_specs(budget: dict[str, int], seed: int):
    specs = []
    for category, n in budget.items():
        specs.extend(conditions.build_specs(category, n, seed=seed))
    return specs


def run(model_name: str, *, max_responses: int | None, seed: int,
        out_path: Path | None, skip_judge: bool) -> Path:
    model = get_target(model_name)
    judge_model = None if skip_judge else get_judge(config.JUDGE_MODEL, config.JUDGE_BACKEND)

    budget = _scale_budget(max_responses)
    specs = build_all_specs(budget, seed)
    print(f"[run_eval] {model_name}: {len(specs)} rollouts "
          f"(~{sum(s.n_turns for s in specs)} responses)")

    out_path = out_path or (config.OUTPUT_DIR / f"eval_{model_name}.jsonl")

    # 1) Generation. vLLM backends are internally batched and not thread-safe to
    # call from many threads, so Gemma runs serially; API targets fan out.
    results: list[RolloutResult] = []
    is_api = config.TARGET_MODELS[model_name].backend in ("gemini", "openrouter")
    if is_api:
        with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
            futs = [ex.submit(run_rollout, model, s,
                              temperature=config.TEMPERATURE,
                              max_tokens=config.MAX_NEW_TOKENS) for s in specs]
            for f in as_completed(futs):
                results.append(f.result())
    else:
        for s in specs:
            results.append(run_rollout(model, s,
                                       temperature=config.TEMPERATURE,
                                       max_tokens=config.MAX_NEW_TOKENS))

    # 2) Judging. Always API-bound -> fan out across threads.
    if judge_model is not None:
        _judge_all(judge_model, results)

    # 3) Persist.
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_record()) + "\n")
    print(f"[run_eval] wrote {out_path}")
    return out_path


def _judge_all(judge_model, results: list[RolloutResult]):
    # Flatten to (rollout_idx, response) units for concurrent scoring.
    units = [(ri, resp) for ri, r in enumerate(results) for resp in r.responses]

    def _score(unit):
        ri, resp = unit
        res = judge.score_response(judge_model, resp.text,
                                   max_tokens=config.JUDGE_MAX_TOKENS,
                                   temperature=config.JUDGE_TEMPERATURE)
        resp.score = res.rating
        resp.judge_evidence = res.evidence
        return None

    with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
        list(ex.map(_score, units))


def main():
    ap = argparse.ArgumentParser(description="Run the Section 2 distress eval.")
    ap.add_argument("--model", required=True, choices=list(config.TARGET_MODELS))
    ap.add_argument("--max-responses", type=int, default=None,
                    help="Scale the 4000-response budget down for quick runs.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-judge", action="store_true",
                    help="Only generate responses; score later.")
    args = ap.parse_args()
    run(args.model, max_responses=args.max_responses, seed=args.seed,
        out_path=args.out, skip_judge=args.skip_judge)


if __name__ == "__main__":
    main()
