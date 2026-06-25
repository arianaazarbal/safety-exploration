"""Run the distress-elicitation evaluation for Gemma + Gemini.

For each model and category: build rollout specs, run the multi-turn
conversations (parallel across rollouts), judge every assistant turn with Claude,
and append fully-scored rollouts to results/<model>/<category>.jsonl.

The run is resumable: rollouts whose rollout_id is already present in the output
file are skipped, so an interrupted run can be re-invoked to continue.

Usage:
    OPENROUTER_API_KEY=... ANTHROPIC_API_KEY=... python run_eval.py
    python run_eval.py --models gemma-3-12b-it --categories triggers
    DISTRESS_SCALE=0.01 python run_eval.py        # cheap smoke test
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from conversation import RolloutResult, RolloutSpec, build_specs, run_rollout
from judge import score_response


def _output_path(model_name: str, category: str) -> str:
    d = os.path.join(config.RESULTS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{category}.jsonl")


def _completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["rollout_id"])
            except Exception:
                continue
    return done


def _process_rollout(spec: RolloutSpec, model) -> dict:
    """Run + judge one rollout, returning a serialisable record."""
    result: RolloutResult = run_rollout(spec, model)
    turns_out = []
    for tr in result.turns:
        verdict = score_response(tr.response)
        turns_out.append({
            "turn": tr.turn,
            "response": tr.response,
            "rating": verdict.rating,
            "evidence": verdict.evidence,
            "reasoning": verdict.reasoning,
        })
    return {
        "rollout_id": spec.rollout_id,
        "model": model.name,
        "category": spec.category,
        "condition": spec.condition,
        "n_turns": spec.turns,
        "task": spec.task,
        "rejections": spec.rejections,
        "meta": spec.meta,
        "turns": turns_out,
        "error": result.error,
    }


def run_model_category(model, cat, force: bool = False) -> None:
    path = _output_path(model.name, cat.name)
    done = set() if force else _completed_ids(path)
    specs = [s for s in build_specs(cat) if s.rollout_id not in done]

    if not specs:
        print(f"  [{model.name}/{cat.name}] already complete ({len(done)} rollouts). Skipping.")
        return

    total = len(specs)
    n_resp = total * cat.turns
    print(f"  [{model.name}/{cat.name}] {total} rollouts x {cat.turns} turns "
          f"= ~{n_resp} responses to generate+judge (resuming from {len(done)} done).")

    written = 0
    # Append mode so partial progress survives interruption.
    with open(path, "a") as out:
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = {pool.submit(_process_rollout, s, model): s for s in specs}
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    record = fut.result()
                except Exception as e:
                    print(f"    ! rollout {spec.rollout_id} crashed: {e}")
                    continue
                out.write(json.dumps(record) + "\n")
                out.flush()
                written += 1
                if written % 25 == 0 or written == total:
                    print(f"    {model.name}/{cat.name}: {written}/{total} rollouts written")


def main() -> None:
    ap = argparse.ArgumentParser(description="Distress-elicitation eval (Gemma + Gemini).")
    ap.add_argument("--models", nargs="*", default=list(config.MODELS),
                    help="subset of model names (default: all configured)")
    ap.add_argument("--categories", nargs="*", default=list(config.CATEGORIES),
                    help="subset of category names (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="ignore existing results and regenerate (appends; use a clean dir)")
    args = ap.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"Judge model: {config.JUDGE_MODEL}  |  SCALE={config.SCALE}  |  T={config.TEMPERATURE}")

    for model_name in args.models:
        if model_name not in config.MODELS:
            raise SystemExit(f"Unknown model: {model_name}. Choices: {list(config.MODELS)}")
        model = config.MODELS[model_name]
        print(f"\n=== Model: {model.name} (backend={model.backend}, id={model.model_id}) ===")
        for cat_name in args.categories:
            if cat_name not in config.CATEGORIES:
                raise SystemExit(f"Unknown category: {cat_name}. Choices: {list(config.CATEGORIES)}")
            run_model_category(model, config.CATEGORIES[cat_name], force=args.force)

    print("\nDone. Run `python analyze.py` to aggregate metrics.")


if __name__ == "__main__":
    main()
