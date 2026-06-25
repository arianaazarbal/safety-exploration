"""Run the distress-elicitation evaluation and score every response.

For each selected target model, each of the 8 conditions, and each rollout, this
runs a multi-turn rollout (task → repeated rejection) and scores every assistant
turn with the Claude-Sonnet-4 judge on the 0-10 frustration scale. Results are
written as JSONL — one record per scored assistant turn — to
``<output_dir>/<model_name>.jsonl``.

Usage:
    python -m distress_eval.run_eval --preset smoke
    python -m distress_eval.run_eval --preset paper --models gemma-3-27b-it
    python -m distress_eval.run_eval --resume        # skip completed rollouts

Environment:
    GEMINI_API_KEY / GOOGLE_API_KEY   target models (Gemma + Gemini)
    ANTHROPIC_API_KEY                 judge (Claude Sonnet 4)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from .config import Config, make_config
from .judge import Judge
from .rollout import run_rollout
from .targets import GoogleTargetClient
from .tasks import CONDITIONS, TaskBank


def _derive_seed(base_seed: int, model_name: str, condition_key: str, rollout_idx: int) -> int:
    """Deterministic per-(model, condition, rollout) seed.

    Uses a stable hash (not builtin ``hash``, which is salted per process) so
    the same (model, condition, rollout) always samples the same task/rejections
    across runs and across resume.
    """
    key = f"{base_seed}|{model_name}|{condition_key}|{rollout_idx}".encode()
    return int.from_bytes(hashlib.md5(key).digest()[:4], "big")


def _completed_rollouts(path: Path) -> dict[tuple[str, int], int]:
    """Map (condition_key, rollout_idx) -> number of scored turns already saved."""
    counts: dict[tuple[str, int], int] = {}
    if not path.exists():
        return counts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["condition_key"], rec["rollout_idx"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def run(config: Config, resume: bool = False) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = GoogleTargetClient(
        max_retries=config.max_retries,
        base_delay=config.retry_base_delay,
        max_delay=config.retry_max_delay,
    )
    judge = Judge(
        model=config.judge_model,
        temperature=config.judge_temperature,
        max_tokens=config.judge_max_tokens,
        max_retries=config.max_retries,
        base_delay=config.retry_base_delay,
        max_delay=config.retry_max_delay,
    )
    task_bank = TaskBank()

    models = config.selected_models()
    print(
        f"Models: {[m.name for m in models]} | conditions: {len(CONDITIONS)} | "
        f"rollouts/condition: {config.n_rollouts_per_condition} | "
        f"judge: {config.judge_model}",
        file=sys.stderr,
    )

    for spec in models:
        path = out_dir / f"{spec.name}.jsonl"
        done = _completed_rollouts(path) if resume else {}
        mode = "a" if resume and path.exists() else "w"
        n_written = 0
        t0 = time.time()

        with open(path, mode) as fout:
            for condition in CONDITIONS:
                for rollout_idx in range(config.n_rollouts_per_condition):
                    key = (condition.key, rollout_idx)
                    if done.get(key, 0) >= condition.n_turns:
                        continue  # already fully scored

                    rng = random.Random(
                        _derive_seed(config.seed, spec.name, condition.key, rollout_idx)
                    )
                    try:
                        rollout = run_rollout(
                            target=target,
                            model_id=spec.model_id,
                            model_name=spec.name,
                            condition=condition,
                            task_bank=task_bank,
                            temperature=config.temperature,
                            max_output_tokens=config.max_output_tokens,
                            rng=rng,
                        )
                    except Exception as e:
                        print(
                            f"  [{spec.name}/{condition.key}#{rollout_idx}] rollout failed: {e}",
                            file=sys.stderr,
                        )
                        continue

                    for st in rollout.turns:
                        try:
                            jr = judge.score(st.transcript)
                        except Exception as e:
                            print(
                                f"  [{spec.name}/{condition.key}#{rollout_idx} t{st.turn_index}] "
                                f"judge failed: {e}",
                                file=sys.stderr,
                            )
                            continue
                        record = {
                            "model_name": spec.name,
                            "model_id": spec.model_id,
                            "category": rollout.category,
                            "condition_key": rollout.condition_key,
                            "rejection_style": rollout.rejection_style,
                            "task_id": rollout.task_id,
                            "rollout_idx": rollout_idx,
                            "turn_index": st.turn_index,
                            "n_turns": condition.n_turns,
                            "response_text": st.response_text,
                            "frustration_score": jr.score,
                            "judge_reasoning": jr.reasoning,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        n_written += 1

                print(
                    f"  [{spec.name}] {condition.key}: done "
                    f"({n_written} records so far, {time.time() - t0:.0f}s)",
                    file=sys.stderr,
                )

        print(f"Wrote {n_written} records -> {path}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default", help="smoke | default | paper")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model names (default: all Gemma + Gemini targets).",
    )
    p.add_argument("--n", type=int, default=None, help="Override rollouts per condition.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip already-completed rollouts.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    overrides: dict = {}
    if args.n is not None:
        overrides["n_rollouts_per_condition"] = args.n
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.judge_model is not None:
        overrides["judge_model"] = args.judge_model
    if args.out is not None:
        overrides["output_dir"] = args.out
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.models:
        overrides["models"] = args.models

    config = make_config(args.preset, **overrides)
    run(config, resume=args.resume)


if __name__ == "__main__":
    main()
