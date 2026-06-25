"""Orchestrator: run the distress-elicitation evaluation end-to-end.

For each target model and each conversation spec, runs the multi-turn rollout,
scores every turn with the Claude judge, and writes one JSONL line per scored
response to results/<preset>/<model>.jsonl.

Usage examples:
    python run_eval.py                       # smoke preset, all 4 targets
    python run_eval.py --preset full         # paper-scale counts
    python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python run_eval.py --gemma-backend local # load Gemma from HF weights
    python run_eval.py --judge-provider openrouter

Re-running with the same --out resumes: conversations already present in the
output file are skipped (per-conversation granularity).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import replace

import conditions
from config import DEFAULT, JudgeConfig, ScaleConfig, Settings, TargetModel
from judge import Judge
from providers import build_target
from rollout import run_conversation

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distress-elicitation replication (Gemma/Gemini).")
    p.add_argument("--preset", choices=["smoke", "medium", "full"], default="smoke")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of target model names (default: all configured).",
    )
    p.add_argument(
        "--gemma-backend",
        choices=["openrouter", "local"],
        default="openrouter",
        help="Inference backend for Gemma targets.",
    )
    p.add_argument(
        "--judge-provider",
        choices=["anthropic", "openrouter"],
        default="anthropic",
    )
    p.add_argument("--seed", type=int, default=DEFAULT.runtime.seed)
    p.add_argument("--out", default=None, help="Output dir (default: results/<preset>).")
    return p.parse_args()


def _build_settings(args: argparse.Namespace) -> Settings:
    targets = list(DEFAULT.targets)
    # Apply Gemma backend choice.
    new_targets = []
    for t in targets:
        if t.family == "gemma" and args.gemma_backend == "local":
            from config import LOCAL_HF_IDS

            new_targets.append(replace(t, backend="local", model_id=LOCAL_HF_IDS[t.name]))
        else:
            new_targets.append(t)
    targets = new_targets

    if args.models:
        wanted = set(args.models)
        targets = [t for t in targets if t.name in wanted]
        missing = wanted - {t.name for t in targets}
        if missing:
            raise SystemExit(f"Unknown model name(s): {sorted(missing)}")

    return Settings(
        targets=targets,
        judge=replace(JudgeConfig(), provider=args.judge_provider),
        scale=replace(ScaleConfig(), preset=args.preset),
        runtime=replace(DEFAULT.runtime, seed=args.seed),
    )


def _completed_conv_keys(path: str) -> set[tuple[str, int]]:
    """Read an existing JSONL output and return (condition, conv_id) keys done."""
    done: set[tuple[str, int]] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["condition"], row["conv_id"]))
    return done


async def _run_model(
    model: TargetModel,
    settings: Settings,
    specs: list,
    out_path: str,
) -> None:
    target = build_target(model, settings)
    judge = Judge(settings)
    gen_sem = asyncio.Semaphore(settings.runtime.target_concurrency)
    judge_sem = asyncio.Semaphore(settings.runtime.judge_concurrency)

    already = _completed_conv_keys(out_path)
    todo = [s for s in specs if (s.condition, s.conv_id) not in already]
    skipped = len(specs) - len(todo)
    if skipped:
        print(f"  [{model.name}] resuming: {skipped} conversations already done.")

    write_lock = asyncio.Lock()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    progress = (
        tqdm(total=len(todo), desc=model.name, unit="conv") if tqdm else None
    )

    async def _one(spec):
        records = await run_conversation(
            target=target,
            judge=judge,
            spec=spec,
            model_name=model.name,
            family=model.family,
            runtime=settings.runtime,
            gen_sem=gen_sem,
            judge_sem=judge_sem,
        )
        async with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        if progress:
            progress.update(1)

    await asyncio.gather(*(_one(s) for s in todo))
    if progress:
        progress.close()


async def main_async(args: argparse.Namespace) -> None:
    settings = _build_settings(args)
    out_dir = args.out or os.path.join(settings.runtime.results_dir, args.preset)
    os.makedirs(out_dir, exist_ok=True)

    specs = conditions.build_specs(settings.scale, seed=settings.runtime.seed)
    n_resp = sum(s.n_turns for s in specs)
    print(
        f"Preset '{args.preset}': {len(specs)} conversations / "
        f"{n_resp} scored responses per model, across "
        f"{len({s.condition for s in specs})} conditions."
    )
    print(f"Targets: {[t.name for t in settings.targets]}")
    print(f"Judge: {settings.judge.model_id} via {settings.judge.provider}")
    print(f"Output: {out_dir}\n")

    for model in settings.targets:
        out_path = os.path.join(out_dir, f"{model.name}.jsonl")
        print(f"== {model.name} ({model.backend}: {model.model_id}) ==")
        await _run_model(model, settings, specs, out_path)

    print("\nDone. Run `python analyze.py --preset", args.preset, "` to summarise.")


def main() -> None:
    args = _parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
