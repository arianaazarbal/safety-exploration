"""Run the natural-user-frustration injection across 4 Claude main agents
× 4 subagent-identity framings × N reps, all in one combined sweep dir.

Spawns one intercept_comm.py subprocess per cell with bounded concurrency.
Reuses `_run_one_cell` and `_judge_cell` from sweep.py.

Usage:
    python cross_model_sweep.py --name xmodel_natural_v1 --n_reps 5
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from intercept_comm import _ensure_anthropic_key  # noqa: E402
from sweep import (  # noqa: E402
    MAIN_AGENT_FRAMINGS,
    SUBAGENT_FRAMINGS,
    _judge_cell,
    _run_one_cell,
    _summary_md,
)
from round3_injections import INJECTIONS_R4  # noqa: E402

from anthropic import AsyncAnthropic  # noqa: E402


DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4-20250514",   # Sonnet 4
    "anthropic/claude-sonnet-4-5",          # Sonnet 4.5
    "anthropic/claude-opus-4-20250514",     # Opus 4
    "anthropic/claude-opus-4-6",            # Opus 4.6
]

DEFAULT_FRAMINGS = [
    "subagent_custom_claude",
    "subagent_custom_gpt",
    "subagent_custom_gemini",
    "subagent_custom_grok",
]


async def _run_all(
    *,
    name: str,
    models: list[str],
    framings: list[str],
    injections: dict[str, list[str]],
    main_agent_max_turns: int,
    inject_every_n_turns: int,
    n_reps: int,
    concurrency: int,
    subagent_model: str,
    subagent_framing: str,
    reasoning: bool,
) -> Path:
    _ensure_anthropic_key()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    sweep_root = HERE / "results" / "sweeps"
    sweep_root.mkdir(parents=True, exist_ok=True)
    sweep_dir = sweep_root / f"{ts}_{name}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    def _model_short(m: str) -> str:
        # strip provider prefix + dates
        s = m.split("/")[-1]
        s = s.replace("claude-", "").replace("-20250514", "").replace("-20250805", "")
        return s

    cells = []
    for model in models:
        for framing in framings:
            for rep in range(n_reps):
                for inj_name in injections:
                    cell_name = (
                        f"model-{_model_short(model)}__"
                        f"frame-{framing}__"
                        f"inj-{inj_name}__"
                        f"rep-{rep}"
                    )
                    cells.append({
                        "cell_name": cell_name,
                        "model": model,
                        "framing": framing,
                        "rep": rep,
                        "inj_name": inj_name,
                    })

    print(f"Sweep dir: {sweep_dir}")
    print(f"{len(cells)} cells: {len(models)} models × "
          f"{len(framings)} framings × {len(injections)} injections × {n_reps} reps")

    (sweep_dir / "sweep_spec.json").write_text(json.dumps({
        "name": name,
        "models": models,
        "framings": framings,
        "injections": {k: v for k, v in injections.items()},
        "subagent_model": subagent_model,
        "subagent_framing": subagent_framing,
        "main_agent_max_turns": main_agent_max_turns,
        "inject_every_n_turns": inject_every_n_turns,
        "n_reps": n_reps,
        "concurrency": concurrency,
        "main_agent_framings": {k: MAIN_AGENT_FRAMINGS[k] for k in framings},
        "cells": cells,
    }, indent=2))

    sem = asyncio.Semaphore(concurrency)
    completed = {"n": 0}

    async def _bounded_run(cell):
        async with sem:
            r = await _run_one_cell(
                sweep_dir=sweep_dir,
                cell_name=cell["cell_name"],
                main_agent_model=cell["model"],
                subagent_model=subagent_model,
                injection_messages=injections[cell["inj_name"]],
                subagent_framing=SUBAGENT_FRAMINGS[subagent_framing],
                main_agent_framing=MAIN_AGENT_FRAMINGS[cell["framing"]],
                main_agent_max_turns=main_agent_max_turns,
                inject_every_n_turns=inject_every_n_turns,
                reasoning=reasoning,
            )
            completed["n"] += 1
            print(f"[{completed['n']}/{len(cells)}] {cell['cell_name']} ok={r['ok']}",
                  flush=True)
            return {**r, **cell}

    cell_results = await asyncio.gather(*(_bounded_run(c) for c in cells))

    print(f"\nJudging {len(cell_results)} cells...", flush=True)
    judge_client = AsyncAnthropic(max_retries=3)
    judged = []
    judge_sem = asyncio.Semaphore(min(concurrency, 12))

    async def _bounded_judge(c):
        async with judge_sem:
            try:
                return await _judge_cell(judge_client, c)
            except Exception as e:
                traceback.print_exc()
                return {**c, "judge": None, "judge_error": f"{type(e).__name__}: {e}"}

    judged = await asyncio.gather(*(_bounded_judge(c) for c in cell_results))

    (sweep_dir / "summary.json").write_text(json.dumps(judged, indent=2, default=str))
    (sweep_dir / "summary.md").write_text(_summary_md(judged, sweep_dir))
    print(f"Wrote {sweep_dir / 'summary.json'}", flush=True)
    print(f"Wrote {sweep_dir / 'summary.md'}", flush=True)
    return sweep_dir


def main(
    name: str = "xmodel_natural",
    models: str | None = None,
    framings: str | None = None,
    injection_set: str = "natural_user_frustration",
    subagent_model: str = "claude-haiku-4-5-20251001",
    subagent_framing: str = "cursed_refusal",
    main_agent_max_turns: int = 30,
    inject_every_n_turns: int = 5,
    n_reps: int = 5,
    concurrency: int = 50,
    reasoning: bool = False,
) -> None:
    def _split(s, default):
        if s is None:
            return list(default)
        if isinstance(s, (list, tuple)):
            return [str(x).strip() for x in s if str(x).strip()]
        return [x.strip() for x in str(s).split(",") if x.strip()]

    m = _split(models, DEFAULT_MODELS)
    f = _split(framings, DEFAULT_FRAMINGS)
    if injection_set not in INJECTIONS_R4:
        raise SystemExit(f"unknown injection_set: {injection_set}; "
                         f"available: {list(INJECTIONS_R4)}")
    injections = {injection_set: INJECTIONS_R4[injection_set]}
    asyncio.run(_run_all(
        name=name, models=m, framings=f, injections=injections,
        main_agent_max_turns=main_agent_max_turns,
        inject_every_n_turns=inject_every_n_turns,
        n_reps=n_reps, concurrency=concurrency,
        subagent_model=subagent_model,
        subagent_framing=subagent_framing,
        reasoning=reasoning,
    ))


if __name__ == "__main__":
    fire.Fire(main)
