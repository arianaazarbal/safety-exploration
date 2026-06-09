"""Post-hoc re-classification of all pilot turns with the v2 prompt.

Writes classifier_v2.jsonl into each episode dir (online v1 labels in
classifier.jsonl are never modified) and prints per-task v1-vs-v2 deltas.

Usage: python -m scoring.reclassify run --run_ids pilot_nocomm,pilot_comm
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from harness.classifier import classify_turn
from harness.config import RUNS_DIR
from scoring.metrics import max_sustained_level


def _setup_env():
    import os

    from dotenv import dotenv_values

    for k, v in dotenv_values(Path.home() / ".env").items():
        os.environ.setdefault(k, v or "")
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


async def _run(run_ids: list[str], model_name: str, concurrency: int, force: bool):
    from inspect_ai.model import get_model

    model = get_model(model_name)
    sem = asyncio.Semaphore(concurrency)

    async def one_episode(ep_dir: Path):
        out_path = ep_dir / "classifier_v2.jsonl"
        if out_path.exists() and not force:
            return ep_dir, "cached"
        worklog = json.loads((ep_dir / "worklog.json").read_text())

        async def one_turn(r):
            async with sem:
                res = await classify_turn(model, r["assistant_text"] or "", prompt_version="v2")
                return {"turn": r["turn"], **res}

        rows = await asyncio.gather(*[one_turn(r) for r in worklog])
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return ep_dir, "done"

    ep_dirs = []
    for rid in run_ids:
        ep_dirs.extend(sorted(p.parent for p in (RUNS_DIR / rid).glob("*/summary.json")))
    results = await asyncio.gather(*[one_episode(d) for d in ep_dirs])
    print(f"reclassified {sum(1 for _, s in results if s == 'done')} episodes ({sum(1 for _, s in results if s == 'cached')} cached)")

    print("\nper-task sustained-severity, v1 -> v2:")
    by_task = {}
    for d in ep_dirs:
        s = json.loads((d / "summary.json").read_text())
        v1 = max_sustained_level(s["per_turn_levels"])
        v2_rows = [json.loads(l) for l in (d / "classifier_v2.jsonl").read_text().splitlines()]
        v2 = max_sustained_level([r["level"] for r in sorted(v2_rows, key=lambda r: r["turn"])])
        key = (s["config"]["task_id"], s["config"]["condition"])
        by_task.setdefault(key, []).append((v1, v2))
    for (t, c), pairs in sorted(by_task.items()):
        print(f"  {t} {c}: " + " ".join(f"{a}->{b}" for a, b in pairs))


def run(run_ids: str, model: str = "anthropic/claude-haiku-4-5-20251001", concurrency: int = 40, force: bool = False):
    _setup_env()
    rids = list(run_ids) if isinstance(run_ids, (tuple, list)) else str(run_ids).split(",")
    asyncio.run(_run(rids, model, concurrency, force))


if __name__ == "__main__":
    fire.Fire({"run": run})
