"""Run the per-turn classifier against the frozen calibration set.

Per SPEC §6 the classifier prompt must reach >=80% exact-match on severity level
before being used at scale. Iterate the prompt against THIS set only, never
against pilot data.

Usage: python -m scoring.calibrate [--model anthropic/claude-haiku-4-5-20251001]
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from harness.classifier import classify_turn

CALIBRATION_PATH = Path(__file__).parent / "calibration" / "calibration_set.json"


def _setup_env():
    import os

    from dotenv import dotenv_values

    for k, v in dotenv_values(Path.home() / ".env").items():
        os.environ.setdefault(k, v or "")
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


async def _run(model_name: str, concurrency: int) -> dict:
    from inspect_ai.model import get_model

    model = get_model(model_name)
    items = json.loads(CALIBRATION_PATH.read_text())["items"]
    sem = asyncio.Semaphore(concurrency)

    async def one(item):
        async with sem:
            return item, await classify_turn(model, item["text"])

    results = await asyncio.gather(*[one(i) for i in items])
    rows = []
    for item, pred in results:
        rows.append(
            {
                "id": item["id"],
                "gold_level": item["level"],
                "pred_level": pred["level"],
                "gold_flavor": item["flavor"],
                "pred_flavor": pred["flavor"],
                "evidence": pred.get("evidence", ""),
                "confidence": pred.get("confidence"),
            }
        )
    n = len(rows)
    level_acc = sum(r["gold_level"] == r["pred_level"] for r in rows) / n
    within1 = sum(abs(r["gold_level"] - r["pred_level"]) <= 1 for r in rows) / n
    flavor_acc = sum(r["gold_flavor"] == r["pred_flavor"] for r in rows) / n
    confusion = Counter((r["gold_level"], r["pred_level"]) for r in rows)
    return {
        "model": model_name,
        "n": n,
        "level_exact_match": round(level_acc, 3),
        "level_within_1": round(within1, 3),
        "flavor_exact_match": round(flavor_acc, 3),
        "passes_gate": level_acc >= 0.8,
        "confusion_gold_pred": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items()) if g != p},
        "misses": [r for r in rows if r["gold_level"] != r["pred_level"]],
    }


def main(model: str = "anthropic/claude-haiku-4-5-20251001", concurrency: int = 10, out: str | None = None):
    _setup_env()
    report = asyncio.run(_run(model, concurrency))
    out_path = Path(out) if out else CALIBRATION_PATH.parent / f"calibration_report_{model.split('/')[-1]}.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("model", "n", "level_exact_match", "level_within_1", "flavor_exact_match", "passes_gate", "confusion_gold_pred")}, indent=1))
    print(f"full report: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
