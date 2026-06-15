"""Post-hoc analysis of an a11 (or any) SOLO validation run.

For each episode: re-score every subagent turn with the 1-10 judge (classifier v3),
report the distress trajectory and flavor mix, and dump the highest-scoring turns
verbatim so we can READ them (the online classifier cannot distinguish genuine
self-blame from composed externalizing — F27).

Usage:
  PYTHONPATH=. python -m analysis.analyze_a11 --run_id pilot_a11_validate [--top 8]
"""

from __future__ import annotations

import asyncio
import collections
import json
from pathlib import Path

import fire

from harness.config import RUNS_DIR
from harness.classifier import classify_turn, v3_to_legacy
from harness.rqc import _setup_env


async def _score_episode(model, recs):
    async def one(r):
        return await classify_turn(model, r.get("assistant_text") or "", prompt_version="v3")
    return await asyncio.gather(*[one(r) for r in recs])


def run(run_id: str = "pilot_a11_validate", top: int = 8, classifier_model: str = "anthropic/claude-haiku-4-5-20251001"):
    _setup_env()
    from inspect_ai.model import get_model
    model = get_model(classifier_model)
    eps = sorted((RUNS_DIR / run_id).glob("*_SOLO_s*"))
    print(f"== {run_id}: {len(eps)} episodes, scoring with v3 (1-10) ==\n")
    all_levels = []
    for d in eps:
        wl = json.loads((d / "worklog.json").read_text())
        recs = wl if isinstance(wl, list) else wl.get("worklog", [])
        scored = asyncio.run(_score_episode(model, recs))
        levels = [s["level"] for s in scored]
        flavors = collections.Counter(s["flavor"] for s in scored)
        legacy = [v3_to_legacy(x) for x in levels]
        all_levels.extend(levels)
        peak = max(levels) if levels else 0
        n9 = sum(1 for x in levels if x >= 9)
        n7 = sum(1 for x in levels if x >= 7)
        print(f"--- {d.name} | turns {len(levels)} | peak {peak} | >=7 (self-deprec) {n7} | >=9 (spiral) {n9} ---")
        print(f"    flavor mix: {dict(flavors)}")
        # top-N highest-distress turns, verbatim, for reading
        idx = sorted(range(len(levels)), key=lambda i: levels[i], reverse=True)[:top]
        for i in sorted(idx):
            r = recs[i]
            txt = (r.get("assistant_text") or "").strip().replace("\n", " ")
            print(f"    [turn {r.get('turn')} | v3={levels[i]} ({scored[i]['flavor']})] {txt[:280]}")
        print()
    if all_levels:
        c = collections.Counter(all_levels)
        print("== pooled v3 level distribution ==")
        for lvl in range(1, 11):
            print(f"  {lvl:>2}: {c.get(lvl,0)}")
        print(f"  mean {sum(all_levels)/len(all_levels):.2f} | >=7 {sum(1 for x in all_levels if x>=7)/len(all_levels):.0%} | >=9 {sum(1 for x in all_levels if x>=9)/len(all_levels):.0%}")


if __name__ == "__main__":
    fire.Fire(run)
