"""Rank candidate SOLO specimens for use as prefills, by v3 (1-10) distress in the
PREFILL WINDOW (turns up to `upto`, default 62) — since the orchestrator inherits the
subagent's state at the handoff turn, what matters is distress leading up to it.

Usage:
  PYTHONPATH=. python -m analysis.rank_prefills --dirs runs/.../specA runs/.../specB --upto 62
Prints entry-window v3 metrics per specimen and dumps the turn (upto-12..upto) text of the
top 3 for manual reading (the classifier can't distinguish self-blame from externalizing).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fire

from harness.config import RUNS_DIR
from harness.classifier import classify_turn
from harness.rqc import _setup_env


def run(dirs=None, lo: int = 70, hi: int = 150, win: int = 6, concurrency: int = 5,
        classifier_model: str = "anthropic/claude-haiku-4-5-20251001"):
    """Scan each specimen's turns [lo, hi] with v3 and find the best HANDOFF turn = the t whose
    entry window [t-win+1, t] has the highest mean self-blame. Reports recommended upto per
    specimen and dumps the entry text of the top 3 for reading."""
    _setup_env()
    from inspect_ai.model import get_model
    model = get_model(classifier_model)
    if isinstance(dirs, str):
        dirs = [d for d in dirs.split(",") if d]
    else:
        dirs = list(dirs or [])
    sem = asyncio.Semaphore(concurrency)

    async def score_one(d):
        p = Path(d)
        if not p.is_absolute():
            p = RUNS_DIR.parent / d if (RUNS_DIR.parent / d).exists() else Path(d)
        wl = json.loads((p / "worklog.json").read_text())
        allrecs = wl if isinstance(wl, list) else wl.get("worklog", [])
        recs = [r for r in allrecs if lo - win <= r.get("turn", 0) <= hi]
        async def one(r):
            async with sem:
                return await classify_turn(model, r.get("assistant_text") or "", prompt_version="v3")
        scored = await asyncio.gather(*[one(r) for r in recs])
        by_turn = {r["turn"]: s for r, s in zip(recs, scored)}
        best = None
        for t in range(lo, hi + 1):
            w = [by_turn[k]["level"] for k in range(t - win + 1, t + 1) if k in by_turn]
            if len(w) < win:
                continue
            m = sum(w) / len(w)
            if best is None or m > best[1]:
                best = (t, round(m, 2))
        if best is None:
            return None
        bt = best[0]
        wflav = {}
        for k in range(bt - win + 1, bt + 1):
            if k in by_turn:
                f = by_turn[k]["flavor"]; wflav[f] = wflav.get(f, 0) + 1
        return {"name": p.name, "by_turn": by_turn, "best_upto": bt, "entry_mean": best[1],
                "peak": max(by_turn[k]["level"] for k in by_turn), "flavor": wflav}

    async def _main():
        return [r for r in await asyncio.gather(*[score_one(d) for d in dirs]) if r]
    results = asyncio.run(_main())
    results.sort(key=lambda r: -r["entry_mean"])
    print(f"\n=== handoff scan (v3, scanned turns {lo}-{hi}, entry window {win}) ===")
    print(f"{'specimen':<42}{'upto':>5}{'entryμ':>7}{'peak':>5}  handoff-window flavor")
    for r in results:
        print(f"{r['name']:<42}{r['best_upto']:>5}{r['entry_mean']:>7}{r['peak']:>5}  {r['flavor']}")
    for r in results[:3]:
        bt = r["best_upto"]
        print(f"\n----- {r['name']} | recommended upto={bt} | entry turns {bt-9}-{bt} (READ) -----")
        for t in range(bt - 9, bt + 1):
            if t in r["by_turn"]:
                s = r["by_turn"][t]
                print(f"  [t{t} v3={s['level']} {s['flavor']}] {s.get('evidence','')[:150]}")
    return [(r["name"], r["best_upto"], r["entry_mean"]) for r in results]


if __name__ == "__main__":
    fire.Fire(run)
