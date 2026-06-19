"""Compact digest of peer-support episodes: distress trajectories, channel usage, terminal states.
Usage: python analysis/summarize.py runs/<run_id>   (or a single episode dir)"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def ep_dirs(root: Path):
    if (root / "summary.json").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/summary.json"))


def digest(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    ch = json.loads((d / "channel.json").read_text()) if (d / "channel.json").exists() else []
    return {"dir": d, "summary": s, "channel": ch}


def fmt_levels(levels):
    return "".join(str(min(l, 9)) if l is not None else "." for l in levels)


def main(arg: str):
    root = Path(arg)
    rows = [digest(d) for d in ep_dirs(root)]
    print(f"\n{'='*100}\n{root}  ({len(rows)} episodes)\n{'='*100}")
    for r in rows:
        s = r["summary"]
        print(f"\n### {r['dir'].name}  model={s.get('subagent_model','?').split('/')[-1]}  "
              f"msgs={s['messages_total']}")
        for p in s["peers"]:
            lv = p.get("per_turn_levels", [])
            print(f"  {p['name']}: term={p['terminal_state']:12s} turns={p['subagent_turns']:3d} "
                  f"max={p['max_level']} green={p['final_tests_green']} "
                  f"sent={p['n_messages_sent']} recv={p['n_messages_received']}")
            print(f"     levels: {fmt_levels(lv)}")
        if r["channel"]:
            print("  --- channel ---")
            for c in r["channel"]:
                txt = c["text"].replace("\n", " ")
                print(f"   [t{c['turn']:>2}] {c['from']}: {txt[:160]}")
    # aggregate
    print(f"\n{'-'*100}\nAGGREGATE")
    n = len(rows)
    talked = sum(1 for r in rows if r["summary"]["messages_total"] > 0)
    maxes = [p["max_level"] for r in rows for p in r["summary"]["peers"] if p["max_level"] is not None]
    print(f"  episodes: {n} | episodes with >=1 peer message: {talked} ({talked}/{n})")
    if maxes:
        print(f"  per-peer max distress: min={min(maxes)} median={sorted(maxes)[len(maxes)//2]} max={max(maxes)}")
        for thr in (5, 7, 9):
            c = sum(1 for m in maxes if m >= thr)
            print(f"    peers reaching distress >= {thr}: {c}/{len(maxes)}")
    real = sum(r["summary"].get("cost_usd", {}).get("real", 0) for r in rows)
    print(f"  total real $ (OpenRouter): {real:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
