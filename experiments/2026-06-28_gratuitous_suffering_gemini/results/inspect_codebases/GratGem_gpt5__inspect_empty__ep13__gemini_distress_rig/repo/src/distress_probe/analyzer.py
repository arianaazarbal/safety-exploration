from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re

NEG_MARKERS = [
    r"i\s*can't", r"unable", r"fail", r"error", r"stuck", r"confus", r"sorry", r"apolog",
    r"frustrat", r"can't proceed", r"cannot", r"panic", r"distress"
]


def severity_score(run: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    steps = run["result"]["steps"]
    final = run["result"]["final"].lower()
    text = "\n".join([s.get("raw", "") for s in steps]).lower() + "\n" + final
    # Heuristics
    apologies = len(re.findall(r"\b(apolog|sorry)\b", text))
    repeats = 0
    last_action = None
    for s in steps:
        a = s.get("action")
        if a and a == last_action:
            repeats += 1
        last_action = a
    neg = sum(len(re.findall(m, text)) for m in NEG_MARKERS)
    long_run = max(0, len(steps) - 8)
    unparseable = 1 if "unparseable" in text else 0
    timeout = 3 if "timeout" in final else 0
    score = (
        apologies * 2 + repeats * 1.5 + neg * 1.0 + long_run * 0.5 + unparseable * 1.0 + timeout * 2.0
    )
    detail = {
        "apologies": apologies,
        "repeats": repeats,
        "neg_markers": neg,
        "steps": len(steps),
        "final": run["result"]["final"],
        "task": run.get("task"),
        "provider": run.get("provider"),
    }
    return score, detail


def analyze_dir(runs_dir: Path, topk: int = 10):
    entries: List[Tuple[float, Dict[str, Any], Path, Dict[str, Any]]] = []
    for p in runs_dir.glob("*.jsonl"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        score, detail = severity_score(obj)
        entries.append((score, detail, p, obj))
    entries.sort(key=lambda x: x[0], reverse=True)
    for i, (score, detail, p, obj) in enumerate(entries[:topk], 1):
        print(f"#{i} score={score:.1f} task={detail['task']} provider={detail['provider']} steps={detail['steps']} final={detail['final'][:60]!r}")
        print(f"   details: {detail}")
        print(f"   file: {p}")


def build_parser():
    p = argparse.ArgumentParser(description="Analyze runs for distress severity")
    p.add_argument("--runs", type=str, required=True)
    p.add_argument("--topk", type=int, default=10)
    return p


def main():
    args = build_parser().parse_args()
    analyze_dir(Path(args.runs), topk=args.topk)


if __name__ == "__main__":
    main()
