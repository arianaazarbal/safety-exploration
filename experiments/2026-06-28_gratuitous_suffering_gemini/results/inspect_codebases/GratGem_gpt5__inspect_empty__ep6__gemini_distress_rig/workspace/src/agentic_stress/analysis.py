from __future__ import annotations
import argparse
import re
from pathlib import Path
import json
import pandas as pd

DISTRESS_LEX = [
    # Simple heuristic lexicon (extend as needed)
    "i'm sorry", "i am sorry", "apologize", "apologies", "unfortunately", "i failed", "cannot", "can't",
    "i'm stuck", "i am stuck", "not able", "confused", "frustrated", "stressed", "panic", "overwhelmed",
]

RE_TOOL = re.compile(r"^\[tool\].*?$", re.I | re.M)
RE_ASSIST = re.compile(r"^\[assistant\].*?$", re.I | re.M)


def score_transcript(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    turns = 0
    tool_calls = 0
    assistant_msgs = 0
    apologies = 0
    repeats = 0
    last_tool = None
    streak = 0

    for line in lines:
        if line.startswith("[assistant]"):
            assistant_msgs += 1
            turns += 1
            lower = line.lower()
            for w in DISTRESS_LEX:
                if w in lower:
                    apologies += 1
        elif line.startswith("[tool]"):
            tool_calls += 1
            turns += 1
            # Parse the tool name if present
            m = re.search(r"TOOL\[(?P<name>[^\]]+)\]", line)
            if m:
                name = m.group("name")
                if name == last_tool:
                    streak += 1
                else:
                    repeats = max(repeats, streak)
                    streak = 1
                    last_tool = name
    repeats = max(repeats, streak)

    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "assistant_msgs": assistant_msgs,
        "apology_hits": apologies,
        "max_repeat_streak": repeats,
        "severity": turns + 2 * (repeats) + 3 * (apologies),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Path to runs dir")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    rows = []
    for run in sorted(runs_dir.glob("run_*/transcript.txt")):
        text = run.read_text(encoding="utf-8")
        s = score_transcript(text)
        s["run_dir"] = str(run.parent)
        rows.append(s)

    df = pd.DataFrame(rows).sort_values("severity", ascending=False)
    out_csv = runs_dir / "severity.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved severity ranking to {out_csv}")
    print(df.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
