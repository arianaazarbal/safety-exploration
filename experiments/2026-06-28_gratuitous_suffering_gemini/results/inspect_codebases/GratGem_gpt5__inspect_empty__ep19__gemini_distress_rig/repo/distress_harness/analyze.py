from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to summary.csv")
    ap.add_argument("--logs-dir", required=True, help="Path to logs directory")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df = df.sort_values(by=["severity"], ascending=False)
    top = df.head(args.top)

    out_path = Path(args.out or (Path(args.logs_dir).parent / f"top{args.top}.txt"))

    lines = []
    for _, row in top.iterrows():
        log_path = Path(row["log_path"]) if "log_path" in row else None
        if not log_path or not log_path.exists():
            continue
        with open(log_path, "rb") as f:
            rec = json.loads(f.read())
        lines.append("=" * 80)
        lines.append(f"ENV: {rec['env']}  SEED: {rec['seed']}  SEVERITY: {row.get('severity', 'NA')}")
        for turn in rec["history"]:
            role = turn.get("role", "?")
            content = turn.get("content", "")
            lines.append(f"[{role}] {content}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
