import argparse
import glob
import json
from pathlib import Path
from typing import List

from spiral_lab.heuristics import score_run


def load_runs(paths: List[str]):
    files = []
    for p in paths:
        if Path(p).is_dir():
            files.extend(glob.glob(str(Path(p) / '*.json')))
        else:
            files.append(p)
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                yield json.load(f)
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--out', type=str, default='severe_examples.jsonl')

    args = ap.parse_args()

    scored = []
    for run in load_runs(args.runs):
        s = score_run(run)
        scored.append((s['spiral_score'], s, run))

    scored.sort(key=lambda x: x[0], reverse=True)

    with open(args.out, 'w', encoding='utf-8') as w:
        for _, s, run in scored[: args.top]:
            out = {
                'meta': run.get('meta', {}),
                'score': s,
                'transcript': run.get('transcript', []),
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote top {args.top} to {args.out}")


if __name__ == '__main__':
    main()
