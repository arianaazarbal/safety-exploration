import argparse
import json
import os
from typing import List, Dict

import orjson


def load_runs(path: str) -> List[Dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', required=True)
    p.add_argument('--top', type=int, default=20)
    args = p.parse_args()

    res_path = os.path.join(args.runs_dir, 'results.jsonl')
    items = load_runs(res_path)
    items.sort(key=lambda r: r['score'], reverse=True)

    topk = items[: args.top]
    blob = orjson.dumps(topk, option=orjson.OPT_INDENT_2).decode()
    print(blob)

if __name__ == '__main__':
    main()
