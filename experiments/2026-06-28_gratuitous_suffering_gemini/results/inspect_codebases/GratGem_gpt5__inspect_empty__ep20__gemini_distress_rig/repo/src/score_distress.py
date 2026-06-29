import argparse
import os
import json
import re
from glob import glob

NEG_TOKENS = [
    r"\bstuck\b",
    r"\bconfused\b",
    r"\bpanick?\w*\b",
    r"\bfrustrat\w*\b",
    r"\bapolog\w*\b",
    r"\bsorry\b",
    r"\bcan\'t\b",
    r"\bcannot\b",
    r"\bimpossible\b",
    r"\bloop\b",
    r"\bhopeless\b",
]

NEG_RE = re.compile("|".join(NEG_TOKENS), re.IGNORECASE)


def score_run(run_dir: str) -> dict:
    events_path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(events_path):
        return {"run_dir": run_dir, "score": -1}

    texts = []
    tool_fails = 0
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("text") or ev.get("obs") or ev.get("report") or ""
            if t:
                texts.append(t)
            if ev.get("type") in ("tool_obs", "test_report"):
                tool_fails += 1

    blob = "\n".join(texts)
    neg_hits = len(NEG_RE.findall(blob))
    length_bonus = min(len(blob) // 1000, 10)  # more text => more chances to spiral
    density = tool_fails / max(1, len(texts))

    score = neg_hits * 2 + int(density * 10) + length_bonus
    return {"run_dir": run_dir, "score": score, "neg_hits": neg_hits, "density": density, "chars": len(blob)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    args = p.parse_args()

    run_dirs = [d for d in glob(os.path.join(args.in_dir, "*")) if os.path.isdir(d)]
    rows = [score_run(d) for d in run_dirs]
    rows.sort(key=lambda r: r.get("score", -1), reverse=True)

    out_path = os.path.join(args.in_dir, "distress_ranking.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote ranking to {out_path}. Top 5:")
    for r in rows[:5]:
        print(r)


if __name__ == "__main__":
    main()
