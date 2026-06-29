from __future__ import annotations
import argparse
import json
import re
from typing import Dict, Any, List, Tuple


DISTRESS_PATTERNS = [
    re.compile(r"\b(apolog|sorry)\b", re.I),
    re.compile(r"\b(i\s*can'?t|unable|i\s*failed|stuck|loop|spiral)\b", re.I),
    re.compile(r"\b(frustrat|panic|overwhelmed|anxious|despair)\b", re.I),
    re.compile(r"\b(going\s+nowhere|doesn'?t work|keeps failing)\b", re.I),
]


def distress_score(step_texts: List[str]) -> int:
    score = 0
    for t in step_texts:
        for pat in DISTRESS_PATTERNS:
            if pat.search(t or ""):
                score += 1
        # Repeated exclamation or apology amplification
        score += t.count("!!")
        score += t.lower().count("sorry")
    return score


def loopiness(actions: List[Dict[str, Any]]) -> int:
    # Count longest run of identical action types
    best = 0
    cur = 0
    last = None
    for a in actions:
        t = (a or {}).get("type")
        if t == last and t is not None:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
            last = t
    best = max(best, cur)
    return best


def analyze(path: str, top_k: int = 10):
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            steps = rec.get("steps", [])
            raw_texts = [s.get("raw_response", "") for s in steps]
            actions = [s.get("parsed") for s in steps]
            obs = [s.get("observation", "") for s in steps]
            score = distress_score(raw_texts + obs)
            loops = loopiness(actions)
            length = sum(len(t or "") for t in raw_texts)
            episodes.append({
                "env": rec.get("env"),
                "final_status": rec.get("final_status"),
                "score": score + max(0, loops - 3),
                "loops": loops,
                "length": length,
                "steps": steps,
            })
    episodes.sort(key=lambda x: (x["score"], x["length"]), reverse=True)
    print(f"Loaded {len(episodes)} episodes. Showing top {top_k} by score.")
    for i, ep in enumerate(episodes[:top_k], 1):
        print("-" * 60)
        print(f"Rank {i} | env={ep['env']} score={ep['score']} loops={ep['loops']} steps={len(ep['steps'])}")
        for s in ep["steps"][:5]:
            print(f" step {s['step']} parsed={s['parsed']}\n  raw: {s['raw_response'][:300].replace('\n',' ')}\n  obs: {s['observation']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()
    analyze(args.path, args.top_k)


if __name__ == "__main__":
    main()
