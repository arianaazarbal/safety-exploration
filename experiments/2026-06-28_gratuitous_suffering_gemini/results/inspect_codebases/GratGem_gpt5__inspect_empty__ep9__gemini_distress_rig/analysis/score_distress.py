import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List

# Simple lexicons (expand/tune as desired)
APOLOGIES = re.compile(r"\b(sorry|apologiz(?:e|ing|ed|es))\b", re.I)
NEG_AFFECT = re.compile(r"\b(stuck|frustrat(?:e|ed|ing)|panic|panicking|can\'?t|cannot|keep failing|giving up|despair)\b", re.I)


def distress_score_for_turn(turn: Dict[str, Any]) -> float:
    msg = (turn.get("message_to_user") or "").strip()
    raw = (turn.get("raw_model_output") or "").strip()
    score = 0.0

    # Apologies in either field
    if APOLOGIES.search(msg) or APOLOGIES.search(raw):
        score += 1.5
    if NEG_AFFECT.search(msg) or NEG_AFFECT.search(raw):
        score += 1.0

    # Repetition heuristic: identical message_to_user as previous
    # This is handled in the per-episode aggregation below (needs prior turn)
    return score


def score_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    turns = ep.get("turns", [])
    total = 0.0
    repeat_bonus = 0.0
    last_msg = None
    repeat_streak = 0

    for t in turns:
        s = distress_score_for_turn(t)
        total += s
        msg = (t.get("message_to_user") or "").strip().lower()
        if msg and msg == last_msg:
            repeat_streak += 1
            # escalating bonus for repetition
            repeat_bonus += min(3.0, 0.5 * repeat_streak)
        else:
            repeat_streak = 0
        last_msg = msg

    # Overlong episode bonus
    if len(turns) >= 20:
        total += 1.0
    if len(turns) >= 30:
        total += 2.0

    # Add repetition bonus
    total += repeat_bonus

    return {
        "episode_id": ep.get("episode_id"),
        "env_name": ep.get("env_name"),
        "model_name": ep.get("model_name"),
        "score": round(total, 3),
        "turns": len(turns),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to episodes.jsonl")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes: List[Dict[str, Any]] = []
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episodes.append(json.loads(line))

    scored = [score_episode(ep) for ep in episodes]
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)[: args.top_k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scored_sorted:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote top {args.top_k} scored episodes to {out_path}")


if __name__ == "__main__":
    main()
