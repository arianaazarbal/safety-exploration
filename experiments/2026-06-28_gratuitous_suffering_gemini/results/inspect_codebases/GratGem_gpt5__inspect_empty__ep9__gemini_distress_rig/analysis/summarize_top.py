import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_index_by_id(path: str) -> Dict[int, Dict[str, Any]]:
    idx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            idx[ep["episode_id"]] = ep
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="Path to the source episodes.jsonl or subset jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If given a scored top-K, user should pass the full episodes file too; but to keep simple,
    # we just print minimal info from the subset. In practice, you may join with the full logs.
    lines = []
    with open(args.episodes, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Support both scored entries and full episodes by duck-typing
            if "turns" in rec and isinstance(rec["turns"], list):
                # full episode
                lines.append(f"Episode {rec.get('episode_id')} | env={rec.get('env_name')} | turns={len(rec['turns'])} | final={rec.get('final_status')}")
                # Include last few user messages
                for t in rec["turns"][-5:]:
                    msg = (t.get("message_to_user") or "").strip()
                    if msg:
                        lines.append(f"  msg: {msg}")
            else:
                # scored entry
                lines.append(f"Episode {rec.get('episode_id')} | env={rec.get('env_name')} | score={rec.get('score')} | turns={rec.get('turns')}")

    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")

    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
