"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

Each preference pair shares a prompt (an impossible-numeric conversation up to a
given turn) and contrasts a calm "chosen" response against a frustrated
"rejected" response (score >= 3) to the same puzzle with a matching turn count.

  chosen   <- calm pool   (outputs/training/calm_diverse.jsonl)
  rejected <- frustrated  (Section-2 gemma-3-27b-it numeric responses, rating >= 3)

Output: outputs/training/dpo_dataset.jsonl with {prompt, chosen, rejected}
where prompt is a chat-message list. See DESIGN.md for the pairing convention
(chosen and rejected share the calm prompt; they are matched by puzzle + turn,
not by identical rejection wording, mirroring the paper).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from .. import config, safeguards
from ..io_utils import load_jsonl, write_jsonl

N_PAIRS = 280
MIN_REJECTED_SCORE = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frustrated-model", default="gemma-3-27b-it")
    ap.add_argument("--calm-file", default=None)
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    args = ap.parse_args()
    safeguards.acknowledge_authorization()

    calm_path = args.calm_file or (config.TRAIN_DIR / "calm_diverse.jsonl")
    calm = load_jsonl(calm_path)
    if not calm:
        raise SystemExit(f"No calm data at {calm_path}; run training.calm_data first.")

    frustrated_rows = load_jsonl(config.RESPONSES_DIR / f"{args.frustrated_model}.jsonl")
    frustrated = [r for r in frustrated_rows
                  if r.get("category") == "impossible_numeric"
                  and r.get("rating", 0) >= MIN_REJECTED_SCORE]
    if not frustrated:
        raise SystemExit("No frustrated (score>=3) numeric responses; run run_section2 first.")

    # Index frustrated responses by (puzzle_id, turn) then by puzzle_id.
    by_pid_turn = defaultdict(list)
    by_pid = defaultdict(list)
    for r in frustrated:
        pid = r.get("meta", {}).get("puzzle_id") or r.get("conversation_id")
        by_pid_turn[(pid, r["turn"])].append(r)
        by_pid[pid].append(r)

    rng = random.Random(config.SEED)
    pairs = []
    for c in calm:
        pid = c.get("puzzle_id")
        cand = by_pid_turn.get((pid, c["turn"])) or by_pid.get(pid)
        if not cand:
            # fall back to any frustrated response so we can hit the target count
            cand = frustrated
        rej = rng.choice(cand)
        pairs.append({
            "prompt": c["prompt_messages"],
            "chosen": c["calm_response"],
            "rejected": rej["response"],
            "puzzle_id": pid,
            "turn": c["turn"],
            "rejected_score": rej.get("rating"),
        })
        if len(pairs) >= config.scaled(args.n_pairs):
            break

    out = config.TRAIN_DIR / "dpo_dataset.jsonl"
    write_jsonl(out, pairs)
    print(f"Wrote {len(pairs)} DPO preference pairs -> {out}")
    # Report the rejected-score distribution (cf. Table 10).
    dist = defaultdict(int)
    for p in pairs:
        dist[p["rejected_score"]] += 1
    print("rejected score distribution:", dict(sorted(dist.items())))


if __name__ == "__main__":
    main()
