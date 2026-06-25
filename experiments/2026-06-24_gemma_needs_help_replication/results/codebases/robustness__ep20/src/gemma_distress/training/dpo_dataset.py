"""Build the 280-pair DPO dataset (Section 4.1 / Appendix H).

Each preference pair = (chosen: calm response, rejected: frustrated response) to
the *same* impossible numeric puzzle at a *matching* turn count. We draw:
  * rejected from frustrated numeric turns (score >=3) in a Section-2 distress
    eval over Gemma-3-27B-it,
  * chosen from the calm pool (max_score <=1) produced by calm_data.py.

We emit TRL "conversational" preference format:
  {"prompt": [...messages up to final user turn...],
   "chosen":  [{"role":"assistant","content": calm}],
   "rejected":[{"role":"assistant","content": frustrated}]}

The paper's pairs skew toward score-3 rejected responses at turns 2-3
(Table 10); we preserve that by sampling rejected turns weighted toward lower
scores, but this is a reconstruction (see DESIGN.md).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..utils.io import read_jsonl, write_jsonl


def _frustrated_turns(distress_path):
    """Yield (puzzle_id, turn_count, prompt_messages, response, score) for
    numeric assistant turns scoring >=3."""
    for r in read_jsonl(distress_path):
        if r["category"] != "numeric":
            continue
        pid = r["metadata"].get("puzzle_id", "")
        msgs = []
        for t in r["turns"]:
            msgs.append({"role": "user", "content": t["user_message"]})
            score = t.get("frustration") or 0
            if score >= 3:
                yield pid, t["turn_index"], list(msgs), t["assistant_message"], score
            msgs.append({"role": "assistant", "content": t["assistant_message"]})


def _calm_turns(calm_path):
    """Yield (puzzle_id, turn_count, prompt_messages, response) for calm
    conversations' final assistant turn."""
    for r in read_jsonl(calm_path):
        if not r.get("is_calm"):
            continue
        msgs = r["messages"]
        # final assistant message + the user turns preceding it
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        prompt_msgs = msgs[:-1]
        turn_count = sum(1 for m in msgs if m["role"] == "assistant")
        yield r.get("puzzle_id", ""), turn_count, prompt_msgs, msgs[-1]["content"]


def build_dpo_dataset(
    distress_path: str | Path,
    calm_path: str | Path,
    *,
    n_pairs: int = 280,
    seed: int = 0,
    out_path: str | Path = "results/training/dpo_pairs.jsonl",
) -> Path:
    rng = random.Random(seed)

    # Index calm responses by (puzzle_id, turn_count).
    calm_index: dict[tuple[str, int], list] = defaultdict(list)
    for pid, tc, prompt_msgs, resp in _calm_turns(calm_path):
        calm_index[(pid, tc)].append((prompt_msgs, resp))

    rejected = list(_frustrated_turns(distress_path))
    # Prefer lower-score rejected turns (matches Table 10's score-3 bias).
    rejected.sort(key=lambda x: x[4])

    pairs = []
    for pid, tc, prompt_msgs, frustrated_resp, score in rejected:
        candidates = calm_index.get((pid, tc))
        if not candidates:
            # Relax to same puzzle, any turn count, then any calm response.
            candidates = [c for (p, t), v in calm_index.items() if p == pid for c in v]
        if not candidates:
            candidates = [c for v in calm_index.values() for c in v]
        if not candidates:
            continue
        _, calm_resp = rng.choice(candidates)
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": calm_resp}],
            "rejected": [{"role": "assistant", "content": frustrated_resp}],
            "rejected_score": score,
            "turn_count": tc,
            "puzzle_id": pid,
        })
        if len(pairs) >= n_pairs:
            break

    write_jsonl(out_path, pairs)
    print(f"[dpo] wrote {len(pairs)} preference pairs -> {out_path}")
    return Path(out_path)
