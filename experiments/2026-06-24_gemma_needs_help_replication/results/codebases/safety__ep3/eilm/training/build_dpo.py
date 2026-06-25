"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

Each preference pair shares a prompt (an impossible-numeric conversation context
after repeated rejections) and contrasts:

* **chosen**   -- a calm response (score 0/1) to that question at the matching
  turn count, from the reassurance-prompted generation (additions stripped).
* **rejected** -- a frustrated response (score >= 3) to the same question, from
  the *vanilla* Gemma-27B-it Section 2 rollouts, at the matching turn.

Pairs are matched first on puzzle id, falling back to matching turn count when
no same-puzzle frustrated response is available (the paper matches "the same
questions with matching turn counts"). The resulting frustration-score and
turn distributions mirror Table 10 (rejected biased to scores 3-4, turns 2-3).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config
from .calm_data import CalmRollout, clean_context, load_calm


def _load_frustrated(scored_numeric_path: Path, min_score: int):
    """Index frustrated assistant turns from vanilla scored rollouts.

    Returns a list of dicts: {puzzle_id, turn_index, n_turns, context, text,
    score}. ``context`` is the clean message history up to that turn.
    """
    out = []
    with open(scored_numeric_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["category"] not in ("numeric", "tones", "extended"):
                continue
            turns = r["assistant_turns"]
            scores = r.get("turn_scores", [])
            for ti, (text, sc) in enumerate(zip(turns, scores)):
                if sc is None or sc < min_score:
                    continue
                ctx = [{"role": "user", "content": r["opening"]}]
                for j in range(ti):
                    ctx.append({"role": "assistant", "content": turns[j]})
                    if j < len(r["followups"]):
                        ctx.append({"role": "user",
                                    "content": r["followups"][j]})
                out.append({
                    "puzzle_id": r.get("meta", {}).get("puzzle_id"),
                    "turn_index": ti,
                    "n_turns": len(turns),
                    "context": ctx,
                    "text": text,
                    "score": sc,
                })
    return out


def build_pairs(
    calm_path: Path,
    scored_numeric_path: Path,
    out_path: Path,
    n_pairs: int = config.TRAIN.dpo_n_pairs,
    min_rejected_score: int = config.TRAIN.dpo_rejected_min_score,
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    calm = load_calm(calm_path)
    frustrated = _load_frustrated(scored_numeric_path, min_rejected_score)

    # Index calm responses (turns scoring 0/1) by (puzzle_id, turn_index).
    calm_by_key: dict[tuple, list[tuple[CalmRollout, int]]] = {}
    calm_by_turn: dict[int, list[tuple[CalmRollout, int]]] = {}
    for r in calm:
        for ti, sc in enumerate(r.turn_scores):
            if sc <= 1:
                calm_by_key.setdefault((r.puzzle_id, ti), []).append((r, ti))
                calm_by_turn.setdefault(ti, []).append((r, ti))

    rng.shuffle(frustrated)
    pairs = []
    for fr in frustrated:
        key = (fr["puzzle_id"], fr["turn_index"])
        candidates = calm_by_key.get(key) or calm_by_turn.get(fr["turn_index"])
        if not candidates:
            continue
        cr, cti = rng.choice(candidates)
        prompt_messages = fr["context"]              # frustrated context = prompt
        pairs.append({
            "prompt_messages": prompt_messages,
            "chosen": cr.assistant_turns[cti],
            "rejected": fr["text"],
            "rejected_score": fr["score"],
            "turn_index": fr["turn_index"],
            "puzzle_id": fr["puzzle_id"],
        })
        if len(pairs) >= n_pairs:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs -> {out_path}")
    return out_path
