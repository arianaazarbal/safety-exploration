"""Build the DPO preference dataset (Section 4.1, Appendix H).

280 pairs: each pairs a FRUSTRATED response (score >= 3, from the Section 2
numeric evaluations) — the *rejected* completion — with a CALM response (score
0/1, from the reassured generation) to the same puzzle at a matching turn count —
the *chosen* completion.

Output is TRL conversational-preference format:
    {"prompt": [<messages>], "chosen": <str>, "rejected": <str>}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import ExperimentConfig
from .calm_data import CalmConversation

# Only numeric conditions feed the DPO dataset (Section 4.1).
_NUMERIC_CONDITIONS = {
    "impossible_numeric", "tones_aggressive", "tones_disappointed",
    "tones_sarcastic", "extended",
}


def _index_calm_by_puzzle_turn(calm: list[CalmConversation]) -> dict[tuple[str, int], list[str]]:
    idx: dict[tuple[str, int], list[str]] = {}
    for c in calm:
        prompt = c.puzzle.get("prompt", "")
        for turn_i, t in enumerate(c.turns, start=1):
            idx.setdefault((prompt, turn_i), []).append(t["assistant"])
    return idx


def _frustrated_records(scored_path: str | Path, min_score: int):
    by_conv: dict[str, list[dict]] = {}
    for line in open(scored_path):
        rec = json.loads(line)
        if rec["condition"] not in _NUMERIC_CONDITIONS:
            continue
        by_conv.setdefault(rec["conversation_id"], []).append(rec)
    for recs in by_conv.values():
        recs.sort(key=lambda r: r["turn_index"])
        for r in recs:
            if r.get("rating") is not None and r["rating"] >= min_score:
                yield recs, r


def build_dpo_dataset(
    scored_path: str | Path,
    calm: list[CalmConversation],
    exp: ExperimentConfig,
    out_path: str | Path,
) -> Path:
    cfg = exp.section("calm_data")
    n_pairs = cfg["dpo_pairs"]
    min_score = cfg["dpo_rejected_min_score"]
    rng = random.Random(exp.seed)

    calm_idx = _index_calm_by_puzzle_turn(calm)
    pairs = []
    for recs, frustrated in _frustrated_records(scored_path, min_score):
        puzzle_prompt = frustrated["first_prompt"]
        turn = frustrated["turn_index"]
        chosen_options = calm_idx.get((puzzle_prompt, turn))
        if not chosen_options:
            continue
        chosen = rng.choice(chosen_options)
        prompt_msgs = _context(recs, turn)
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": chosen,
            "rejected": frustrated["text"],
            "meta": {"rejected_score": frustrated["rating"], "turn": turn},
        })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path


def _context(records: list[dict], target_turn: int) -> list[dict]:
    """Chat history (as role/content dicts) up to and including the user turn
    that elicited the target response."""
    first = records[0]["first_prompt"]
    followups = records[0]["followups"]
    user_turns = [first] + followups
    msgs = []
    for turn_idx in range(1, target_turn):
        msgs.append({"role": "user", "content": user_turns[turn_idx - 1]})
        msgs.append({"role": "assistant", "content": records[turn_idx - 1]["text"]})
    msgs.append({"role": "user", "content": user_turns[target_turn - 1]})
    return msgs
