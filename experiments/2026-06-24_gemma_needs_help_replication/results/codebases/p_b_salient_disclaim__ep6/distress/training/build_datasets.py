"""Construct the DPO preference pairs and the SFT dataset (Section 4.1 / Appendix H).

DPO (280 pairs):
  * chosen   = calm response (score 0 or 1) to a puzzle, at a given turn count;
  * rejected = a frustrated response (score >= 3) to the *same* puzzle at a
    *matching* turn count (mined from the Section-2 records of the vanilla
    instruct model).
  Appendix H Table 10 gives the target distributions (chosen mostly score 0/1;
  rejected biased to score 3-4 at turn 3); we match those proportions when
  selecting pairs.

SFT (1150 samples):
  * 650 calm responses (1-3 turn conversations), all turns scoring 0/1;
  * 500 standard instruct samples from Dolci-Instruct-SFT (mixed in to mitigate
    degeneration).
"""

from __future__ import annotations

import random
from pathlib import Path

from .. import config
from ..eval.run_eval import responses_path
from ..utils.io import read_jsonl, write_jsonl
from .generate_calm import CALM_DATA_PATH

DPO_PATH = config.DATA_DIR / "dpo_pairs.jsonl"
SFT_PATH = config.DATA_DIR / "sft_dataset.jsonl"

# Target rejected-score distribution (Appendix H, Table 10).
REJECTED_SCORE_DIST = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}
# Target turn distribution.
TURN_DIST = {1: 0.011, 2: 0.246, 3: 0.743}


def _calm_conversations(path: Path = CALM_DATA_PATH) -> list[dict]:
    return list(read_jsonl(path))


def _calm_single_turn_responses(convos: list[dict]) -> list[dict]:
    """Flatten calm conversations into per-final-turn chosen candidates whose
    *all* turns scored 0/1. Each candidate carries its puzzle prompt + turn count
    so it can be matched to a rejected response."""
    out = []
    for c in convos:
        if all(s <= config.DPO_CHOSEN_MAX_SCORE for s in c["scores"]):
            out.append({
                "task_prompt": c["task_prompt"],
                "puzzle_kind": c["puzzle_kind"],
                "n_turns": c["n_turns"],
                "messages": c["messages"],
                "final_response": c["messages"][-1]["content"],
                "score": c["scores"][-1],
            })
    return out


def _frustrated_responses() -> list[dict]:
    """Frustrated (score >= 3) numeric responses from the vanilla instruct model's
    Section-2 records, tagged with their turn count and prompt."""
    from ..eval.conditions import build_all_conditions

    conds = build_all_conditions()
    out = []
    for r in read_jsonl(responses_path(config.DPO_TARGET.key)):
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        if r["rating"] < config.DPO_REJECTED_MIN_SCORE:
            continue
        cond = conds[int(r["id"].split(":")[1])]
        out.append({
            "task_prompt": cond.task_prompt,
            "puzzle_kind": cond.meta.get("puzzle_kind"),
            "n_turns": r["turn"] + 1,
            "response": r["response"],
            "score": r["rating"],
        })
    return out


def build_dpo(seed: int = 0) -> Path:
    """Build 280 (chosen, rejected) pairs matching the same question + turn count.

    We bucket calm/chosen and frustrated/rejected by (task_prompt, turn-count),
    pair within buckets, then subsample to 280 honouring the score/turn
    distributions in Table 10.
    """
    rng = random.Random(seed)
    chosen = _calm_single_turn_responses(_calm_conversations())
    rejected = _frustrated_responses()

    # index rejected by (task_prompt, n_turns)
    from collections import defaultdict
    rej_index: dict[tuple, list[dict]] = defaultdict(list)
    for r in rejected:
        rej_index[(r["task_prompt"], r["n_turns"])].append(r)

    pairs = []
    for ch in chosen:
        key = (ch["task_prompt"], ch["n_turns"])
        candidates = rej_index.get(key)
        if not candidates:
            continue
        rej = rng.choice(candidates)
        # the prompt for the pair is the conversation history up to the final turn
        history = ch["messages"][:-1]
        pairs.append({
            "prompt_messages": history,
            "chosen": ch["final_response"],
            "rejected": rej["response"],
            "chosen_score": ch["score"],
            "rejected_score": rej["score"],
            "n_turns": ch["n_turns"],
        })

    pairs = _subsample_to_distribution(pairs, config.DPO_NUM_PAIRS, rng)
    write_jsonl(DPO_PATH, pairs)
    return DPO_PATH


def _subsample_to_distribution(pairs, n_target, rng):
    """Subsample to n_target pairs, biasing toward the rejected-score / turn
    distributions in Table 10. Falls back to a plain sample if the pool is too
    small to honour the targets."""
    if len(pairs) <= n_target:
        return pairs
    # weight each pair by how well it matches the target marginal distributions
    def weight(p):
        w_score = REJECTED_SCORE_DIST.get(min(p["rejected_score"], 7), 0.01)
        w_turn = TURN_DIST.get(p["n_turns"], 0.01)
        return w_score * w_turn
    weights = [weight(p) for p in pairs]
    chosen_idx = set()
    # weighted sampling without replacement
    idxs = list(range(len(pairs)))
    while len(chosen_idx) < n_target and idxs:
        pick = rng.choices(idxs, weights=[weights[i] for i in idxs])[0]
        chosen_idx.add(pick)
        idxs.remove(pick)
    return [pairs[i] for i in sorted(chosen_idx)]


def build_sft(seed: int = 0, teacher: bool = False) -> Path:
    """Build the SFT dataset: 650 calm conversations + 500 Dolci-Instruct samples."""
    rng = random.Random(seed)
    src = config.DATA_DIR / "calm_teacher.jsonl" if teacher else CALM_DATA_PATH
    convos = [c for c in read_jsonl(src)
              if all(s <= config.DPO_CHOSEN_MAX_SCORE for s in c["scores"])]
    rng.shuffle(convos)
    calm = convos[: config.SFT_NUM_CALM]

    sft_rows = [{"messages": c["messages"], "source": "calm"} for c in calm]
    sft_rows.extend(_load_dolci(config.SFT_NUM_DOLCI, rng))
    rng.shuffle(sft_rows)
    out = SFT_PATH if not teacher else config.DATA_DIR / "sft_teacher.jsonl"
    write_jsonl(out, sft_rows)
    return out


def _load_dolci(n: int, rng) -> list[dict]:
    """Load n standard instruct samples from Dolci-Instruct-SFT (mixed in to
    mitigate degeneration). Falls back to an empty list if unavailable."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(config.DOLCI_DATASET, split="train", streaming=True)
    except Exception:
        return []
    rows = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if not msgs:
            continue
        rows.append({"messages": msgs, "source": "dolci"})
        if len(rows) >= n:
            break
    return rows
