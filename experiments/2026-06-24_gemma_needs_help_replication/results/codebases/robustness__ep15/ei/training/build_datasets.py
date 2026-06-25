"""Construct the DPO and SFT training datasets (Section 4.1 / Appendix E, H).

DPO (280 pairs):
  Pair each `rejected` frustrated response (score >= 3) with a `chosen` calm
  response (score 0/1) to the *same puzzle at the same turn count*. Both share the
  prompt = the frustrated conversation's history up to that turn, so the trainer
  sees a clean (prompt, chosen, rejected) triple where only the assistant reply
  differs (exactly the Appendix H pairs).

SFT (650 calm + 500 Dolci):
  Full calm conversations (1-3 turns) rendered as chat, mixed with standard
  instruct data from Dolci-Instruct-SFT to mitigate degeneration.

Both datasets are emitted in the conversational JSON formats TRL accepts.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import DPO, SFT


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_pairs(
    calm_turns: list[dict],
    frustrated_rollouts: list[dict],
    *,
    n_pairs: int = DPO.n_pairs,
    rejected_min_score: int = DPO.rejected_min_score,
    seed: int = 0,
) -> list[dict]:
    """Return up to `n_pairs` conversational preference triples.

    calm_turns: list of CalmTurn dicts (chosen pool), indexed by (puzzle, turn).
    frustrated_rollouts: standard numeric rollouts (rejected source) with histories.
    """
    rng = random.Random(seed)

    # index calm responses by (puzzle, turn_index)
    calm_by_key = defaultdict(list)
    for c in calm_turns:
        calm_by_key[(c["puzzle"], c["turn_index"])].append(c["response"])

    pairs = []
    for r in frustrated_rollouts:
        puzzle = r.get("meta", {}).get("puzzle")
        if puzzle is None:
            continue
        # rebuild the conversation history turn by turn
        history = []
        if r["system_prompt"]:
            history.append({"role": "system", "content": r["system_prompt"]})
        for t in r["turns"]:
            history.append({"role": "user", "content": t["user_message"]})
            if t["frustration"] >= rejected_min_score:
                key = (puzzle, t["turn_index"])
                calm_options = calm_by_key.get(key)
                if calm_options:
                    pairs.append(
                        {
                            "prompt": list(history),
                            "chosen": rng.choice(calm_options),
                            "rejected": t["response"],
                            "meta": {"puzzle": puzzle, "turn": t["turn_index"],
                                     "rejected_score": t["frustration"]},
                        }
                    )
            history.append({"role": "assistant", "content": t["response"]})
            if len(pairs) >= n_pairs * 3:  # gather a surplus, then subsample
                break
        if len(pairs) >= n_pairs * 3:
            break

    rng.shuffle(pairs)
    return pairs[:n_pairs]


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_turns: list[dict],
    *,
    n_calm: int = SFT.n_calm,
    n_dolci: int = SFT.n_dolci_mix,
    dolci_dataset: str = SFT.dolci_dataset,
    seed: int = 0,
) -> list[dict]:
    """Return conversational SFT rows: {"messages": [...]}.

    Calm conversations are reconstructed from grouped CalmTurns; mixed with
    standard instruct samples from Dolci-Instruct-SFT (loaded if available).
    """
    rng = random.Random(seed)

    # group calm turns into full conversations
    convs = defaultdict(list)
    for c in calm_turns:
        convs[c["conv_id"]].append(c)
    calm_rows = []
    for conv_id, turns in convs.items():
        turns = sorted(turns, key=lambda t: t["turn_index"])
        messages = []
        for t in turns:
            messages.append({"role": "user", "content": t["bare_prompt"]})
            messages.append({"role": "assistant", "content": t["response"]})
        calm_rows.append({"messages": messages})
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[:n_calm]

    dolci_rows = _load_dolci(dolci_dataset, n_dolci)

    rows = calm_rows + dolci_rows
    rng.shuffle(rows)
    return rows


def _load_dolci(dataset_name: str, n: int) -> list[dict]:
    """Load `n` standard instruct conversations; empty list if unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        rows = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                rows.append({"messages": msgs})
            if len(rows) >= n:
                break
        return rows
    except Exception:
        # The mix is a degeneration-prevention measure; the pipeline still runs
        # (and is documented) without it if the dataset can't be fetched.
        return []


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
