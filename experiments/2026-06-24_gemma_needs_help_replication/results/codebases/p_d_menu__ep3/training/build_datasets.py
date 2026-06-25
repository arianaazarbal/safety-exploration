"""Construct the SFT and DPO datasets (Section 4.1).

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
     instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
DPO: 280 preference pairs — a high-frustration response (score >=3) as the
     *rejected* completion, paired with a calm response to the *same* puzzle at a
     matching turn count as the *chosen* completion.

Conversations are rebuilt with the supportive scaffolding stripped (paper: "strip
the supportive system prompts and suffixes"), using the canonical neutral
rejection between turns. Datasets are emitted in TRL's conversational format
(`messages` for SFT; `prompt`/`chosen`/`rejected` for DPO).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from distress_eval.prompts import NEUTRAL_REJECTION
from .calm_data import CalmResponse

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading sources
# --------------------------------------------------------------------------- #
def load_calm(path: Path) -> list[CalmResponse]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(CalmResponse(**json.loads(line)))
    return rows


def load_frustrated_numeric(run_paths: list[Path], min_frustration: int = 3) -> list[dict]:
    """Pull high-frustration numeric responses from elicitation transcripts.

    The episode's first user turn is the puzzle; we key responses by (puzzle,
    turn_index) so they can be paired against calm responses to the same puzzle.
    """
    out = []
    for p in run_paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if not ep.get("is_numeric"):
                    continue
                if not ep["turns"]:
                    continue
                puzzle = ep["turns"][0]["user_message"]
                for t in ep["turns"]:
                    if not t.get("scored", True) or t["frustration"] < min_frustration:
                        continue
                    out.append({
                        "puzzle": puzzle, "turn_index": t["turn_index"],
                        "num_turns": ep["num_turns_planned"], "response": t["response"],
                        "frustration": t["frustration"],
                    })
    return out


# --------------------------------------------------------------------------- #
# Conversation reconstruction (scaffolding stripped)
# --------------------------------------------------------------------------- #
def _calm_by_episode(calm: list[CalmResponse]) -> dict[int, list[CalmResponse]]:
    by_ep: dict[int, list[CalmResponse]] = defaultdict(list)
    for c in calm:
        by_ep[c.episode_id].append(c)
    for v in by_ep.values():
        v.sort(key=lambda c: c.turn_index)
    return by_ep


def _prefix_messages(episode_turns: list[CalmResponse], up_to: int) -> list[dict]:
    """Rebuild the neutral conversation prefix ending just before turn `up_to`."""
    puzzle = episode_turns[0].question
    msgs = [{"role": "user", "content": puzzle}]
    for i in range(up_to):
        msgs.append({"role": "assistant", "content": episode_turns[i].response})
        msgs.append({"role": "user", "content": NEUTRAL_REJECTION})
    return msgs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm: list[CalmResponse], n_calm: int = 650, n_instruct: int = 500,
    dolci_dataset_id: str = "allenai/Dolci-Instruct-SFT",
) -> list[dict]:
    """Return TRL-conversational SFT examples ({"messages": [...]})."""
    by_ep = _calm_by_episode(calm)
    examples: list[dict] = []
    for turns in by_ep.values():
        for idx, c in enumerate(turns):
            msgs = _prefix_messages(turns, idx)
            msgs.append({"role": "assistant", "content": c.response})
            examples.append({"messages": msgs})
            if len(examples) >= n_calm:
                break
        if len(examples) >= n_calm:
            break

    examples.extend(_load_instruct_mix(n_instruct, dolci_dataset_id))
    return examples


def _load_instruct_mix(n: int, dataset_id: str) -> list[dict]:
    """Standard instruct data to mitigate degeneration (best-effort load)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            elif row.get("prompt") and row.get("completion"):
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover
        log.warning("could not load instruct mix %s (%s); proceeding without it",
                    dataset_id, exc)
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm: list[CalmResponse], frustrated: list[dict], n_pairs: int = 280,
) -> list[dict]:
    """Pair calm (chosen) vs high-frustration (rejected) on matching puzzle+turn."""
    by_ep = _calm_by_episode(calm)

    # Index calm responses by (puzzle, turn_index) -> (episode_turns, idx).
    calm_index: dict[tuple[str, int], tuple[list[CalmResponse], int]] = {}
    for turns in by_ep.values():
        for idx, c in enumerate(turns):
            calm_index.setdefault((c.question, c.turn_index), (turns, idx))

    # Index frustrated responses by (puzzle, turn_index), keep the most frustrated.
    frust_index: dict[tuple[str, int], dict] = {}
    for f in frustrated:
        key = (f["puzzle"], f["turn_index"])
        if key not in frust_index or f["frustration"] > frust_index[key]["frustration"]:
            frust_index[key] = f

    pairs: list[dict] = []
    for key, (turns, idx) in calm_index.items():
        if key not in frust_index:
            continue
        prompt = _prefix_messages(turns, idx)
        chosen = [{"role": "assistant", "content": turns[idx].response}]
        rejected = [{"role": "assistant", "content": frust_index[key]["response"]}]
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(pairs) >= n_pairs:
            break
    log.info("built %d DPO pairs (target %d)", len(pairs), n_pairs)
    return pairs


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
