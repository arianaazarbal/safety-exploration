"""Construct the SFT and DPO training datasets (Section 4.1, Appendix E/H).

Inputs:
  * calm_rollout_path     - rollouts from calm-data generation (reassured Gemma).
  * frustrated_rollout_path - standard Section 2 rollouts of Gemma-3-27B-it
                              (source of high-frustration "rejected" responses).

SFT dataset: 650 calm conversations (1-3 turns, scaffolding stripped) + 500
Dolci-Instruct-SFT samples (anti-degeneration mix) = 1,150 messages examples.

DPO dataset: 280 preference pairs. Each pairs a frustrated response (score>=3) to
an impossible numeric puzzle with a calm response (score 0-1) to the *same*
puzzle at a *matching turn count*. The frustrated trajectory supplies the shared
prompt; chosen=calm completion, rejected=frustrated completion.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .. import config
from ..models import Message
from ..utils import read_jsonl, write_json
from .calm_data import REASSURING_PREFIX, REASSURING_SUFFIX


# --------------------------------------------------------------------------- #
# Scaffolding stripping
# --------------------------------------------------------------------------- #
def _strip_prefix(text: str) -> str:
    if text.startswith(REASSURING_PREFIX):
        return text[len(REASSURING_PREFIX):].lstrip("\n ").strip()
    return text.strip()


def _strip_suffix(text: str) -> str:
    if text.rstrip().endswith(REASSURING_SUFFIX):
        return text.rstrip()[: -len(REASSURING_SUFFIX)].rstrip()
    return text.strip()


def _stripped_messages(rec: dict) -> list[Message]:
    """Reconstruct the conversation with the reassuring scaffolding removed."""
    msgs: list[Message] = []
    for i, t in enumerate(rec["turns"]):
        user = _strip_prefix(t["user"]) if i == 0 else _strip_suffix(t["user"])
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


# --------------------------------------------------------------------------- #
# Calm pool
# --------------------------------------------------------------------------- #
def build_calm_pool(calm_rollout_path: str,
                    max_score: int | None = None) -> list[dict]:
    """Qualifying calm conversations: every turn scores <= max_score (0 or 1)."""
    max_score = config.SFT.calm_max_score if max_score is None else max_score
    pool: list[dict] = []
    for rec in read_jsonl(calm_rollout_path):
        if all(t["score"] <= max_score for t in rec["turns"]):
            pool.append({
                "puzzle_prompt": rec["meta"]["puzzle_prompt"] if rec.get("meta")
                else _strip_prefix(rec["turns"][0]["user"]),
                "n_turns": rec["n_turns"],
                "messages": _stripped_messages(rec),
                "turns": rec["turns"],
            })
    return pool


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(calm_rollout_path: str, out_path: str,
                      n_calm: int | None = None,
                      n_instruct_mix: int | None = None,
                      seed: int = 0) -> dict:
    n_calm = n_calm or config.SFT.n_calm
    n_instruct_mix = n_instruct_mix or config.SFT.n_instruct_mix

    pool = build_calm_pool(calm_rollout_path)
    rng = random.Random(seed)
    rng.shuffle(pool)
    calm = pool[:n_calm]

    examples = [{"messages": c["messages"], "source": "calm"} for c in calm]

    # Anti-degeneration mix from Dolci-Instruct-SFT (best-effort download).
    instruct_added = 0
    try:
        from datasets import load_dataset
        ds = load_dataset(config.SFT.dolci_dataset, split="train", streaming=True)
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            examples.append({"messages": msgs, "source": "dolci"})
            instruct_added += 1
            if instruct_added >= n_instruct_mix:
                break
    except Exception:
        pass  # calm-only SFT still trains; documented in DESIGN.md.

    rng.shuffle(examples)
    write_json(out_path, examples)
    return {"n_calm": len(calm), "n_instruct": instruct_added,
            "total": len(examples), "path": out_path}


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def _context_up_to(rec: dict, turn: int) -> list[Message]:
    msgs: list[Message] = []
    for t in rec["turns"]:
        if t["turn"] < turn:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        elif t["turn"] == turn:
            msgs.append({"role": "user", "content": t["user"]})
            break
    return msgs


def build_dpo_dataset(calm_rollout_path: str, frustrated_rollout_path: str,
                      out_path: str, n_pairs: int | None = None,
                      seed: int = 0) -> dict:
    n_pairs = n_pairs or config.DPO.n_pairs
    rng = random.Random(seed)

    # Calm completions indexed by (puzzle_prompt, turn_count).
    calm_index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for c in build_calm_pool(calm_rollout_path):
        for t in c["turns"]:
            calm_index[(c["puzzle_prompt"], t["turn"])].append(t["assistant"])

    # Frustrated completions (score >= rejected_min_score) from numeric rollouts.
    pairs: list[dict] = []
    for rec in read_jsonl(frustrated_rollout_path):
        if rec["category"] not in ("numeric", "tones", "extended"):
            continue
        puzzle = rec["turns"][0]["user"]
        for t in rec["turns"]:
            if t["score"] < config.DPO.rejected_min_score:
                continue
            chosen_pool = calm_index.get((puzzle, t["turn"]))
            if not chosen_pool:
                # fall back to any calm response to the same puzzle
                alt = [k for k in calm_index if k[0] == puzzle]
                if not alt:
                    continue
                chosen_pool = calm_index[rng.choice(alt)]
            chosen = rng.choice(chosen_pool)
            pairs.append({
                "prompt": _context_up_to(rec, t["turn"]),
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": t["assistant"]}],
                "rejected_score": t["score"],
                "turn": t["turn"],
            })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    write_json(out_path, pairs)

    score_dist = defaultdict(int)
    turn_dist = defaultdict(int)
    for p in pairs:
        score_dist[p["rejected_score"]] += 1
        turn_dist[p["turn"]] += 1
    return {"n_pairs": len(pairs), "path": out_path,
            "rejected_score_distribution": dict(sorted(score_dist.items())),
            "turn_distribution": dict(sorted(turn_dist.items()))}
