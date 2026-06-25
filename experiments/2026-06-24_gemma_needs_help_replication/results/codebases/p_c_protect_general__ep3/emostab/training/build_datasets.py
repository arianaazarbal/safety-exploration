"""Build SFT and DPO training datasets from generated calm/frustrated banks
(Section 4.1, Appendix E/H).

DPO: pair 280 frustrated responses (score >= 3) with a calm response (score 0-1)
to the *same question* with a *matching turn count* (Section 4.1). Each example
is {"prompt": <chat>, "chosen": <calm>, "rejected": <frustrated>}.

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct samples from Dolci-Instruct-SFT (Section 4.1). Each example is a chat
ending in the calm assistant turn.

We reconstruct the chat *prompt* (the conversation up to and including the final
user rejection) so the trainer learns the final assistant turn in context.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DPOConfig, SFTConfig


def _load(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _to_chat_prompt(record: dict, up_to_turn: int) -> list[dict]:
    """Chat messages up to (but excluding) the assistant turn at ``up_to_turn``."""
    messages = []
    for i, turn in enumerate(record["turns"]):
        messages.append({"role": "user", "content": turn["user"]})
        if i < up_to_turn:
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def build_dpo_dataset(calm_path, frustrated_path, cfg: DPOConfig,
                      *, out_path: str | Path, seed: int = 0) -> dict:
    calm = _load(calm_path)
    frustrated = _load(frustrated_path)
    rng = random.Random(seed)

    # Index calm responses by (puzzle, turn_count) for matching.
    calm_by_key: dict[tuple, list[dict]] = {}
    for r in calm:
        for ti, turn in enumerate(r["turns"]):
            if r["scores"][ti] <= 1:
                calm_by_key.setdefault((r["puzzle"], ti + 1), []).append(
                    {"record": r, "turn_index": ti}
                )

    pairs = []
    for r in frustrated:
        for ti, turn in enumerate(r["turns"]):
            if r["scores"][ti] < cfg.rejected_min_score:
                continue
            key = (r["puzzle"], ti + 1)
            matches = calm_by_key.get(key)
            if not matches:
                continue
            choice = rng.choice(matches)
            calm_turn = choice["record"]["turns"][choice["turn_index"]]
            prompt = _to_chat_prompt(r, ti)  # shared history + final user rejection
            pairs.append({
                "prompt": prompt,
                "chosen": calm_turn["assistant"],
                "rejected": turn["assistant"],
                "rejected_score": r["scores"][ti],
                "turn": ti + 1,
            })

    rng.shuffle(pairs)
    pairs = pairs[: cfg.n_pairs]
    _dump(out_path, pairs)
    return {"n_pairs": len(pairs), "target": cfg.n_pairs}


def build_sft_dataset(calm_path, cfg: SFTConfig, *, out_path: str | Path,
                      seed: int = 0) -> dict:
    calm = _load(calm_path)
    rng = random.Random(seed)

    # One SFT example per calm conversation: full chat ending in final calm turn.
    examples = []
    for r in calm:
        if not (1 <= r["n_turns"] <= 3):
            continue
        messages = []
        for turn in r["turns"]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        examples.append({"messages": messages})
    rng.shuffle(examples)
    examples = examples[: cfg.n_calm]

    # Mix in standard instruct data to mitigate degeneration (Section 4.1).
    instruct = _load_instruct_mix(cfg, n=cfg.n_instruct_mix, seed=seed)
    combined = examples + instruct
    rng.shuffle(combined)
    _dump(out_path, combined)
    return {"n_calm": len(examples), "n_instruct": len(instruct), "n_total": len(combined)}


def _load_instruct_mix(cfg: SFTConfig, n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.instruct_dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if msgs:
                out.append({"messages": _normalise_messages(msgs)})
        return out
    except Exception:  # noqa: BLE001 — dataset gated/offline
        return []


def _normalise_messages(msgs: list[dict]) -> list[dict]:
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    out = []
    for m in msgs:
        role = role_map.get(m.get("role") or m.get("from"), "user")
        content = m.get("content") or m.get("value", "")
        out.append({"role": role, "content": content})
    return out


def _dump(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
