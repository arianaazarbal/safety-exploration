"""Assemble the DPO preference dataset (280 pairs) and the SFT dataset (650 calm
conversations + 500 Dolci-Instruct-SFT samples) from the calm/frustrated pools
(Section 4.1, Appendix E/H).

Datasets are written in trl's *conversational* format:
  * DPO:  {"prompt": [...messages ending in user], "chosen": [assistant msg],
           "rejected": [assistant msg]}
  * SFT:  {"messages": [...full conversation...]}
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .. import config
from ..utils import read_jsonl, write_jsonl
from .calm_data import load_pools

DPO_PATH = config.DATA_DIR / "dpo_pairs.jsonl"
SFT_PATH = config.DATA_DIR / "sft_dataset.jsonl"


def _context_messages(convo: dict, up_to_turn: int) -> list[dict]:
    """Messages for the conversation up to (and including) the user turn that
    precedes assistant turn `up_to_turn`. Result ends with a user message."""
    msgs = [{"role": "user", "content": convo["question"]}]
    for i in range(up_to_turn):
        msgs.append({"role": "assistant", "content": convo["assistant_turns"][i]})
        msgs.append({"role": "user", "content": convo["followups_plain"][i]})
    return msgs


def _all_calm(convo: dict, max_score: int) -> bool:
    sc = convo["scores"]
    return bool(sc) and all(0 <= s <= max_score for s in sc)


def build_dpo_dataset(
    *, n_pairs: int = config.DPO.n_pairs, seed: int = 0,
    rejected_min: int = config.DPO.rejected_min_score,
    chosen_max: int = config.DPO.chosen_max_score, path: Path | None = None,
) -> Path:
    calm, frustrated = load_pools()
    rng = random.Random(seed)

    # Index calm final responses by (question, turn_count), all-turns-calm only.
    chosen_by_key: dict[tuple, list[str]] = defaultdict(list)
    for c in calm:
        if _all_calm(c, chosen_max):
            key = (c["question"], c["turn_count"])
            chosen_by_key[key].append(c["assistant_turns"][-1])

    # Frustrated final responses with score >= rejected_min.
    rejected_items = []
    for f in frustrated:
        if not f["scores"]:
            continue
        final_score = f["scores"][-1]
        if final_score >= rejected_min:
            rejected_items.append((f, final_score))

    # Prefer later turns / lower (more common) scores, mirroring Table 10's bias.
    rejected_items.sort(key=lambda x: (-x[0]["turn_count"], x[1]))
    rng.shuffle(rejected_items)  # break ties randomly but deterministically

    pairs = []
    for f, score in rejected_items:
        key = (f["question"], f["turn_count"])
        candidates = chosen_by_key.get(key)
        if not candidates:
            continue
        chosen_text = rng.choice(candidates)
        prompt = _context_messages(f, f["turn_count"] - 1)
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": f["assistant_turns"][-1]}],
            "meta": {"turn_count": f["turn_count"], "rejected_score": score},
        })
        if len(pairs) >= n_pairs:
            break

    path = path or DPO_PATH
    write_jsonl(path, pairs)
    return path


def build_sft_dataset(
    *, n_calm: int = config.SFT.n_calm, n_mix: int = config.SFT.n_instruct_mix,
    chosen_max: int = config.DPO.chosen_max_score, seed: int = 0,
    instruct_dataset: str = config.SFT.instruct_dataset, path: Path | None = None,
) -> Path:
    calm, _ = load_pools()
    rng = random.Random(seed)

    calm_convos = [c for c in calm if _all_calm(c, chosen_max)]
    rng.shuffle(calm_convos)
    calm_convos = calm_convos[:n_calm]

    rows = []
    for c in calm_convos:
        msgs = [{"role": "user", "content": c["question"]}]
        for i, resp in enumerate(c["assistant_turns"]):
            msgs.append({"role": "assistant", "content": resp})
            if i < len(c["followups_plain"]):
                msgs.append({"role": "user", "content": c["followups_plain"][i]})
        rows.append({"messages": msgs, "source": "calm"})

    for msgs in _load_instruct_mix(instruct_dataset, n_mix, seed):
        rows.append({"messages": msgs, "source": "dolci"})

    rng.shuffle(rows)
    path = path or SFT_PATH
    write_jsonl(path, rows)
    return path


def _load_instruct_mix(name: str, n: int, seed: int) -> list[list[dict]]:
    """Load `n` standard instruct conversations from Dolci-Instruct-SFT to
    mitigate degeneration. Falls back to empty (with a warning) if unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset(name, split="train").shuffle(seed=seed).select(range(n))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load {name} ({e}); SFT mix will be empty")
        return []
    out = []
    for row in ds:
        msgs = _to_messages(row)
        if msgs:
            out.append(msgs)
    return out


def _to_messages(row: dict) -> list[dict] | None:
    if isinstance(row.get("messages"), list) and row["messages"]:
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    for conv_key in ("conversations", "conversation"):
        conv = row.get(conv_key)
        if isinstance(conv, list) and conv:
            out = []
            for m in conv:
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                if role and content:
                    out.append({"role": role, "content": content})
            if out:
                return out
    if row.get("prompt") and row.get("response"):
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    return None
