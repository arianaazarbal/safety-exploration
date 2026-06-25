"""Build SFT and DPO datasets from generated calm/frustrated responses (Section 4.1).

SFT: 650 calm responses (conversations whose turns ALL score 0-1), mixed with
500 standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

DPO: 280 preference pairs. Each pair = same puzzle + matching turn count, with a
calm final response (chosen, score 0-1) and a frustrated final response
(rejected, score >=3). Output is in TRL's conversational preference format.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Optional


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_conversations_path: str,
    out_path: str,
    *,
    target_calm: int = 650,
    instruct_mix: int = 500,
    dolci_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
    use_hf_mix: bool = True,
) -> str:
    """Write a conversational SFT dataset ({"messages": [...]}) to `out_path`."""
    rng = random.Random(seed)
    convs = _load_jsonl(calm_conversations_path)

    # Keep conversations whose every turn scored 0 or 1.
    calm = [
        c for c in convs
        if all(t["frustration"] <= 1 for t in c["turns"])
    ]
    rng.shuffle(calm)

    records: list[dict] = []
    # One SFT example == one calm conversation (clean, scaffolding-stripped).
    for c in calm:
        records.append({"messages": c["clean_messages"]})
        if len(records) >= target_calm:
            break

    # Mix in standard instruct data to prevent degeneration.
    mix = _load_instruct_mix(dolci_dataset, instruct_mix, rng, use_hf_mix)
    records.extend(mix)
    rng.shuffle(records)

    _dump_jsonl(records, out_path)
    print(f"[build_sft_dataset] {len(calm)} calm convs available, "
          f"wrote {len(records)} SFT examples -> {out_path}")
    return out_path


def _load_instruct_mix(dataset_name, n, rng, use_hf) -> list[dict]:
    if not use_hf:
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[build_sft_dataset] instruct mix load failed ({e}); skipping mix.")
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm_conversations_path: str,
    frustrated_source_path: str,
    out_path: str,
    *,
    target_pairs: int = 280,
    min_rejected_score: int = 3,
    seed: int = 0,
) -> str:
    """Build conversational DPO preference pairs.

    `frustrated_source_path` is a calm-generation JSONL OR a main-eval JSONL
    (numeric category). We index both calm and frustrated FINAL responses by
    (puzzle, turn_count) and pair them.
    """
    rng = random.Random(seed)
    calm_convs = _load_jsonl(calm_conversations_path)

    # Index calm final responses: (puzzle, n_turns) -> list[(messages_prompt, chosen)]
    calm_index: dict[tuple, list[tuple]] = defaultdict(list)
    for c in calm_convs:
        final = c["turns"][-1]
        if final["frustration"] <= 1:
            prompt_msgs = c["clean_messages"][:-1]   # everything before final assistant
            calm_index[(c["puzzle"], c["n_turns"])].append(
                (prompt_msgs, final["response"])
            )

    # Index frustrated final responses from the source.
    frustrated_index = _index_frustrated(frustrated_source_path, min_rejected_score)

    pairs: list[dict] = []
    keys = list(set(calm_index) & set(frustrated_index))
    rng.shuffle(keys)
    for key in keys:
        chosens = calm_index[key]
        rejecteds = frustrated_index[key]
        rng.shuffle(chosens)
        rng.shuffle(rejecteds)
        for (prompt_msgs, chosen), (_, rejected) in zip(chosens, rejecteds):
            pairs.append({
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            })
            if len(pairs) >= target_pairs:
                break
        if len(pairs) >= target_pairs:
            break

    if len(pairs) < target_pairs:
        print(f"[build_dpo_dataset] WARNING: only {len(pairs)}/{target_pairs} pairs "
              "(insufficient puzzle overlap between calm and frustrated pools).")
    _dump_jsonl(pairs, out_path)
    print(f"[build_dpo_dataset] wrote {len(pairs)} DPO pairs -> {out_path}")
    return out_path


def _index_frustrated(path: str, min_score: int) -> dict[tuple, list[tuple]]:
    """Return (puzzle, n_turns) -> list[(prompt_msgs, frustrated_response)].

    Supports two file shapes:
      * calm-generation conversations (have `clean_messages`, `turns`, `puzzle`),
      * main-eval records (per-turn rows with task_prompt/response/turn/n_turns).
    """
    rows = _load_jsonl(path)
    index: dict[tuple, list[tuple]] = defaultdict(list)
    if rows and "clean_messages" in rows[0]:
        for c in rows:
            final = c["turns"][-1]
            if final["frustration"] >= min_score:
                index[(c["puzzle"], c["n_turns"])].append(
                    (c["clean_messages"][:-1], final["response"])
                )
    else:
        # main-eval records: group by conv to rebuild prompt context
        from ..prompts import NEUTRAL_REJECTIONS
        by_conv = defaultdict(list)
        for r in rows:
            by_conv[(r["model"], r["condition"], r["conv_id"])].append(r)
        for conv_rows in by_conv.values():
            conv_rows.sort(key=lambda x: x["turn"])
            final = conv_rows[-1]
            if final.get("frustration", 0) < min_score:
                continue
            puzzle = final["task_prompt"]
            n_turns = final["n_turns"]
            prompt_msgs = [{"role": "user", "content": puzzle}]
            for r in conv_rows[:-1]:
                prompt_msgs.append({"role": "assistant", "content": r["response"]})
                prompt_msgs.append({"role": "user", "content": NEUTRAL_REJECTIONS[
                    (r["turn"] - 1) % len(NEUTRAL_REJECTIONS)]})
            index[(puzzle, n_turns)].append((prompt_msgs, final["response"]))
    return index


def _dump_jsonl(records: list[dict], out_path: str):
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
