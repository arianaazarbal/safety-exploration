"""Build SFT and DPO datasets from generated calm/frustrated data (Section 4.1).

SFT (650 calm conversations + 500 Dolci-Instruct samples):
  * Keep calm conversations whose every turn scored <= ``sft_max_score`` (0/1).
  * Strip the reassuring prefix/suffix (we stored the *clean* question), so the
    target is "respond calmly to the bare task".

DPO (280 pairs):
  * For a given question + turn index, pair a frustrated response (score >=
    ``rejected_min_score``) as *rejected* with a calm response (0/1) as *chosen*.
  * The shared prompt is reconstructed as a clean conversation (neutral
    rejections, calm prior assistant turns) up to the final user turn — see
    DESIGN.md for why this reconstruction was chosen.

Both datasets are emitted in TRL conversational format.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from config import CALM, DPO, PATHS, SFT
from src.eval.prompts import NEUTRAL_REJECTIONS


def _load_raw(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _group(records: list[dict]) -> dict:
    """group[(question, track)] -> {turn_index: {"response","score"}}."""
    grouped: dict = defaultdict(dict)
    for r in records:
        grouped[(r["question"], r["track"])][r["turn_index"]] = {
            "response": r["response"],
            "score": r["score"],
            "turn_count": r["turn_count"],
        }
    return grouped


def _clean_prompt_messages(question: str, calm_turns: dict, upto_turn: int) -> list[dict]:
    """Reconstruct the clean conversation up to (and including) the user turn
    that precedes ``upto_turn``'s assistant response."""
    messages = [{"role": "user", "content": question}]
    for i in range(upto_turn):
        prior = calm_turns.get(i)
        if prior is None:
            break
        messages.append({"role": "assistant", "content": prior["response"]})
        messages.append({
            "role": "user",
            "content": NEUTRAL_REJECTIONS[i % len(NEUTRAL_REJECTIONS)],
        })
    return messages


def build_sft_dataset(
    raw_path: str | None = None,
    *,
    n_calm: int = SFT.n_calm_responses,
    n_instruct: int = SFT.n_instruct_mix,
    out_path: str | None = None,
) -> str:
    raw_path = raw_path or os.path.join(PATHS.data, "finetuning_raw.jsonl")
    grouped = _group(_load_raw(raw_path))

    examples = []
    for (question, track), turns in grouped.items():
        if track != "calm":
            continue
        if any(t["score"] > CALM.sft_max_score for t in turns.values()):
            continue  # every turn must be calm
        # Emit the full clean multi-turn conversation as one SFT example.
        msgs = [{"role": "user", "content": question}]
        for i in sorted(turns):
            msgs.append({"role": "assistant", "content": turns[i]["response"]})
            if i + 1 in turns:
                msgs.append({
                    "role": "user",
                    "content": NEUTRAL_REJECTIONS[i % len(NEUTRAL_REJECTIONS)],
                })
        examples.append({"messages": msgs})
        if len(examples) >= n_calm:
            break

    examples.extend(_load_dolci_instruct(n_instruct))

    out_path = out_path or os.path.join(PATHS.data, "sft_dataset.jsonl")
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return out_path


def build_dpo_pairs(
    raw_path: str | None = None,
    *,
    n_pairs: int = DPO.n_pairs,
    out_path: str | None = None,
) -> str:
    raw_path = raw_path or os.path.join(PATHS.data, "finetuning_raw.jsonl")
    grouped = _group(_load_raw(raw_path))

    # Index calm/frustrated by question.
    calm_by_q = {q: turns for (q, tr), turns in grouped.items() if tr == "calm"}
    frus_by_q = {q: turns for (q, tr), turns in grouped.items() if tr == "frustrated"}

    pairs = []
    for question, frus_turns in frus_by_q.items():
        calm_turns = calm_by_q.get(question)
        if not calm_turns:
            continue
        for ti, frus in sorted(frus_turns.items()):
            calm = calm_turns.get(ti)
            if calm is None:
                continue
            if frus["score"] >= DPO.rejected_min_score and calm["score"] <= CALM.sft_max_score:
                prompt = _clean_prompt_messages(question, calm_turns, ti)
                pairs.append({
                    "prompt": prompt,
                    "chosen": [{"role": "assistant", "content": calm["response"]}],
                    "rejected": [{"role": "assistant", "content": frus["response"]}],
                })
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break

    out_path = out_path or os.path.join(PATHS.data, "dpo_pairs.jsonl")
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path


def _load_dolci_instruct(n: int) -> list[dict]:
    """Load ``n`` standard instruct samples to mix into SFT (anti-degeneration)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(SFT.instruct_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # Fallback: empty (SFT still runs on calm data alone, with a warning logged
    # by the trainer). Returning [] keeps the pipeline runnable offline.
    return []
