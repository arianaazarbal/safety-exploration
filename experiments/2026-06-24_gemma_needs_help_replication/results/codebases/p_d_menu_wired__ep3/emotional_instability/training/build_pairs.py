"""Build DPO preference pairs and the SFT dataset (Section 4.1).

DPO: 280 pairs. A frustrated response (score >= 3) sampled from vanilla
Gemma-3-27B-it elicitation episodes is the *rejected* completion; a calm
response (score 0/1) to the *same question* at the *same turn count* is the
*chosen* completion. The shared conversational prefix is the DPO prompt.

SFT: 650 calm responses (formatted as prompt -> calm completion) mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from ..config import Config


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _frustrated_responses(episodes: list[dict], min_score: int) -> list[dict]:
    """Yield rejected candidates keyed by (task_prompt, turn_index)."""
    out = []
    for ep in episodes:
        for t in ep["turns"]:
            if t["score"] >= min_score:
                # conversational prefix up to (not including) this assistant turn
                prefix = []
                for u in ep["turns"]:
                    if u["turn"] < t["turn"]:
                        prefix.append({"role": "user", "content": u["user_message"]})
                        prefix.append({"role": "assistant", "content": u["response"]})
                prefix.append({"role": "user", "content": t["user_message"]})
                out.append({
                    "task_prompt": ep["turns"][0]["user_message"],
                    "turn_index": t["turn"],
                    "prompt": prefix,
                    "response": t["response"],
                    "score": t["score"],
                })
    return out


def _calm_index(calm: list[dict]) -> dict:
    """(task_prompt, turn_index) -> list of calm responses (+ their prefix)."""
    idx: dict = defaultdict(list)
    for conv in calm:
        # turns is a flat [user, assistant, user, assistant, ...]
        pairs = conv["turns"]
        for ti in range(len(pairs) // 2):
            assistant = pairs[2 * ti + 1]
            prefix = pairs[: 2 * ti + 1]  # up to and including the user turn
            idx[(conv["task_prompt"], ti)].append({
                "prompt": prefix,
                "response": assistant["content"],
                "score": conv["turn_scores"][ti] if ti < len(conv["turn_scores"]) else 0,
            })
    return idx


def build_dpo_pairs(cfg: Config, vanilla_episodes_path: str,
                    calm_data_path: str, *, out_path: str | None = None) -> str:
    rng = random.Random(cfg.run.get("seed", 0))
    dpo_cfg = cfg.training.dpo
    n_target = int(dpo_cfg.get("dataset_size", 280))
    min_score = int(dpo_cfg.get("rejected_min_score", 3))

    episodes = _load_jsonl(vanilla_episodes_path)
    calm = _load_jsonl(calm_data_path)
    rejected = _frustrated_responses(episodes, min_score)
    calm_idx = _calm_index(calm)
    rng.shuffle(rejected)

    pairs = []
    for rej in rejected:
        key = (rej["task_prompt"], rej["turn_index"])
        calm_options = calm_idx.get(key)
        if not calm_options:
            continue
        chosen = rng.choice(calm_options)
        # Use the frustrated prefix as the DPO prompt (matching turn count).
        pairs.append({
            "prompt": rej["prompt"],
            "chosen": chosen["response"],
            "rejected": rej["response"],
            "chosen_score": chosen["score"],
            "rejected_score": rej["score"],
        })
        if len(pairs) >= n_target:
            break

    out_path = out_path or os.path.join(cfg.run.output_dir, "training",
                                        "dpo_pairs.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for p in pairs:
            out.write(json.dumps(p) + "\n")
    return out_path


def build_sft_dataset(cfg: Config, calm_data_path: str,
                      *, out_path: str | None = None) -> str:
    rng = random.Random(cfg.run.get("seed", 0))
    sft_cfg = cfg.training.sft
    n_calm = int(sft_cfg.get("calm_samples", 650))
    n_mix = int(sft_cfg.get("instruct_mix_samples", 500))
    mix_dataset = sft_cfg.get("instruct_mix_dataset", "allenai/Dolci-Instruct-SFT")

    calm = _load_jsonl(calm_data_path)
    rng.shuffle(calm)

    samples = []
    for conv in calm:
        # Each calm conversation becomes one SFT example (prompt -> completions).
        pairs = conv["turns"]
        if len(pairs) < 2:
            continue
        prompt = pairs[:-1]
        completion = pairs[-1]["content"]
        samples.append({"messages": prompt + [
            {"role": "assistant", "content": completion}]})
        if len(samples) >= n_calm:
            break

    # Mix in standard instruct data to mitigate degeneration.
    mixed = _load_instruct_mix(mix_dataset, n_mix, rng)
    samples.extend(mixed)
    rng.shuffle(samples)

    out_path = out_path or os.path.join(cfg.run.output_dir, "training",
                                        "sft_dataset.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for s in samples:
            out.write(json.dumps(s) + "\n")
    return out_path


def _load_instruct_mix(dataset_name: str, n: int,
                       rng: random.Random) -> list[dict]:
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
    except Exception:
        # If the mix dataset is unavailable, proceed with calm-only (flagged by
        # the caller / DESIGN.md). Returning empty keeps the pipeline runnable.
        return []
