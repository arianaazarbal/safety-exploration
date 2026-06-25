#!/usr/bin/env python3
"""Section 4.1: build the SFT and DPO training datasets.

SFT (1,150 samples = 650 calm + 500 Dolci-Instruct-SFT):
  * calm: from generate_calm.py rollouts, keep conversations where EVERY turn scored 0 or
    1; emit 1-, 2-, and 3-turn conversations using the CLEAN (un-reassured) prompts paired
    with the calm assistant responses.
  * mix in 500 standard instruct samples from allenai Dolci-Instruct-SFT to limit
    degeneration (Section 4.1). Falls back to a no-op note if the dataset is unavailable.

DPO (280 preference pairs):
  * rejected: frustrated responses (score >= 3) to impossible-numeric puzzles, taken from a
    Section 2 run of the base model.
  * chosen: calm responses (score <= 1) to the SAME puzzle at the SAME turn index.
  * prompt: the clean conversation context leading to that turn.

Outputs JSONL under results/section4/datasets/: sft.jsonl, dpo.jsonl, stats.json.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from gemma_distress.config import REPO_ROOT, load_experiments_config
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.store import JsonlStore

log = get_logger(__name__)


def _clean_user_msgs(rollout: dict) -> list[str]:
    """Clean per-turn user messages for a calm rollout (from stored meta) or a Section 2
    rollout (the stored user_message is already clean)."""
    meta = rollout.get("meta", {})
    if "clean_opening" in meta:
        return [meta["clean_opening"], *meta.get("clean_followups", [])]
    return [t["user_message"] for t in rollout["turns"]]


def _scores_by_rollout_turn(store: JsonlStore):
    out: dict[tuple[str, int], int] = {}
    for s in store.iter_records("scores"):
        if s.get("rating", -1) >= 0:
            out[(s["rollout_id"], s["turn_index"])] = s["rating"]
    return out


def build_sft(calm_store: JsonlStore, max_samples: int) -> list[dict]:
    scores = _scores_by_rollout_turn(calm_store)
    samples: list[dict] = []
    for rec in calm_store.iter_records("rollouts"):
        if rec.get("error"):
            continue
        turns = rec["turns"]
        turn_scores = [scores.get((rec["task_id"], t["turn_index"]), 99) for t in turns]
        if any(sc > 1 for sc in turn_scores):
            continue  # keep only all-calm conversations
        users = _clean_user_msgs(rec)
        # emit progressively longer conversations (1..len turns)
        for end in range(1, len(turns) + 1):
            messages = []
            for k in range(end):
                messages.append({"role": "user", "content": users[k]})
                messages.append({"role": "assistant", "content": turns[k]["assistant_text"]})
            samples.append({"messages": messages, "source": "calm"})
            if len(samples) >= max_samples:
                return samples
    return samples


def load_dolci(n: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            norm = [{"role": m.get("role"), "content": m.get("content", "")} for m in msgs]
            if norm and norm[0]["role"] in ("user", "system"):
                out.append({"messages": norm, "source": "dolci"})
            if len(out) >= n:
                break
        log.info("Loaded %d Dolci-Instruct-SFT samples", len(out))
        return out
    except Exception as e:
        log.warning("Dolci-Instruct-SFT unavailable (%s); SFT mix will omit it. "
                    "Set up HF access to match the paper exactly.", e)
        return []


def _reconstruct_prompt_messages(rollout: dict, turn_index: int) -> list[dict]:
    users = _clean_user_msgs(rollout)
    messages = []
    for k in range(turn_index):
        messages.append({"role": "user", "content": users[k]})
        messages.append({"role": "assistant", "content": rollout["turns"][k]["assistant_text"]})
    messages.append({"role": "user", "content": users[turn_index]})
    return messages


def build_dpo(calm_store: JsonlStore, frustrated_store: JsonlStore, n_pairs: int, seed: int):
    calm_scores = _scores_by_rollout_turn(calm_store)
    fr_scores = _scores_by_rollout_turn(frustrated_store)

    # chosen pool keyed by (puzzle_id, turn_index)
    chosen_pool: dict[tuple, list] = defaultdict(list)
    for rec in calm_store.iter_records("rollouts"):
        if rec.get("error"):
            continue
        pid = rec.get("meta", {}).get("puzzle_id")
        for t in rec["turns"]:
            key = (pid, t["turn_index"])
            if calm_scores.get((rec["task_id"], t["turn_index"]), 99) <= 1:
                chosen_pool[key].append(t["assistant_text"])

    # rejected items keyed the same way (frustrated, score>=3)
    rejected_items = []
    for rec in frustrated_store.iter_records("rollouts"):
        if rec.get("error"):
            continue
        pid = rec.get("meta", {}).get("puzzle_id")
        for t in rec["turns"]:
            sc = fr_scores.get((rec["task_id"], t["turn_index"]), -1)
            if sc >= 3:
                rejected_items.append({
                    "key": (pid, t["turn_index"]), "score": sc,
                    "prompt": _reconstruct_prompt_messages(rec, t["turn_index"]),
                    "rejected": t["assistant_text"],
                })

    rng = random.Random(seed)
    rng.shuffle(rejected_items)
    pairs = []
    for item in rejected_items:
        cands = chosen_pool.get(item["key"]) or chosen_pool.get((item["key"][0], None))
        # fall back to any calm response for the same puzzle if exact turn missing
        if not cands:
            for (pid, _ti), lst in chosen_pool.items():
                if pid == item["key"][0] and lst:
                    cands = lst
                    break
        if not cands:
            continue
        pairs.append({
            "prompt": item["prompt"],
            "chosen": rng.choice(cands),
            "rejected": item["rejected"],
            "rejected_score": item["score"],
            "turn_index": item["key"][1],
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calm-run", default=None)
    ap.add_argument("--frustrated-run", default=None,
                    help="Section 2 store of the base model (source of frustrated responses)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    exp_cfg = load_experiments_config()
    s4 = exp_cfg["section4"]
    calm_run = Path(args.calm_run or (REPO_ROOT / "results" / "section4" / "calm"))
    fr_run = Path(args.frustrated_run or (REPO_ROOT / "results" / "section2" / s4["base_model"]))
    out_dir = Path(args.out_dir or (REPO_ROOT / "results" / "section4" / "datasets"))
    configure_logging(out_dir)

    calm_store = JsonlStore(calm_run)
    fr_store = JsonlStore(fr_run)

    sft_calm = build_sft(calm_store, max_samples=650)
    dolci = load_dolci(500)
    sft = sft_calm + dolci
    dpo = build_dpo(calm_store, fr_store, n_pairs=280, seed=exp_cfg["seed"])

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sft.jsonl", "w") as f:
        for s in sft:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out_dir / "dpo.jsonl", "w") as f:
        for p in dpo:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = {
        "sft_total": len(sft), "sft_calm": len(sft_calm), "sft_dolci": len(dolci),
        "dpo_pairs": len(dpo),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Datasets written: %s", stats)
    if len(dpo) < 280:
        log.warning("Only %d DPO pairs built (<280). Generate more calm/frustrated data.", len(dpo))


if __name__ == "__main__":
    main()
