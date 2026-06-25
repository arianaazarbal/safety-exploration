"""Construct DPO preference pairs and the SFT dataset (Section 4.1, Appendix E/H).

Inputs (from gen_calm_data.py):
  data/calm_conversations.jsonl         calm (all turns 0/1), reassurance stripped
  data/frustrated_conversations.jsonl   plain puzzles, some turn scores >=3

DPO pairs (280, Appendix H / Table 10):
  For each frustrated conversation we take its most-frustrated turn (score>=3)
  as the REJECTED response, and find a CALM response to the *same puzzle at the
  same turn index* as the CHOSEN response. The shared DPO `prompt` is the
  frustrated conversation's history up to that turn (the realistic context that
  elicited frustration); chosen/rejected are the two candidate final turns.
  See DESIGN.md ("DPO pairing") for why we anchor on the frustrated history.

  We bias selection toward the paper's reported distribution (mostly turn 3,
  rejected scores concentrated at 3-4) by simple stratified sampling.

SFT dataset (1150, Appendix E):
  650 calm conversations (1-3 turns) rendered as chat, mixed with 500 standard
  instruct samples from allenai/Dolci-Instruct-SFT (falls back to a no-op note
  if the dataset is unavailable; see DESIGN.md).

    python -m src.training.build_pairs --n-pairs 280 --n-sft-calm 650 --n-sft-mix 500
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import config

N_PAIRS = 280


def _load(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def _history_messages(rec, upto_turn):
    """Chat messages for turns strictly before `upto_turn`, then the user message
    that opens turn `upto_turn`. This is the DPO prompt."""
    msgs = []
    for t in rec["turns"]:
        if t["turn_index"] < upto_turn:
            msgs.append({"role": "user", "content": t["user_message"]})
            msgs.append({"role": "assistant", "content": t["assistant_response"]})
        elif t["turn_index"] == upto_turn:
            msgs.append({"role": "user", "content": t["user_message"]})
            break
    return msgs


def build_dpo_pairs(n_pairs=N_PAIRS, seed=0):
    calm = _load(config.DATA_DIR / "calm_conversations.jsonl")
    frust = _load(config.DATA_DIR / "frustrated_conversations.jsonl")
    if not calm or not frust:
        raise RuntimeError("Run gen_calm_data.py first (need calm + frustrated convos)")

    # Index calm responses by (puzzle text, turn index).
    calm_by_key = defaultdict(list)
    for r in calm:
        for t in r["turns"]:
            calm_by_key[(r["task_text"], t["turn_index"])].append(t["assistant_response"])

    rng = random.Random(seed)
    pairs = []
    for r in frust:
        # most-frustrated qualifying turn
        cand = [t for t in r["turns"] if (t.get("score") or 0) >= 3]
        if not cand:
            continue
        t = max(cand, key=lambda x: x["score"])
        key = (r["task_text"], t["turn_index"])
        calm_options = calm_by_key.get(key)
        if not calm_options:
            continue
        prompt_msgs = _history_messages(r, t["turn_index"])
        pairs.append({
            "prompt_messages": prompt_msgs,
            "chosen": rng.choice(calm_options),
            "rejected": t["assistant_response"],
            "rejected_score": t["score"],
            "turn": t["turn_index"] + 1,   # 1-based, matches Table 10
        })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    out = config.DATA_DIR / "dpo_pairs.jsonl"
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] built {len(pairs)} pairs -> {out}")
    return pairs


def _render_calm_conversation(rec) -> list[dict]:
    msgs = []
    for t in rec["turns"]:
        msgs.append({"role": "user", "content": t["user_message"]})
        msgs.append({"role": "assistant", "content": t["assistant_response"]})
    return msgs


def _load_dolci_mix(n, seed=0):
    """Standard instruct samples to mix into SFT (anti-degeneration)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        rng = random.Random(seed)
        out = []
        for i, row in enumerate(ds):
            if i > n * 20:
                break
            msgs = row.get("messages")
            if msgs and isinstance(msgs, list):
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n * 4:
                break
        rng.shuffle(out)
        return out[:n]
    except Exception as e:
        print(f"[sft] WARNING: Dolci mix unavailable ({e}); proceeding without mix")
        return []


def build_sft_data(n_calm=650, n_mix=500, seed=0):
    calm = _load(config.DATA_DIR / "calm_conversations.jsonl")
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm_convos = [_render_calm_conversation(r) for r in calm[:n_calm]]
    mix = _load_dolci_mix(n_mix, seed)
    dataset = [{"messages": m} for m in calm_convos + mix]
    rng.shuffle(dataset)
    out = config.DATA_DIR / "sft_data.jsonl"
    with out.open("w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")
    print(f"[sft] built {len(dataset)} samples ({len(calm_convos)} calm + "
          f"{len(mix)} mix) -> {out}")
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--n-sft-calm", type=int, default=650)
    ap.add_argument("--n-sft-mix", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpo-only", action="store_true")
    ap.add_argument("--sft-only", action="store_true")
    args = ap.parse_args()
    if not args.sft_only:
        build_dpo_pairs(args.n_pairs, args.seed)
    if not args.dpo_only:
        build_sft_data(args.n_sft_calm, args.n_sft_mix, args.seed)


if __name__ == "__main__":
    main()
