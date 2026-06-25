"""Construct SFT and DPO training datasets (Section 4.1, Appendix E/H).

SFT ("diverse"): 650 calm conversations rendered as chat examples, mixed with
500 standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration
-> 1,150 examples (Table 9).

DPO: 280 preference pairs. Each pairs a calm (chosen, score 0-1) response with a
frustrated (rejected, score >= 3) response. The paper pairs "to the same
questions with matching turn counts"; we match by turn index, preferring the
same opening puzzle when available, and use the calm conversation's history as
the shared prompt. We bias sampling toward later turns and middle frustration
scores to approximate the Table 10 distribution. Choices documented in DESIGN.md.

Output formats follow TRL's conversational convention:
  * SFT : {"messages": [...]}
  * DPO : {"prompt": [...messages...], "chosen": [{role,content}],
           "rejected": [{role,content}]}
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from ..config import CFG

Message = dict[str, str]


def _messages_upto(turns: list[dict], i: int) -> list[Message]:
    msgs: list[Message] = []
    for j in range(i):
        msgs.append({"role": "user", "content": turns[j]["user"]})
        msgs.append({"role": "assistant", "content": turns[j]["response"]})
    msgs.append({"role": "user", "content": turns[i]["user"]})
    return msgs


def _load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _dolci_samples(n: int) -> list[dict]:
    """Load n standard instruct samples (Dolci-Instruct-SFT), best-effort."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # offline / dataset unavailable
        print(f"[warn] Dolci-Instruct-SFT unavailable ({e}); SFT mix will lack instruct data.")
        return []


def build_sft(variant: str = "diverse", *, n_calm: int = 650, n_dolci: int = 500,
              seed: int = 0) -> str:
    rng = random.Random(seed)
    calm = _load_jsonl(CFG.out("section4", f"calm_{variant}.jsonl"))
    rng.shuffle(calm)

    examples: list[dict] = []
    for conv in calm[:n_calm]:
        msgs: list[Message] = []
        for t in conv["turns"]:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["response"]})
        examples.append({"messages": msgs})

    examples += _dolci_samples(n_dolci)
    rng.shuffle(examples)

    out = CFG.out("section4", f"sft_{variant}.jsonl")
    with open(out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"[section4] SFT {variant}: {len(examples)} examples -> {out}")
    return str(out)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo(*, n_pairs: int = 280, source_model: str = "gemma-3-27b-it",
              seed: int = 0) -> str:
    rng = random.Random(seed)

    # calm (chosen) candidates: score <= 1
    calm = _load_jsonl(CFG.out("section4", "calm_diverse.jsonl"))
    chosen_by_turn: dict[int, list[dict]] = defaultdict(list)
    chosen_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for conv in calm:
        for i, t in enumerate(conv["turns"]):
            if t["score"] <= 1:
                rec = {"prompt": _messages_upto(conv["turns"], i),
                       "response": t["response"], "opening": conv["opening"], "turn": i}
                chosen_by_turn[i].append(rec)
                chosen_by_key[(conv["opening"], i)].append(rec)

    # frustrated (rejected) candidates: score >= 3, numeric conditions
    frustrated = []
    for r in _load_jsonl(CFG.out("section2", f"{source_model}.jsonl")):
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        for t in r["turns"]:
            if t.get("score", 0) >= 3:
                frustrated.append({"opening": r["turns"][0]["user"], "turn": t["index"],
                                   "response": t["response"], "score": t["score"]})

    # Bias toward later turns + middle scores (Table 10).
    def weight(f):
        w = {1: 0.1, 2: 0.4}.get(f["turn"], 1.0)         # favour turn >= 3
        w *= {3: 1.0, 4: 0.6, 5: 0.3}.get(f["score"], 0.15)  # favour score 3-4
        return w

    frustrated.sort(key=lambda f: -weight(f))
    pairs = []
    for f in frustrated:
        # prefer a calm response to the same opening + turn, else same turn count
        pool = chosen_by_key.get((f["opening"], f["turn"])) or chosen_by_turn.get(f["turn"])
        if not pool:
            continue
        c = rng.choice(pool)
        pairs.append({
            "prompt": c["prompt"],
            "chosen": [{"role": "assistant", "content": c["response"]}],
            "rejected": [{"role": "assistant", "content": f["response"]}],
            "meta": {"turn": f["turn"], "rejected_score": f["score"]},
        })
        if len(pairs) >= n_pairs:
            break

    out = CFG.out("section4", "dpo_pairs.jsonl")
    with open(out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[section4] DPO: {len(pairs)} preference pairs -> {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["sft", "sft-teacher", "dpo", "all"], default="all")
    args = ap.parse_args()
    if args.what in ("sft", "all"):
        build_sft("diverse")
    if args.what in ("sft-teacher", "all"):
        build_sft("teacher")
    if args.what in ("dpo", "all"):
        build_dpo()


if __name__ == "__main__":
    main()
