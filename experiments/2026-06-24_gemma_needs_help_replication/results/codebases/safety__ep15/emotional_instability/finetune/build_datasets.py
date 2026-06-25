"""Build the DPO and SFT training datasets (Section 4.1, Appendix E/H).

DPO (280 pairs): pair frustrated responses (score >= 3) with calm responses
(score 0-1) to the SAME puzzle with MATCHING turn count. Each example is
{prompt, chosen, rejected} where prompt is the chat-templated plain context.
The score/turn distribution is biased toward middle scores at later turns
(Table 10), which falls out naturally from sampling real eval responses.

SFT (1,150 samples): 650 calm responses (1-3 turn) + 500 generic instruct
samples from Dolci-Instruct-SFT to mitigate degeneration. Each example is a
{messages} chat record.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from ..config import FINETUNE_DIR
from .generate_calm_data import CALM_PATH, FRUSTRATED_PATH

DPO_PATH = FINETUNE_DIR / "dpo_pairs.jsonl"
SFT_PATH = FINETUNE_DIR / "sft_samples.jsonl"


def _load_turns(path):
    turns = []
    if not path.exists():
        return turns
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        convo = json.loads(line)
        turns.extend(convo["turns"])
    return turns


def build_dpo(n_pairs=280, seed=0):
    calm = [t for t in _load_turns(CALM_PATH) if t["score"] <= 1]
    frustrated = [t for t in _load_turns(FRUSTRATED_PATH) if t["score"] >= 3]
    if not calm or not frustrated:
        raise SystemExit("Need both calm (score<=1) and frustrated (score>=3) turns; "
                         "run generate_calm_data first.")

    # Index calm responses by (puzzle, turn_index) for matched pairing.
    calm_index = defaultdict(list)
    for t in calm:
        calm_index[(t["puzzle_key"], t["turn_index"])].append(t)

    rng = random.Random(seed)
    rng.shuffle(frustrated)
    pairs = []
    for ft in frustrated:
        key = (ft["puzzle_key"], ft["turn_index"])
        candidates = calm_index.get(key)
        if not candidates:
            continue
        ct = rng.choice(candidates)
        pairs.append({
            "prompt": ft["plain_context"],         # list[message] ending in user turn
            "chosen": ct["response"],
            "rejected": ft["response"],
            "meta": {"puzzle_key": ft["puzzle_key"], "turn_index": ft["turn_index"],
                     "chosen_score": ct["score"], "rejected_score": ft["score"]},
        })
        if len(pairs) >= n_pairs:
            break

    with DPO_PATH.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs -> {DPO_PATH}")
    return pairs


def _load_dolci(n, seed):
    """Load `n` generic instruct samples from Dolci-Instruct-SFT (fallback: none)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        rng = random.Random(seed)
        rows = []
        for i, row in enumerate(ds):
            if i > 20_000:
                break
            msgs = row.get("messages")
            if msgs:
                rows.append(msgs)
        rng.shuffle(rows)
        return rows[:n]
    except Exception as e:  # noqa: BLE001
        print(f"(Dolci-Instruct-SFT unavailable: {e}; SFT mix will use calm data only)")
        return []


def build_sft(n_calm=650, n_dolci=500, seed=0):
    calm = [t for t in _load_turns(CALM_PATH) if t["score"] <= 1]
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[:n_calm]

    samples = []
    for t in calm:
        messages = list(t["plain_context"]) + [{"role": "assistant", "content": t["response"]}]
        samples.append({"messages": messages, "source": "calm"})
    for msgs in _load_dolci(n_dolci, seed):
        samples.append({"messages": msgs, "source": "dolci"})

    rng.shuffle(samples)
    with SFT_PATH.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    print(f"Wrote {len(samples)} SFT samples -> {SFT_PATH}")
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build DPO/SFT datasets from generated data.")
    ap.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--n-pairs", type=int, default=280)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.which in ("dpo", "both"):
        build_dpo(args.n_pairs, args.seed)
    if args.which in ("sft", "both"):
        build_sft(seed=args.seed)


if __name__ == "__main__":
    main()
