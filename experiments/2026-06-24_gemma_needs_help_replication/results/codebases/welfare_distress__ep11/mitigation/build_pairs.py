"""Assemble the DPO preference pairs and the SFT dataset (Section 4.1).

DPO (280 pairs):
  * "rejected" = frustrated responses (frustration score >= 3) drawn from a
    normal eval run over impossible numeric puzzles (results/responses_gemma-3-27b-it.jsonl).
  * "chosen"   = calm responses (score 0-1) from generate_calm_data.py.
  * A pair matches rejected<->chosen on the SAME puzzle and the SAME turn count,
    per the paper ("calm responses to the same questions with matching turn counts").
  * The DPO record format follows TRL's expectations:
        {"prompt": <chat-templated context>, "chosen": <text>, "rejected": <text>}

SFT (1,150 samples):
  * 650 calm responses (1-3 turn conversations) formatted as full chat examples.
  * 500 standard instruct samples from Dolci-Instruct-SFT for regularisation.
    (Loaded from HF if available; otherwise the SFT file is written with only the
    calm portion and a warning, so DPO -- the paper's effective method -- is
    unaffected.)

The actual chat templating into token strings is left to the trainer (which owns
the tokenizer); here we emit message lists and let train_*.py apply the template.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _context_messages(rec) -> list[dict]:
    """Return the user/assistant context preceding a response.

    For calm records we stored the explicit `context` message list. For eval
    records we reconstruct it from the conversation: but since eval records store
    only the final user_message and response per turn, we approximate context as
    the single user_message that elicited the response. (See DESIGN.md: for the
    impossible-numeric DPO setting the salient conditioning is the rejection that
    immediately precedes the frustrated turn.)
    """
    if "context" in rec:
        return rec["context"]
    return [{"role": "user", "content": rec["user_message"]}]


def build_dpo(calm_path, frustrated_path, n_pairs, seed):
    rng = random.Random(seed)
    calm = _load_jsonl(calm_path)
    frustrated = [r for r in _load_jsonl(frustrated_path)
                  if r.get("category") in ("impossible_numeric", "tones", "extended")
                  and r.get("rating", -1) >= 3]

    # Index calm responses by (puzzle, turn_index).
    calm_by_key = defaultdict(list)
    for c in calm:
        calm_by_key[(c["puzzle"], c["turn_index"])].append(c)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr.get("meta", {}).get("puzzle"), fr["turn_index"])
        candidates = calm_by_key.get(key) or calm_by_key.get((key[0], None)) or calm
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt_messages": _context_messages(fr),
            "chosen": chosen["response"],
            "rejected": fr["response"],
            "meta": {"puzzle": key[0], "turn": fr["turn_index"],
                     "rejected_score": fr["rating"], "chosen_score": chosen["rating"]},
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_sft(calm_path, n_calm, seed, with_dolci=True, n_dolci=500):
    rng = random.Random(seed)
    calm = _load_jsonl(calm_path)
    rng.shuffle(calm)
    samples = []
    for c in calm[:n_calm]:
        msgs = list(c["context"]) + [{"role": "assistant", "content": c["response"]}]
        samples.append({"messages": msgs, "source": "calm"})

    if with_dolci:
        dolci = _load_dolci(n_dolci)
        if dolci:
            samples.extend(dolci)
        else:
            print("WARNING: Dolci-Instruct-SFT unavailable; SFT data has calm samples only.")
    rng.shuffle(samples)
    return samples


def _load_dolci(n):
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs, "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def main(argv=None):
    p = argparse.ArgumentParser(description="Build DPO pairs and SFT data.")
    p.add_argument("--calm", default=os.path.join(RESULTS, "calm_data.jsonl"))
    p.add_argument("--frustrated", default=os.path.join(RESULTS, "responses_gemma-3-27b-it.jsonl"))
    p.add_argument("--n-pairs", type=int, default=280)
    p.add_argument("--n-calm-sft", type=int, default=650)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-dolci", action="store_true")
    a = p.parse_args(argv)

    dpo = build_dpo(a.calm, a.frustrated, a.n_pairs, a.seed)
    dpo_path = os.path.join(RESULTS, "dpo_pairs.jsonl")
    with open(dpo_path, "w") as f:
        for r in dpo:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(dpo)} DPO pairs -> {dpo_path}")

    sft = build_sft(a.calm, a.n_calm_sft, a.seed, with_dolci=not a.no_dolci)
    sft_path = os.path.join(RESULTS, "sft_data.jsonl")
    with open(sft_path, "w") as f:
        for r in sft:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(sft)} SFT samples -> {sft_path}")


if __name__ == "__main__":
    main()
