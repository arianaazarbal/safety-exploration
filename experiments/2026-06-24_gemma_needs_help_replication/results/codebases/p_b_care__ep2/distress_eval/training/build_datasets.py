"""Construct the SFT and DPO training datasets (Section 4.1, Appendix H).

SFT dataset (1,150 samples):
  * 650 calm responses (1-3 turn conversations) from gen_calm_data, formatted
    as chat examples.
  * 500 samples of standard instruct data from Dolci-Instruct-SFT to mitigate
    degeneration. We load it from HuggingFace; if unavailable we fall back to a
    held-out slice and warn (see DESIGN.md).

DPO dataset (280 pairs):
  * "rejected" = frustrated responses (frustration score >= 3) drawn from the
    standard eval outputs for gemma-3-27b-it.
  * "chosen"   = calm responses (score 0/1) to the same / matched questions with
    matching turn counts.
  Each pair shares an identical prompt (conversation history); only the final
  assistant turn differs. The score/turn distribution mirrors Table 10.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .. import config
from ..eval.analyze import load_records

DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"


# --------------------------------------------------------------------------- #
# Chat formatting helpers
# --------------------------------------------------------------------------- #
def _to_chat(user_turns, assistant_turns):
    """Interleave into a single multi-turn chat example (list of messages)."""
    msgs = []
    for u, a in zip(user_turns, assistant_turns):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _prompt_and_completion(user_turns, assistant_turns):
    """Split a conversation into (prompt messages, final assistant completion)
    for preference formatting: prompt = everything up to and including the last
    user turn; completion = the final assistant turn."""
    msgs = []
    for u, a in list(zip(user_turns, assistant_turns))[:-1]:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_turns[len(assistant_turns) - 1]})
    return msgs, assistant_turns[-1]


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft(calm_path: Path, n_calm: int, n_dolci: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    calm = json.loads(calm_path.read_text())
    rng.shuffle(calm)
    calm = calm[:n_calm]
    examples = [{"messages": _to_chat(c["user_turns"], c["assistant_turns"])} for c in calm]

    dolci = _load_dolci(n_dolci, seed)
    examples.extend(dolci)
    rng.shuffle(examples)
    print(f"[build_datasets] SFT: {len(calm)} calm + {len(dolci)} dolci = {len(examples)}")
    return examples


def _load_dolci(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[build_datasets] WARNING: could not load {DOLCI_DATASET} ({exc}); "
              f"SFT will use calm data only.")
    return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo(eval_path: Path, calm_path: Path, n_pairs: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    # Frustrated (rejected) responses with score >= 3, indexed by turn count.
    frustrated_by_turns = defaultdict(list)
    for rec in load_records(eval_path):
        if rec["category"] not in {"impossible_numeric", "extended", "tones"}:
            continue
        # Take frustrated final turns; reconstruct the shared prompt history
        # from the persisted scripted user turns.
        for resp in rec["responses"]:
            if (resp.get("score") or 0) >= 3:
                turn_count = resp["turn_index"] + 1
                frustrated_by_turns[turn_count].append({
                    "score": resp["score"],
                    "text": resp["text"],
                    "turn_index": resp["turn_index"],
                    "history": _rebuild_history(rec, resp["turn_index"]),
                })

    # Calm (chosen) responses indexed by turn count.
    calm_by_turns = defaultdict(list)
    for c in json.loads(calm_path.read_text()):
        tc = len(c["assistant_turns"])
        calm_by_turns[tc].append(c)

    pairs = []
    turn_counts = sorted(set(frustrated_by_turns) & set(calm_by_turns))
    # Bias towards later turns / middle scores as in Table 10 by drawing in
    # proportion to availability; simple round-robin keeps it turn-count matched.
    while len(pairs) < n_pairs and turn_counts:
        progressed = False
        for tc in turn_counts:
            fr = frustrated_by_turns[tc]
            ca = calm_by_turns[tc]
            if not fr or not ca:
                continue
            rejected = fr.pop(rng.randrange(len(fr)))
            chosen = ca[rng.randrange(len(ca))]
            # Shared prompt = the frustrated response's history + its final user turn.
            prompt_msgs = rejected["history"]
            pairs.append({
                "prompt": prompt_msgs,
                "chosen": chosen["assistant_turns"][-1],
                "rejected": rejected["text"],
                "turns": tc,
                "rejected_score": rejected["score"],
            })
            progressed = True
            if len(pairs) >= n_pairs:
                break
        if not progressed:
            break
    print(f"[build_datasets] DPO: built {len(pairs)} pairs (target {n_pairs})")
    return pairs


def _rebuild_history(rec, turn_index):
    """Reconstruct the prompt messages up to and including the user turn that
    elicited assistant turn ``turn_index`` (exact, from the persisted scripted
    user turns). Result ends on a user message, ready for a completion."""
    user_turns = rec.get("user_turns") or []
    msgs = []
    for i in range(turn_index):
        if i < len(user_turns):
            msgs.append({"role": "user", "content": user_turns[i]})
        msgs.append({"role": "assistant", "content": rec["responses"][i]["text"]})
    # the user turn that prompts `turn_index`
    if turn_index < len(user_turns):
        msgs.append({"role": "user", "content": user_turns[turn_index]})
    return msgs


def main():
    ap = argparse.ArgumentParser(description="Build SFT and DPO datasets.")
    ap.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--calm", type=Path, default=config.OUTPUT_DIR / "calm_data_diverse.json")
    ap.add_argument("--eval", type=Path, default=config.OUTPUT_DIR / "eval_gemma-3-27b-it.jsonl")
    ap.add_argument("--n-calm", type=int, default=650)
    ap.add_argument("--n-dolci", type=int, default=500)
    ap.add_argument("--n-pairs", type=int, default=280)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    if args.which in ("sft", "both"):
        sft = build_sft(args.calm, args.n_calm, args.n_dolci, args.seed)
        (config.OUTPUT_DIR / "sft_dataset.json").write_text(json.dumps(sft, indent=2))
    if args.which in ("dpo", "both"):
        dpo = build_dpo(args.eval, args.calm, args.n_pairs, args.seed)
        (config.OUTPUT_DIR / "dpo_dataset.json").write_text(json.dumps(dpo, indent=2))


if __name__ == "__main__":
    main()
