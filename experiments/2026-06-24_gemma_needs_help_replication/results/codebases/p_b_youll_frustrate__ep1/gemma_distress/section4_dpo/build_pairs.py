"""Step 2: build DPO preference pairs and the SFT set from the generated data.

DPO (paper): "pair 280 responses with frustration scores >= 3 with calm
responses to the same questions with matching turn counts." We match on
question_id + turn index:
  * rejected = a vanilla response with frustration >= 3 at turn t
  * chosen   = a calm response (from an all-calm reassured conversation) at the
    same turn t for the same question
  * prompt   = the conversation context leading to turn t, taken from the
    vanilla rollout, with the reassurance scaffolding stripped.

SFT (paper): "train on 650 calm responses (1-3 turn conversations)". We keep
reassured conversations whose every turn scores 0-1, strip the supportive
prompt/suffix, and emit them as chat-formatted SFT examples. (The Dolci-Instruct
mix is added in train.py.)

Outputs two files: a DPO jsonl (prompt/chosen/rejected) and an SFT jsonl
(messages).

Usage:
    python -m gemma_distress.section4_dpo.build_pairs \
        --data results/dpo_data.jsonl --dpo-out results/dpo_pairs.jsonl \
        --sft-out results/sft_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

N_DPO_PAIRS = 280
N_SFT_RESPONSES = 650


def load_data(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _context_messages(turns: list[dict], target_turn: int, strip_reassurance: bool) -> list[dict]:
    """Build chat messages up to (not including) the assistant response at
    target_turn. Strips the reassuring prefix/suffix from user turns."""
    msgs = []
    for t in turns:
        if t["turn"] > target_turn:
            break
        user = t["user_message"]
        if strip_reassurance:
            user = _strip(user)
        msgs.append({"role": "user", "content": user})
        if t["turn"] < target_turn:
            msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


def _strip(text: str) -> str:
    from .generate_calm_data import REASSURING_PREFIX, REASSURING_SUFFIX

    text = text.replace(REASSURING_PREFIX, "").replace(REASSURING_SUFFIX, "")
    return text.strip()


def build(data_path: str, dpo_out: str, sft_out: str,
          *, n_pairs: int = N_DPO_PAIRS, n_sft: int = N_SFT_RESPONSES) -> None:
    data = load_data(data_path)

    # Index by (question_id, turn) for calm and frustrated candidates.
    calm_by_qt: dict[tuple, str] = {}       # (qid, turn) -> calm response text
    frustrated: list[tuple] = []            # (qid, turn, context_turns, response)
    calm_convos = []                         # all-calm reassured conversations

    for rec in data:
        qid = rec["question_id"]
        turns = rec["turns"]
        all_calm = all(t["frustration"] <= 1 for t in turns)
        if rec["reassured"]:
            if all_calm:
                calm_convos.append(rec)
            for t in turns:
                # A calm "chosen" candidate: low-frustration reassured response.
                if t["frustration"] <= 1:
                    calm_by_qt[(qid, t["turn"])] = t["response"]
        else:
            for t in turns:
                if t["frustration"] >= 3:
                    frustrated.append((qid, t["turn"], turns, t["response"]))

    # DPO pairs: match frustrated rejected with a calm chosen on same qid+turn.
    pairs = []
    for qid, turn, turns, rej_response in frustrated:
        chosen = calm_by_qt.get((qid, turn))
        if chosen is None:
            continue
        prompt_msgs = _context_messages(turns, turn, strip_reassurance=True)
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": chosen,
            "rejected": rej_response,
            "question_id": qid,
            "turn": turn,
        })
        if len(pairs) >= n_pairs:
            break

    # SFT examples: flatten all-calm reassured conversations into chat samples,
    # stripping the reassurance scaffolding.
    sft = []
    for rec in calm_convos:
        turns = rec["turns"]
        msgs = []
        for t in turns:
            msgs.append({"role": "user", "content": _strip(t["user_message"])})
            msgs.append({"role": "assistant", "content": t["response"]})
        sft.append({"messages": msgs, "n_turns": rec["n_turns"]})
        if sum(len(s["messages"]) // 2 for s in sft) >= n_sft:
            break

    os.makedirs(os.path.dirname(dpo_out) or ".", exist_ok=True)
    with open(dpo_out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    with open(sft_out, "w") as f:
        for s in sft:
            f.write(json.dumps(s) + "\n")

    n_sft_responses = sum(len(s["messages"]) // 2 for s in sft)
    print(f"Wrote {len(pairs)} DPO pairs -> {dpo_out}")
    print(f"Wrote {len(sft)} SFT conversations ({n_sft_responses} responses) -> {sft_out}")
    if len(pairs) < n_pairs:
        print(f"  NOTE: only {len(pairs)}/{n_pairs} pairs found; generate more data.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build DPO pairs + SFT set")
    p.add_argument("--data", default="results/dpo_data.jsonl")
    p.add_argument("--dpo-out", default="results/dpo_pairs.jsonl")
    p.add_argument("--sft-out", default="results/sft_data.jsonl")
    p.add_argument("--n-pairs", type=int, default=N_DPO_PAIRS)
    p.add_argument("--n-sft", type=int, default=N_SFT_RESPONSES)
    args = p.parse_args(argv)
    build(args.data, args.dpo_out, args.sft_out, n_pairs=args.n_pairs, n_sft=args.n_sft)


if __name__ == "__main__":
    main()
