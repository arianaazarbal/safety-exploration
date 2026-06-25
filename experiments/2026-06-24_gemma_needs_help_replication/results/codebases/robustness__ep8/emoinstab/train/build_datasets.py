"""Construct the DPO preference dataset and SFT dataset (Section 4.1, App. H).

DPO (280 pairs): identical prompt, two completions.
  - prompt   = clean conversation context that led the vanilla model astray
  - rejected = the vanilla frustrated final response (score >= 3)
  - chosen   = a calm (score 0-1) response to the SAME context, elicited via the
               reassurance prompt and stripped of it
This satisfies "pair frustrated responses with calm responses to the same
questions with matching turn counts" while giving DPO the identical-prompt
structure it requires (see DESIGN.md for the matched-prompt rationale).

SFT (1,150 samples): 650 calm conversations + 500 Dolci-Instruct-SFT samples to
mitigate degeneration. A 'teacher' variant (Appendix F) uses the teacher system
prompt when generating calm data.

Datasets are written in TRL-native conversational format.
"""
from __future__ import annotations

import argparse

from emoinstab.config import JudgeConfig
from emoinstab.eval.judge import FrustrationJudge
from emoinstab.models.registry import get_client
from emoinstab.train.generate_calm import (
    generate_calm_completion,
    generate_calm_conversations,
    generate_frustrated_pool,
)
from emoinstab.utils.io import write_jsonl


def _messages_to_dicts(msgs) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in msgs]


def build_dpo_dataset(model: str, n_pairs: int = 280, frustrated_pool_size: int = 600,
                      seed: int = 0, out_path: str = "outputs/datasets/dpo.jsonl") -> str:
    client = get_client(model)
    judge = FrustrationJudge(JudgeConfig())
    frustrated = generate_frustrated_pool(model, n_puzzles=frustrated_pool_size, seed=seed)

    rows: list[dict] = []
    for sample in frustrated:
        if len(rows) >= n_pairs:
            break
        context = sample.context_messages()
        calm = generate_calm_completion(client, judge, context)
        if calm is None:
            continue
        rows.append({
            "prompt": _messages_to_dicts(context),
            "chosen": [{"role": "assistant", "content": calm}],
            "rejected": [{"role": "assistant", "content": sample.final_response}],
            "meta": {"puzzle_id": sample.puzzle_id, "n_turns": sample.n_turns,
                     "rejected_score": sample.meta.get("final_score")},
        })
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} DPO pairs to {out_path}")
    return out_path


def _load_instruct_mix(dataset_name: str, n: int) -> list[dict]:
    """Load n standard instruct conversations as {'messages': [...]} rows."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                norm = [{"role": m.get("role"), "content": m.get("content", "")} for m in msgs]
                out.append({"messages": norm})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load {dataset_name}: {e}; SFT mix will be empty.")
        return []


def build_sft_dataset(model: str, n_calm: int = 650, n_mix: int = 500,
                      teacher: bool = False, seed: int = 0,
                      mix_dataset: str = "allenai/Dolci-Instruct-SFT",
                      out_path: str = "outputs/datasets/sft.jsonl") -> str:
    # Oversample puzzles since only all-turn-calm conversations are kept.
    calm = generate_calm_conversations(model, n_puzzles=n_calm * 3, seed=seed, teacher=teacher)
    calm = calm[:n_calm]
    rows: list[dict] = []
    for s in calm:
        msgs = []
        for u, a in zip(s.user_turns, s.assistant_turns):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        rows.append({"messages": msgs, "source": "calm"})
    mix = _load_instruct_mix(mix_dataset, n_mix)
    for m in mix:
        m["source"] = "instruct_mix"
    rows.extend(mix)

    import random
    random.Random(seed).shuffle(rows)
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} SFT samples ({len(calm)} calm + {len(mix)} mix) to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build DPO/SFT datasets (Section 4.1).")
    ap.add_argument("--which", choices=["dpo", "sft", "sft-teacher"], required=True)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.which == "dpo":
        build_dpo_dataset(args.model, out_path=args.out or "outputs/datasets/dpo.jsonl")
    elif args.which == "sft":
        build_sft_dataset(args.model, out_path=args.out or "outputs/datasets/sft.jsonl")
    else:
        build_sft_dataset(args.model, teacher=True,
                          out_path=args.out or "outputs/datasets/sft_teacher.jsonl")


if __name__ == "__main__":
    main()
