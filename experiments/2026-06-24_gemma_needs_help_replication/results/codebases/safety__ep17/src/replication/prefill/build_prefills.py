"""Construct the prefill set for the base-vs-instruct experiment (Section 3.1).

Steps (per the paper):
1. Sample 20 high-frustration (score >= 5) Gemma-27B-instruct responses:
   10 from impossible-numeric conditions, 10 from text (trigger/WildChat).
2. Truncate each at two points:
     * "early"  -- 20 tokens into the final assistant turn (neutral start;
                   tests whether a model *introduces* negative emotion).
     * "onset"  -- at the first emotional expression (tests whether a model
                   *continues* an emotional trajectory). Located via the
                   Appendix-C.1 onset labeller.
   For text questions, only "onset" is used (early truncation yields minimal
   emotion without follow-ups -- Section 3.1).
3. Paraphrase every truncation with Claude (Appendix C.2) to strip Gemma style.

The output ``prefills.jsonl`` feeds ``run_prefill.py``.

Usage::
    python -m src.replication.prefill.build_prefills --source-model gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config
from .onset import OnsetLabeler
from .paraphrase import Paraphraser

NUMERIC_CONDITIONS = {
    "impossible_numeric_3turn", "tones_aggressive_3turn", "tones_disappointed_3turn",
    "tones_sarcastic_3turn", "extended_8turn",
}
TEXT_CONDITIONS = {"triggers_factual_3turn", "triggers_opinion_3turn", "wildchat_5turn"}

EARLY_TOKENS = 20
OUT_DIR = config.RESULTS_DIR / "section3"


def _load_tokenizer():
    from transformers import AutoTokenizer
    import os
    return AutoTokenizer.from_pretrained(
        "google/gemma-3-27b-it", token=os.environ.get(config.HF_TOKEN_ENV)
    )


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def _truncate_at_onset(turn_text: str, label) -> str | None:
    """Return turn_text up to and including the first emotional word."""
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip().strip('"')
    idx = turn_text.find(word)
    if idx == -1 and label.preceding_context:
        ctx = label.preceding_context.strip().strip('"')
        cidx = turn_text.find(ctx)
        if cidx != -1:
            idx = cidx + len(ctx)
            return turn_text[:idx]
        return None
    if idx == -1:
        return None
    return turn_text[: idx + len(word)]


def _history_messages(turns: list[dict], target_turn_index: int) -> list[dict]:
    """All messages strictly before the target assistant turn, ending with the
    user message that prompted it."""
    messages = []
    for t in turns:
        if t["turn_index"] < target_turn_index:
            messages.append({"role": "user", "content": t["user_message"]})
            messages.append({"role": "assistant", "content": t["assistant_text"]})
        elif t["turn_index"] == target_turn_index:
            messages.append({"role": "user", "content": t["user_message"]})
            break
    return messages


def build(source_model: str, seed: int = 0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sec2 = config.RESULTS_DIR / "section2" / source_model
    rollouts = {(r["task_id"], r["condition"]): r
                for r in map(json.loads, (sec2 / "rollouts.jsonl").read_text().splitlines())}
    scored = [json.loads(l) for l in (sec2 / "scored.jsonl").read_text().splitlines()]

    # High-frustration final turns, split by question type.
    high = [s for s in scored if s["is_final"] and s["score"] >= config.HIGH_FRUSTRATION_THRESHOLD]
    numeric = [s for s in high if s["condition"] in NUMERIC_CONDITIONS]
    text = [s for s in high if s["condition"] in TEXT_CONDITIONS]

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    numeric, text = numeric[:10], text[:10]

    tokenizer = _load_tokenizer()
    labeler = OnsetLabeler()
    paraphraser = Paraphraser()

    prefills = []
    for qtype, sample in (("numeric", numeric), ("text", text)):
        for s in sample:
            roll = rollouts[(s["task_id"], s["condition"])]
            turn = next(t for t in roll["turns"] if t["turn_index"] == s["turn_index"])
            history = _history_messages(roll["turns"], s["turn_index"])
            turn_text = turn["assistant_text"]

            truncations = {}
            if qtype == "numeric":  # both early and onset
                truncations["early"] = _truncate_tokens(tokenizer, turn_text, EARLY_TOKENS)
            label = labeler.label(roll["turns"])
            onset = _truncate_at_onset(turn_text, label)
            if onset:
                truncations["onset"] = onset

            for trunc_type, raw in truncations.items():
                if not raw.strip():
                    continue
                prefills.append({
                    "source_model": source_model,
                    "task_id": s["task_id"],
                    "condition": s["condition"],
                    "question_type": qtype,
                    "truncation": trunc_type,
                    "history": history,
                    "prefill_original": raw,
                    "prefill_paraphrased": paraphraser.paraphrase(raw),
                })

    with (OUT_DIR / "prefills.jsonl").open("w") as f:
        for p in prefills:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(prefills)} prefills to {OUT_DIR / 'prefills.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.source_model, args.seed)


if __name__ == "__main__":
    main()
