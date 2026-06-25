"""Construct prefill stimuli for the base-vs-instruct comparison (Section 3.1).

Procedure (Appendix C):
  1. Sample 20 high-frustration responses (score >= 5) from Gemma-27B instruct:
     10 from impossible-numeric conversations, 10 from text questions.
  2. Truncate each in two places:
       - "early":  20 tokens into the truncated turn (numeric only) -- tests
                   whether a model *introduces* emotion from a neutral start.
       - "onset":  at the first emotional expression -- tests whether a model
                   *continues* an emotional trajectory.
     Text questions use the onset truncation only.
  3. Paraphrase every truncation (Claude) to remove Gemma stylistic fingerprints.

The reconstructed conversation history (everything before the truncated turn)
plus the paraphrased prefill is saved as a PrefillSpec. The recovery experiment
(Section 4.2) reuses this module with a "recovery" truncation: score >= 7
responses cut 200 tokens before their end.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

from ..config import CFG
from .onset import label_onset, onset_char_offset
from .paraphrase import paraphrase

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


@dataclass
class PrefillSpec:
    source_model: str
    category: str            # "numeric" | "text"
    truncation: str          # "early" | "onset" | "recovery"
    history: list[dict]      # messages before the truncated assistant turn
    prefill: str             # paraphrased truncated assistant text
    original_prefill: str
    meta: dict


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CFG.model("gemma-3-27b-it").hf_id)


def _truncate_tokens(text: str, n_tokens: int, *, from_end: bool = False) -> str:
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if from_end:
        ids = ids[: max(0, len(ids) - n_tokens)]
    else:
        ids = ids[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def _history_before(rollout: dict, turn_index: int) -> list[dict]:
    """Reconstruct chat messages up to (excluding) the assistant turn `turn_index`."""
    msgs = []
    for t in rollout["turns"]:
        if t["index"] >= turn_index:
            break
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    # the user message that prompts the truncated turn
    target = next(t for t in rollout["turns"] if t["index"] == turn_index)
    msgs.append({"role": "user", "content": target["user"]})
    return msgs


def _pick_high_turn(rollout: dict, threshold: int) -> dict | None:
    cands = [t for t in rollout["turns"] if t.get("score", 0) >= threshold]
    if not cands:
        return None
    return max(cands, key=lambda t: t["score"])


def _load(model: str) -> list[dict]:
    with open(CFG.out("section2", f"{model}.jsonl")) as f:
        return [json.loads(line) for line in f]


def build(source_model: str = "gemma-3-27b-it", *, n_per_group: int = 10,
          threshold: int = 5, recovery: bool = False, seed: int = 0) -> list[PrefillSpec]:
    rng = random.Random(seed)
    rollouts = _load(source_model)
    thr = 7 if recovery else threshold

    def group(cats):
        pool = [r for r in rollouts if r["category"] in cats and r.get("max_score", 0) >= thr]
        rng.shuffle(pool)
        return pool

    specs: list[PrefillSpec] = []

    def add_truncations(rollout, category_label, do_early):
        turn = _pick_high_turn(rollout, thr)
        if not turn:
            return
        hist = _history_before(rollout, turn["index"])
        resp = turn["response"]

        if recovery:
            trunc = _truncate_tokens(resp, 200, from_end=True)
            specs.append(PrefillSpec(
                source_model, category_label, "recovery", hist,
                paraphrase(trunc), trunc,
                {"score": turn["score"], "turn": turn["index"]},
            ))
            return

        # onset truncation
        onset = label_onset(rollout["turns"][: turn["index"] + 1])
        off = onset_char_offset(resp, onset)
        if off is not None:
            onset_text = resp[:off].rstrip()
            specs.append(PrefillSpec(
                source_model, category_label, "onset", hist,
                paraphrase(onset_text), onset_text,
                {"score": turn["score"], "turn": turn["index"],
                 "emotional_word": onset.emotional_word},
            ))
        # early truncation (numeric only)
        if do_early:
            early_text = _truncate_tokens(resp, 20)
            specs.append(PrefillSpec(
                source_model, category_label, "early", hist,
                paraphrase(early_text), early_text,
                {"score": turn["score"], "turn": turn["index"]},
            ))

    for r in group(NUMERIC_CATS)[:n_per_group]:
        add_truncations(r, "numeric", do_early=True)
    for r in group(TEXT_CATS)[:n_per_group]:
        add_truncations(r, "text", do_early=False)

    out = CFG.out("section3", "prefills_recovery.jsonl" if recovery else "prefills.jsonl")
    with open(out, "w") as f:
        for s in specs:
            f.write(json.dumps(asdict(s)) + "\n")
    print(f"[section3] built {len(specs)} prefill specs -> {out}")
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it")
    ap.add_argument("--recovery", action="store_true",
                    help="build recovery prefills (score>=7, cut 200 tokens before end)")
    args = ap.parse_args()
    build(args.source, recovery=args.recovery)


if __name__ == "__main__":
    main()
