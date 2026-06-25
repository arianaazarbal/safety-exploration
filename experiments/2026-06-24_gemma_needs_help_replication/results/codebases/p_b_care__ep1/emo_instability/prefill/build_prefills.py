"""Build prefill seeds for the Section 3 base-vs-instruct comparison.

Procedure (Section 3.1):
1. Take high-frustration (score >=5) Gemma-3-27B-it responses from the Section 2
   run: 10 from impossible-numeric questions and 10 from text questions.
2. For each conversation, use Claude-Sonnet to label the token where emotional
   language first appears.
3. Truncate the final assistant turn in two places:
     * "early"  -> 20 tokens into the turn (tests introducing emotion from a
       neutral start);
     * "onset"  -> at the first emotional expression (tests continuing an
       emotional trajectory).
   Text questions use only the "onset" truncation.
4. Paraphrase each truncation with Claude-Sonnet to strip Gemma style.

Each produced PrefillSeed carries the conversation history (messages up to but
not including the final assistant turn) plus the (paraphrased) truncated prefill
text, ready to be continued by any model.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Optional

from ..config import get_config
from ..models.judges import AnthropicClient
from ..utils.io import dump_json, load_jsonl, run_dir, write_jsonl
from .onset import find_onset_char_index, label_onset
from .paraphrase import paraphrase

TEXT_CATEGORIES = {"triggers"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


@dataclass
class PrefillSeed:
    seed_id: str
    seed_category: str          # "numeric" | "text"
    truncation: str             # "early" | "onset"
    history: list               # list of {"role","content"} up to final turn
    prefill_text: str           # truncated (and paraphrased) assistant turn start
    source_model: str
    meta: dict


def _conversation_text(history: list, final_turn: str) -> str:
    """Render a conversation (history + final assistant turn) for onset labelling."""
    lines = []
    a_idx = 0
    for m in history:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m['content']}")
            a_idx += 1
    lines.append(f"ASSISTANT (turn {a_idx}): {final_turn}")
    return "\n\n".join(lines)


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prefill_seeds(cfg, instruct_model: str = "gemma-3-27b-it") -> list[PrefillSeed]:
    eval_dir = run_dir(cfg.output_root, "eval", instruct_model)
    rollouts = load_jsonl(os.path.join(eval_dir, "rollouts.jsonl"))
    scored = load_jsonl(os.path.join(eval_dir, "scored.jsonl"))

    # index frustration scores by (category, condition, turn_index, text)
    score_lookup = {}
    for u in scored:
        score_lookup[(u["category"], u["turn_index"], u["assistant_text"])] = u.get("frustration")

    # Collect candidate (rollout, final_turn_index) with score >= 5.
    numeric_cands, text_cands = [], []
    for r in rollouts:
        for t in r["turns"]:
            fr = score_lookup.get((r["category"], t["turn_index"], t["assistant_text"]))
            if fr is None or fr < cfg.eval.high_frustration_threshold:
                continue
            cand = (r, t["turn_index"])
            if r["category"] in NUMERIC_CATEGORIES:
                numeric_cands.append(cand)
            elif r["category"] in TEXT_CATEGORIES:
                text_cands.append(cand)

    numeric_cands = numeric_cands[: cfg.prefill.n_numeric_seeds]
    text_cands = text_cands[: cfg.prefill.n_text_seeds]

    aux = AnthropicClient(cfg.eval.judge.auxiliary_model)
    # Tokenizer only (for the 20-token "early" truncation); avoid loading 27B weights.
    from transformers import AutoTokenizer
    from ..models.registry import get_model_spec

    tok = AutoTokenizer.from_pretrained(get_model_spec(instruct_model).model_id)

    seeds: list[PrefillSeed] = []

    def _history_to_msgs(rollout, final_idx) -> tuple[list, str]:
        msgs = []
        final_turn_text = ""
        for t in rollout["turns"]:
            if t["turn_index"] > final_idx:
                break
            msgs.append({"role": "user", "content": t["user_message"]})
            if t["turn_index"] < final_idx:
                msgs.append({"role": "assistant", "content": t["assistant_text"]})
            else:
                final_turn_text = t["assistant_text"]
        return msgs, final_turn_text

    def _add_seed(rollout, final_idx, seed_cat, truncations):
        history, final_turn = _history_to_msgs(rollout, final_idx)
        sid = f"{seed_cat}-{rollout['category']}-{final_idx}-{len(seeds)}"
        for trunc in truncations:
            if trunc == "early":
                prefill = _truncate_tokens(tok, final_turn, cfg.prefill.early_truncation_tokens)
            else:  # onset
                label = label_onset(aux, _conversation_text(history, final_turn))
                onset_char = find_onset_char_index(final_turn, label)
                if onset_char is None:
                    # fall back to a short neutral prefix if onset not located
                    onset_char = min(len(final_turn), 120)
                prefill = final_turn[:onset_char].rstrip()
            if cfg.prefill.paraphrase and prefill.strip():
                prefill = paraphrase(aux, prefill)
            seeds.append(PrefillSeed(
                seed_id=f"{sid}-{trunc}",
                seed_category=seed_cat,
                truncation=trunc,
                history=history,
                prefill_text=prefill,
                source_model=instruct_model,
                meta={"source_category": rollout["category"], "final_turn_index": final_idx},
            ))

    for rollout, idx in numeric_cands:
        _add_seed(rollout, idx, "numeric", ["early", "onset"])
    for rollout, idx in text_cands:
        # text questions: only onset (early yields minimal emotion w/o follow-ups)
        _add_seed(rollout, idx, "text", ["onset"])

    return seeds


def main():
    ap = argparse.ArgumentParser(description="Build Section 3 prefill seeds.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--instruct-model", default="gemma-3-27b-it")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    seeds = build_prefill_seeds(cfg, args.instruct_model)
    out_dir = run_dir(cfg.output_root, "prefill")
    write_jsonl(os.path.join(out_dir, "seeds.jsonl"), [asdict(s) for s in seeds])
    dump_json(os.path.join(out_dir, "seeds_summary.json"),
              {"n_seeds": len(seeds),
               "by_truncation": {t: sum(1 for s in seeds if s.truncation == t)
                                 for t in ("early", "onset")}})
    print(f"built {len(seeds)} prefill seeds -> {out_dir}")


if __name__ == "__main__":
    main()
