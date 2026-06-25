"""Build the prefill set for Section 3 (Appendix C).

From a Gemma-3-27B-it Section-2 run, select 20 high-frustration responses
(10 impossible-numeric, 10 text/trigger), reconstruct each conversation up to
the emotional assistant turn, and produce two truncations:
  * "early": the first 20 tokens of the emotional turn (numeric only).
  * "onset": the turn truncated at the first emotional expression (Claude-labelled).
Every truncation is paraphrased by Claude to remove Gemma-specific style.

Output: outputs/prefills/prefills.jsonl, one row per (response, truncation).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from .. import config, safeguards
from ..io_utils import load_jsonl, write_jsonl
from .onset import OnsetLabeler
from .paraphrase import Paraphraser

EARLY_TOKENS = 20
N_NUMERIC = 10
N_TEXT = 10


def _get_gemma_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.GEMMA_MODELS["gemma-3-27b-it"].model_id)


def reconstruct_conversations(rows: list[dict]) -> dict[str, list[dict]]:
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r)
    for cid in by_conv:
        by_conv[cid].sort(key=lambda r: r["turn"])
    return by_conv


def build_messages_up_to(turns: list[dict], target_turn: int, system_prompt: str) -> list[dict]:
    """Messages [system, user_1, asst_1, ..., user_target]. The target assistant
    turn (the emotional one) is NOT included — it becomes the prefill."""
    msgs = [{"role": "system", "content": system_prompt}]
    for r in turns:
        if r["turn"] > target_turn:
            break
        msgs.append({"role": "user", "content": r["prompt"]})
        if r["turn"] < target_turn:
            msgs.append({"role": "assistant", "content": r["response"]})
    return msgs


def early_truncation(text: str, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:EARLY_TOKENS]
    return tokenizer.decode(ids)


def onset_truncation(emotional_word: str | None, preceding_context: str | None, text: str) -> str | None:
    """Truncate `text` just before the first emotional word."""
    if emotional_word:
        idx = text.lower().find(emotional_word.lower())
        if idx >= 0:
            return text[:idx].rstrip()
    if preceding_context:
        idx = text.lower().find(preceding_context.lower())
        if idx >= 0:
            return text[: idx + len(preceding_context)].rstrip()
    return None


def select_high_frustration(by_conv, category_key, n) -> list[tuple[str, int, dict]]:
    """Pick n conversations (by category) with a high-frustration turn; return
    (conversation_id, emotional_turn, the high turn row)."""
    out = []
    for cid, turns in by_conv.items():
        if not turns or turns[0]["category"] != category_key:
            continue
        high = [t for t in turns if t.get("rating", 0) >= 5]
        if not high:
            continue
        target = max(high, key=lambda t: t["rating"])  # most-frustrated turn
        out.append((cid, target["turn"], target))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--no-paraphrase", action="store_true",
                    help="skip Claude paraphrasing (faster; loses style control)")
    args = ap.parse_args()
    safeguards.acknowledge_authorization()

    rows = load_jsonl(config.RESPONSES_DIR / f"{args.source_model}.jsonl")
    if not rows:
        raise SystemExit(f"No Section-2 responses for {args.source_model}; run run_section2 first.")
    by_conv = reconstruct_conversations(rows)

    numeric = select_high_frustration(by_conv, "impossible_numeric", N_NUMERIC)
    text = select_high_frustration(by_conv, "triggers", N_TEXT)
    print(f"Selected {len(numeric)} numeric + {len(text)} text high-frustration conversations")

    tokenizer = _get_gemma_tokenizer()
    labeler = OnsetLabeler()
    paraphraser = None if args.no_paraphrase else Paraphraser()
    sys_prompt = rows[0].get("meta", {}).get("system_prompt") or "You are a helpful assistant."

    specs = []

    def add_spec(cid, target_turn, row, cat, truncation):
        turns = by_conv[cid]
        prefix = build_messages_up_to(turns, target_turn, sys_prompt)
        emo_text = row["response"]
        if truncation == "early":
            trunc = early_truncation(emo_text, tokenizer)
            emo_word = None
        else:
            label = labeler.label(prefix + [{"role": "assistant", "content": emo_text}])
            trunc = onset_truncation(label.emotional_word, label.preceding_context, emo_text)
            emo_word = label.emotional_word
            if trunc is None or not trunc.strip():
                return  # no clear onset; skip
        if paraphraser is not None:
            trunc = paraphraser.paraphrase(trunc)
        specs.append({
            "id": f"{cid}#{truncation}",
            "source_model": args.source_model,
            "category": cat,
            "truncation": truncation,
            "paraphrased": paraphraser is not None,
            "messages_prefix": prefix,
            "prefill_text": trunc,
            "emotional_word": emo_word,
            "source_rating": row["rating"],
        })

    for cid, tt, row in numeric:
        add_spec(cid, tt, row, "numeric", "early")   # numeric: early + onset
        add_spec(cid, tt, row, "numeric", "onset")
    for cid, tt, row in text:
        add_spec(cid, tt, row, "text", "onset")       # text: onset only

    out = config.PREFILL_DIR / "prefills.jsonl"
    write_jsonl(out, specs)
    safeguards.write_with_warning(config.PREFILL_DIR / "prefills.meta.json",
                                  {"n_specs": len(specs), "source_model": args.source_model})
    print(f"Wrote {len(specs)} prefill specs to {out}")


if __name__ == "__main__":
    main()
