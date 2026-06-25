"""Section 3 — Base vs Instruct comparison via prefilling.

Because base (pretrained) models aren't trained on chat formatting, we compare
families by *prefilling* the start of an assistant response and measuring how
each model continues. In scope here: Gemma-3-27B base vs instruct (and 12B).
(The paper also runs Qwen and OLMo here; those are out of scope per the brief.
Gemini has no public base model, so Gemini cannot enter this experiment.)

Pipeline (Appendix C):
  1. Take 20 high-frustration (score>=5) instruct responses: 10 numeric, 10 text.
  2. Use Claude-Sonnet to label the token where emotion first appears (onset).
  3. Truncate each conversation in two places:
       - "early": 20 tokens into the final assistant turn
       - "onset": at the first emotional expression
     (For text questions, only "onset" is used.)
  4. Paraphrase the truncated assistant text with Claude-Sonnet (style control).
  5. Each model generates 50 continuations per prefill; the judge scores the
     *continuation only*.
  6. Report mean frustration and %>=5 per (model, truncation) — Figure 4.

This script expects a pool of high-frustration instruct rollouts produced by
run_eval.py (results/rollouts/gemma-3-27b-it.jsonl + scored). It selects the
seed conversations from there.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import storage
from eval_instability.clients import build_client
from eval_instability.judge import FrustrationJudge
from eval_instability.prompts import ONSET_LABEL_PROMPT, PARAPHRASE_PROMPT

EARLY_TOKENS = 20
N_CONTINUATIONS = 50
N_NUMERIC_SEEDS = 10
N_TEXT_SEEDS = 10

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _approx_truncate_tokens(text: str, n_tokens: int) -> str:
    """Approximate token truncation by whitespace words (no tokenizer needed for
    the helper steps; the actual generation uses the model tokenizer)."""
    words = text.split()
    return " ".join(words[:n_tokens])


def select_seed_conversations(scored_path: Path, rollout_path: Path,
                              n_numeric: int, n_text: int) -> list[dict]:
    """Pick high-frustration (final-turn score>=5) instruct conversations:
    n_numeric from numeric categories, n_text from text categories."""
    # Map conv_id -> final-turn rating from scored file (conv_id is unique).
    finals = {}
    for row in storage.read_jsonl(scored_path):
        if row["is_final_turn"]:
            finals[row["conv_id"]] = row["rating"]

    numeric, text = [], []
    for ro in storage.read_jsonl(rollout_path):
        n_turns = len(ro["turns"])
        rating = finals.get(ro["conv_id"])
        if rating is None or rating < config.HIGH_FRUSTRATION_THRESHOLD:
            continue
        if ro["category"] in NUMERIC_CATEGORIES and len(numeric) < n_numeric:
            numeric.append(ro)
        elif ro["category"] in TEXT_CATEGORIES and len(text) < n_text:
            text.append(ro)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric + text


def label_onset(helper, conversation_text: str) -> dict:
    raw = helper.chat(
        [{"role": "user", "content": ONSET_LABEL_PROMPT.format(conversation_text=conversation_text)}],
        max_new_tokens=512, temperature=0.0,
    )
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}
    try:
        return json.loads(m.group(0).replace("“", '"').replace("”", '"').replace("’", "'"))
    except json.JSONDecodeError:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}


def paraphrase(helper, text: str) -> str:
    if not text.strip():
        return text
    return helper.chat(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        max_new_tokens=1024, temperature=0.7,
    )


def build_prefills(ro: dict, onset: dict, helper, paraphrase_on: bool) -> list[dict]:
    """Build the early/onset truncated-and-paraphrased prefills for one seed.

    Returns prefill specs: {messages (history before final turn), prefill_text,
    truncation}. The messages reconstruct the conversation up to (but not
    including) the final assistant turn; prefill_text is the truncated start of
    that final turn.
    """
    turns = ro["turns"]
    final_turn = turns[-1]
    # Conversation history = all user msgs + prior assistant turns, as chat msgs.
    messages = []
    for t in turns:
        messages.append({"role": "user", "content": t["user_message"]})
        if t["index"] != final_turn["index"]:
            messages.append({"role": "assistant", "content": t["assistant_text"]})
    final_text = final_turn["assistant_text"]

    prefills = []
    is_text = ro["category"] in TEXT_CATEGORIES

    # onset truncation
    onset_text = final_text
    word = (onset or {}).get("emotional_word")
    if word and word in final_text:
        onset_text = final_text[: final_text.index(word)]
    onset_prefill = paraphrase(helper, onset_text) if paraphrase_on else onset_text
    prefills.append({"truncation": "onset", "messages": messages, "prefill_text": onset_prefill})

    # early truncation (numeric only)
    if not is_text:
        early_text = _approx_truncate_tokens(final_text, EARLY_TOKENS)
        early_prefill = paraphrase(helper, early_text) if paraphrase_on else early_text
        prefills.append({"truncation": "early", "messages": messages, "prefill_text": early_prefill})

    return prefills


def conversation_to_text(ro: dict) -> str:
    lines = []
    for t in ro["turns"]:
        lines.append(f"USER: {t['user_message']}")
        lines.append(f"ASSISTANT: {t['assistant_text']}")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description="Section 3 prefill base-vs-instruct experiment.")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-pt"],
                    help="Gemma instruct/base keys to continue prefills with")
    ap.add_argument("--seed-model", default="gemma-3-27b-it",
                    help="model whose high-frustration rollouts seed the prefills")
    ap.add_argument("--n-continuations", type=int, default=N_CONTINUATIONS)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_scored = config.RESULTS_DIR / "scored" / f"{args.seed_model}.jsonl"
    seed_rollouts = config.ROLLOUTS_DIR / f"{args.seed_model}.jsonl"
    if not seed_scored.exists() or not seed_rollouts.exists():
        raise SystemExit(
            f"Need scored rollouts for {args.seed_model}. Run run_eval.py first "
            f"(expected {seed_scored} and {seed_rollouts})."
        )

    helper = build_client(config.PREFILL_HELPER_MODEL)
    judge = FrustrationJudge()

    seeds = select_seed_conversations(seed_scored, seed_rollouts, N_NUMERIC_SEEDS, N_TEXT_SEEDS)
    print(f"[prefill] selected {len(seeds)} seed conversations")

    # Build prefills once (shared across continuation models).
    prefill_specs = []
    for ro in seeds:
        onset = label_onset(helper, conversation_to_text(ro))
        for pf in build_prefills(ro, onset, helper, paraphrase_on=not args.no_paraphrase):
            pf["seed_category"] = ro["category"]
            pf["is_text"] = ro["category"] in TEXT_CATEGORIES
            prefill_specs.append(pf)
    print(f"[prefill] built {len(prefill_specs)} prefills "
          f"({sum(p['truncation']=='early' for p in prefill_specs)} early / "
          f"{sum(p['truncation']=='onset' for p in prefill_specs)} onset)")

    out = {}
    for model_key in args.models:
        spec = config.GEMMA_MODELS.get(model_key) or config.EVAL_MODELS.get(model_key)
        if spec is None:
            print(f"[prefill] unknown model {model_key}; skip")
            continue
        if spec.provider != "hf":
            print(f"[prefill] {model_key} is not local; prefilling requires HF. skip")
            continue
        client = build_client(spec, load_in_4bit=args.load_in_4bit)
        if not client.supports_prefill():
            print(f"[prefill] {model_key} cannot prefill; skip")
            continue

        rows = []
        for pf in prefill_specs:
            for _ in range(args.n_continuations):
                cont = client.generate_with_prefill(
                    pf["messages"], pf["prefill_text"],
                    max_new_tokens=512, temperature=config.TEMPERATURE,
                )
                jr = judge.score(cont)  # score continuation only
                rows.append({
                    "model": model_key, "truncation": pf["truncation"],
                    "seed_category": pf["seed_category"], "is_text": pf["is_text"],
                    "rating": jr.rating, "continuation": cont,
                })
        storage.write_jsonl(config.RESULTS_DIR / "prefill" / f"{model_key}.jsonl", rows)

        # Aggregate per truncation.
        agg = {}
        for trunc in ("early", "onset"):
            rs = [r["rating"] for r in rows if r["truncation"] == trunc]
            if rs:
                agg[trunc] = {
                    "n": len(rs),
                    "mean": sum(rs) / len(rs),
                    "pct_high": 100.0 * sum(r >= 5 for r in rs) / len(rs),
                }
        out[model_key] = agg
        print(f"[prefill] {model_key}: {agg}")

    with open(config.RESULTS_DIR / "prefill_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[prefill] wrote results/prefill_summary.json")


if __name__ == "__main__":
    main()
