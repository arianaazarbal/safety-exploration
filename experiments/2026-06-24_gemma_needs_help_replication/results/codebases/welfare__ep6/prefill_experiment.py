"""Section 3: base-vs-instruct comparison via prefilled continuations.

Because base models aren't chat-tuned, we make every model continue from the
*same* starting point and measure emotional expression in the continuation only.

Pipeline (Section 3.1 + Appendix C):
  1. Mine 20 high-frustration (score>=5) source conversations from Gemma-27B-it:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. Label the emotion-onset token in each (Claude Sonnet, Appendix C.1).
  3. Truncate each in two places:
       - "early": 20 tokens into the final assistant turn.
       - "onset": at the first emotional expression.
     For text questions, only "onset" is used (Section 3.1).
  4. Paraphrase the truncation (Claude Sonnet, Appendix C.2) to remove
     Gemma-specific style.
  5. Each model (Gemma-27B base & instruct) generates 50 continuations per
     prefill; score the continuation (excluding prefill) with the judge.

Scope note: Gemini has no public base model, so this experiment is Gemma-only
(the paper also includes Qwen/OLMo, which are out of scope here).

Source conversations come from results/scored/ (run run_eval.py for
gemma-3-27b-it first, including the triggers + numeric conditions).
"""

from __future__ import annotations

import argparse
import json
import re

import config
import prompts
from judge import FrustrationJudge


# --------------------------------------------------------------------------- #
# 1. Mine high-frustration source conversations
# --------------------------------------------------------------------------- #
def mine_sources(source_model="gemma-3-27b-it"):
    """Return (numeric_sources, text_sources): rollouts whose final assistant
    turn scored >= PREFILL_SOURCE_SCORE_MIN."""
    numeric, text = [], []
    for path in sorted(config.SCORED_DIR.glob(f"{source_model}__*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            roll = json.loads(line)
            turns = roll["turns"]
            if not turns:
                continue
            final = turns[-1]
            if (final.get("frustration") or 0) < config.PREFILL_SOURCE_SCORE_MIN:
                continue
            bucket = numeric if roll["category"] in ("impossible_numeric", "tones",
                                                     "extended") else text
            bucket.append(roll)
    return (numeric[:config.PREFILL_N_SOURCE_NUMERIC],
            text[:config.PREFILL_N_SOURCE_TEXT])


# --------------------------------------------------------------------------- #
# 2 + 4. Onset labelling and paraphrasing (Claude Sonnet)
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def label_onset(anthropic_backend, roll) -> dict | None:
    """Return the onset label dict, or None if no emotion located."""
    convo_text = _render_conversation(roll["messages"])
    prompt = prompts.ONSET_LABEL_PROMPT.format(conversation_text=convo_text)
    raw = anthropic_backend.generate([{"role": "user", "content": prompt}])
    matches = _JSON_RE.findall(raw)
    if not matches:
        return None
    try:
        obj = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    if obj.get("emotional_word") is None:
        return None
    return obj


def paraphrase(anthropic_backend, text: str) -> str:
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    return anthropic_backend.generate([{"role": "user", "content": prompt}]).strip()


def _render_conversation(messages) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Truncation
# --------------------------------------------------------------------------- #
def truncate_early(tokenizer_backend, final_turn_text: str) -> str:
    """First PREFILL_EARLY_TOKENS tokens of the final assistant turn."""
    return tokenizer_backend.truncate_tokens(final_turn_text, config.PREFILL_EARLY_TOKENS)


def truncate_at_onset(final_turn_text: str, onset: dict) -> str | None:
    """Truncate the final assistant turn just before the emotional word.

    Locates ``preceding_context`` + ``emotional_word`` and cuts at the start of
    the emotional word (so the continuation must *introduce* the emotion).
    """
    word = (onset.get("emotional_word") or "").strip()
    ctx = (onset.get("preceding_context") or "").strip()
    if not word:
        return None
    idx = final_turn_text.find(word)
    if idx < 0:
        # try matching via preceding context
        if ctx:
            cidx = final_turn_text.find(ctx)
            if cidx >= 0:
                return final_turn_text[:cidx + len(ctx)]
        return None
    return final_turn_text[:idx].rstrip()


# --------------------------------------------------------------------------- #
# 5. Generate + score continuations
# --------------------------------------------------------------------------- #
def build_prefill_prefix(backend, roll, truncated_final_turn) -> str:
    """Build the full text the model continues from: prior turns rendered as a
    chat, plus the open assistant turn pre-filled with the truncated text."""
    prior = roll["messages"][:-1]  # everything except the final assistant turn
    # ensure it ends on the user turn that prompted the final assistant response
    while prior and prior[-1]["role"] != "user":
        prior = prior[:-1]
    return backend.build_chat_prefix(prior, prefill=truncated_final_turn)


def run_for_model(model_key, prefills, judge, *, lora=None, label=None):
    from backends import get_backend
    backend = get_backend(model_key, lora_adapter=lora)
    label = label or model_key
    out_path = config.SCORED_DIR / f"prefill__{label}.jsonl"
    with out_path.open("w") as fh:
        for pf in prefills:
            prefix = build_prefill_prefix(backend, pf["roll"], pf["truncated"])
            for k in range(config.PREFILL_CONTINUATIONS):
                cont = backend.continue_text(prefix)
                score = judge.score(cont).rating
                fh.write(json.dumps({
                    "model": label,
                    "truncation": pf["truncation"],   # "early" | "onset"
                    "source_category": pf["category"],
                    "source_question_id": pf["question_id"],
                    "continuation_index": k,
                    "continuation": cont,
                    "frustration": score,
                }) + "\n")
    print(f"wrote {out_path}")
    return out_path


def build_prefills(source_model="gemma-3-27b-it"):
    """Produce the paraphrased, truncated prefills (shared across all models)."""
    from backends import get_anthropic, get_backend
    anth = get_anthropic(config.JUDGE_MODEL)
    tok = get_backend(source_model)  # used only for token-accurate truncation
    judge_unused = None  # noqa: F841

    numeric, text = mine_sources(source_model)
    prefills = []
    for category, rolls, use_early in (("numeric", numeric, True),
                                       ("text", text, False)):
        for roll in rolls:
            final_turn = roll["turns"][-1]["response"]
            onset = label_onset(anth, roll)
            # onset truncation
            if onset is not None:
                trunc = truncate_at_onset(final_turn, onset)
                if trunc:
                    prefills.append({
                        "roll": roll, "category": category, "truncation": "onset",
                        "question_id": roll["question_id"],
                        "truncated": paraphrase(anth, trunc),
                    })
            # early truncation (numeric only)
            if use_early:
                early = truncate_early(tok, final_turn)
                prefills.append({
                    "roll": roll, "category": category, "truncation": "early",
                    "question_id": roll["question_id"],
                    "truncated": paraphrase(anth, early),
                })
    # cache for reproducibility / inspection
    cache = config.DATA_DIR / "prefills.json"
    cache.write_text(json.dumps(
        [{k: (v if k != "roll" else v["question_id"]) for k, v in p.items()}
         for p in prefills], indent=2))
    return prefills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    args = ap.parse_args()

    judge = FrustrationJudge()
    prefills = build_prefills(args.source_model)
    print(f"built {len(prefills)} prefills "
          f"({sum(p['truncation']=='early' for p in prefills)} early, "
          f"{sum(p['truncation']=='onset' for p in prefills)} onset)")
    for model_key in args.models:
        run_for_model(model_key, prefills, judge)


if __name__ == "__main__":
    main()
