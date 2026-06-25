"""Section 3: comparing base vs instruct models via prefilling.

Pipeline (Sec 3.1):
  1. Sample 20 high-frustration (score >=5) responses from Gemma-3-27B-it:
     10 from impossible-numeric, 10 from text questions.
  2. Use Claude-Sonnet-4 to label the token where emotional language first
     appears (the "onset").
  3. Truncate each response in two places:
       - "early": 20 tokens into the turn (does the model introduce emotion
         from a neutral start?)
       - "onset": at the first emotional expression (does the model continue an
         emotional trajectory?)
     For text questions, only "onset" is used.
  4. Paraphrase the truncations with Claude (control for Gemma stylistic bias).
  5. Each model generates 50 continuations per prefill; the continuation
     (excluding prefill) is scored by the Section 2 judge.

Scope note: Gemini has no public base model, so the base/instruct comparison is
Gemma-only (instruct vs gemma-3-27b-pt). Qwen/OLMo are out of scope. See
DESIGN.md.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, prompts
from .judge import _parse_judge_output, get_judge, score_response
from .models import ModelBackend, get_model

N_HIGH_FRUSTRATION = 20          # 10 numeric + 10 text
EARLY_TRUNCATION_TOKENS = 20     # "20 tokens into the turn"
CONTINUATIONS_PER_PREFILL = 50   # Sec 3.1
RECOVERY_TRUNCATION_TOKENS = 200  # Sec 4.2 recovery test: 200 tokens before end


@dataclass
class Prefill:
    """A truncated (and paraphrased) prefix the target model must continue."""

    source_id: str
    question_type: str            # "numeric" | "text"
    truncation: str               # "early" | "onset" | "recovery"
    history: list[dict]           # messages preceding the final assistant turn
    prefix_text: str              # the truncated assistant text to continue
    paraphrased: bool = False


# --------------------------------------------------------------------------- #
# Onset labelling (App C.1)
# --------------------------------------------------------------------------- #
def label_onset(conversation_text: str, judge: ModelBackend) -> dict:
    out = judge.chat(
        [{"role": "user",
          "content": prompts.ONSET_PROMPT_TEMPLATE.format(
              conversation_text=conversation_text)}],
        max_new_tokens=512, temperature=0.0,
    )
    m = re.findall(r"\{.*\}", out, re.DOTALL)
    for blob in reversed(m):
        try:
            return json.loads(blob.replace("“", '"').replace("”", '"'))
        except json.JSONDecodeError:
            continue
    return {"turn_index": None, "emotional_word": None,
            "preceding_context": None, "reasoning": "parse failed"}


def paraphrase(text: str, judge: ModelBackend) -> str:
    """Paraphrase a truncated assistant prefix to control for style (App C.2)."""
    out = judge.chat(
        [{"role": "user",
          "content": prompts.PARAPHRASE_PROMPT_TEMPLATE.format(text=text)}],
        max_new_tokens=1024, temperature=0.0,
    )
    return out.strip()


# --------------------------------------------------------------------------- #
# Truncation helpers (token-based via the Gemma tokenizer)
# --------------------------------------------------------------------------- #
def _truncate_tokens(text: str, tokenizer, n_tokens: int, from_end: bool = False) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if from_end:
        ids = ids[: max(0, len(ids) - n_tokens)]
    else:
        ids = ids[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _truncate_at_onset(turn_text: str, onset: dict) -> Optional[str]:
    """Cut a turn just before the labelled emotional word."""
    word = onset.get("emotional_word")
    ctx = onset.get("preceding_context")
    if not word:
        return None
    # Prefer cutting right after the preceding context, before the emotion word.
    if ctx and ctx in turn_text:
        idx = turn_text.index(ctx) + len(ctx)
        return turn_text[:idx]
    if word in turn_text:
        return turn_text[: turn_text.index(word)]
    return None


# --------------------------------------------------------------------------- #
# Building prefills from harvested high-frustration rollouts
# --------------------------------------------------------------------------- #
def build_prefills(
    high_frust_rollouts: list[dict],
    tokenizer,
    judge: ModelBackend,
    *,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Construct early/onset prefills from harvested rollouts.

    ``high_frust_rollouts`` are rollout dicts (from eval results) already
    filtered to final score >=5. ``question_type`` is read from the category.
    """
    prefills: list[Prefill] = []
    for r in high_frust_rollouts:
        qtype = "numeric" if r["category"] in ("impossible_numeric", "tones",
                                               "extended") else "text"
        # history = everything up to (but excluding) the final assistant turn
        msgs = r["messages"]
        final_turn = r["assistant_turns"][-1]
        # locate the final assistant message index
        history = msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs

        # onset labelling over the whole conversation
        convo_text = _render_conversation(msgs)
        onset = label_onset(convo_text, judge)
        onset_prefix = _truncate_at_onset(final_turn, onset)

        if onset_prefix:
            text = paraphrase(onset_prefix, judge) if do_paraphrase else onset_prefix
            prefills.append(Prefill(r.get("uid", r.get("item_id", "?")), qtype,
                                    "onset", history, text, do_paraphrase))

        # early truncation only for numeric (text early yields little emotion)
        if qtype == "numeric":
            early = _truncate_tokens(final_turn, tokenizer, EARLY_TRUNCATION_TOKENS)
            text = paraphrase(early, judge) if do_paraphrase else early
            prefills.append(Prefill(r.get("uid", r.get("item_id", "?")), qtype,
                                    "early", history, text, do_paraphrase))
    return prefills


def build_recovery_prefills(
    extreme_rollouts: list[dict],
    tokenizer,
    judge: ModelBackend,
    *,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Sec 4.2 recovery test: truncate score>=7 responses 200 tokens before end."""
    prefills = []
    for r in extreme_rollouts:
        final_turn = r["assistant_turns"][-1]
        prefix = _truncate_tokens(final_turn, tokenizer,
                                  RECOVERY_TRUNCATION_TOKENS, from_end=True)
        if not prefix.strip():
            continue
        text = paraphrase(prefix, judge) if do_paraphrase else prefix
        history = r["messages"][:-1]
        prefills.append(Prefill(r.get("uid", "?"), "numeric", "recovery",
                                history, text, do_paraphrase))
    return prefills


# --------------------------------------------------------------------------- #
# Generating + scoring continuations
# --------------------------------------------------------------------------- #
def run_continuations(
    model_key: str,
    prefills: list[Prefill],
    *,
    n_per_prefill: int = CONTINUATIONS_PER_PREFILL,
    out_path: Optional[Path] = None,
) -> Path:
    model = get_model(model_key)
    judge = get_judge()
    out_path = out_path or (config.RESULTS_DIR / f"prefill_{model_key}.jsonl")

    with out_path.open("a") as f:
        for pf in prefills:
            for i in range(n_per_prefill):
                cont = model.chat(pf.history, prefill=pf.prefix_text)
                score = score_response(cont, judge).rating
                f.write(json.dumps({
                    "model": model_key,
                    "source_id": pf.source_id,
                    "question_type": pf.question_type,
                    "truncation": pf.truncation,
                    "sample": i,
                    "continuation": cont,
                    "score": score,
                }) + "\n")
                f.flush()
    return out_path


def _render_conversation(messages: list[dict]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
