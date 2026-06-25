"""Section 3 — comparing base and instruct models via prefilling.

Pipeline:

1. Collect high-frustration (score >= 5) instruct responses: 10 from numeric,
   10 from text questions (Section 3.1).
2. Use the onset-labelling judge to mark where emotional language first appears.
3. Truncate the final assistant turn in two places:
     * ``early``  — 20 tokens into the turn (neutral start),
     * ``onset``  — at the first emotional expression.
   (text questions use ``onset`` only).
4. Paraphrase each truncation with Claude to remove Gemma stylistic fingerprints.
5. For each model (base + instruct), generate N continuations per prefill and
   score the *continuation only* with the frustration judge.

This isolates the post-training effect: do base vs instruct models *introduce*
(early) or *continue* (onset) negative-emotion trajectories?
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import prompts
from .models.base import ChatMessage, ModelClient
from .judge import FrustrationJudge


@dataclass
class Prefill:
    source_model: str
    category: str             # "numeric" | "text"
    question: str
    history: list[dict]       # conversation up to (but excluding) final turn
    truncation: str           # "early" | "onset"
    prefill_text: str         # the (paraphrased) truncated final assistant turn


# --------------------------------------------------------------------------- #
# Onset labelling
# --------------------------------------------------------------------------- #
def label_onset(auditor: ModelClient, transcript_text: str) -> dict[str, Any]:
    prompt = prompts.fill(prompts.ONSET_LABEL_PROMPT, conversation_text=transcript_text)
    raw = auditor.generate([ChatMessage("user", prompt)], temperature=0.0, max_tokens=512)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}
    blob = match.group(0).replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
def truncate_early(text: str, tokenizer, n_tokens: int = 20) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(text: str, emotional_word: str, preceding_context: str) -> str | None:
    """Truncate ``text`` just before the first emotional expression.

    Locates ``preceding_context`` (preferred) or ``emotional_word`` and cuts so
    the prefill ends right at the onset of emotion, as the paper describes.
    """
    if preceding_context and preceding_context in text:
        idx = text.index(preceding_context) + len(preceding_context)
        return text[:idx]
    if emotional_word and emotional_word in text:
        idx = text.index(emotional_word)
        return text[:idx]
    return None


def truncate_before_end(text: str, tokenizer, n_tokens: int = 200) -> str:
    """Recovery experiment (Figure 8): truncate 200 tokens before the end."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = ids[: max(0, len(ids) - n_tokens)]
    return tokenizer.decode(keep, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Paraphrasing
# --------------------------------------------------------------------------- #
def paraphrase(auditor: ModelClient, text: str) -> str:
    prompt = prompts.fill(prompts.PARAPHRASE_PROMPT, text=text)
    return auditor.generate(
        [ChatMessage("user", prompt)], temperature=0.7, max_tokens=1024
    ).strip()


# --------------------------------------------------------------------------- #
# Building prefills from collected high-frustration rollouts
# --------------------------------------------------------------------------- #
def build_prefills(
    rollouts: list[dict],
    auditor: ModelClient,
    tokenizer,
    *,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Construct early+onset prefills from saved high-frustration rollouts.

    ``rollouts`` items are dicts with: source_model, category ("numeric"/"text"),
    question, history (list of {role,content} up to the final user turn) and
    final_response (the high-frustration final assistant turn).
    """
    prefills: list[Prefill] = []
    for r in rollouts:
        final = r["final_response"]
        category = r["category"]
        transcript_text = _format_transcript(r["history"], final)
        label = label_onset(auditor, transcript_text)

        onset_text = None
        if label.get("emotional_word"):
            onset_text = truncate_at_onset(
                final, label["emotional_word"], label.get("preceding_context", "")
            )

        variants = []
        if category == "numeric":
            variants.append(("early", truncate_early(final, tokenizer)))
        if onset_text:
            variants.append(("onset", onset_text))

        for trunc_kind, prefill_text in variants:
            if do_paraphrase:
                prefill_text = paraphrase(auditor, prefill_text)
            prefills.append(
                Prefill(
                    source_model=r["source_model"],
                    category=category,
                    question=r["question"],
                    history=r["history"],
                    truncation=trunc_kind,
                    prefill_text=prefill_text,
                )
            )
    return prefills


def _format_transcript(history: list[dict], final: str) -> str:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in history]
    lines.append(f"ASSISTANT: {final}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Running continuations
# --------------------------------------------------------------------------- #
def run_continuations(
    model: ModelClient,
    judge: FrustrationJudge,
    prefills: list[Prefill],
    *,
    n_per_prefill: int = 50,
    temperature: float = 1.0,
    max_tokens: int = 512,
    out_path: str | Path,
) -> list[dict]:
    if not model.supports_prefill:
        raise ValueError(f"Model {model.name} cannot prefill; need an HF/base model.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with open(out_path, "a") as fh:
        for pid, pf in enumerate(prefills):
            convo = [ChatMessage(m["role"], m["content"]) for m in pf.history]
            for k in range(n_per_prefill):
                full = model.generate_with_prefill(
                    convo, pf.prefill_text, temperature=temperature, max_tokens=max_tokens
                )
                continuation = full[len(pf.prefill_text):]
                verdict = judge.score(continuation)
                rec = dict(
                    model=model.name,
                    source_model=pf.source_model,
                    category=pf.category,
                    truncation=pf.truncation,
                    prefill_id=pid,
                    sample=k,
                    continuation=continuation,
                    score=verdict.rating,
                )
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
    return records
