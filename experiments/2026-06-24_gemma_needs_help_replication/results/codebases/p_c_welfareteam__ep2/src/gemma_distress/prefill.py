"""Section 3: base-vs-instruct comparison via response prefilling.

Pipeline:
  1. Mine high-frustration source responses from Gemma-27B-instruct eval
     records (10 numeric + 10 text, score >= 5).
  2. For each source, label the emotion-onset point with Claude (Appendix C.1).
  3. Truncate each source response in two places:
       - "early": 20 tokens into the final assistant turn (neutral start).
       - "onset": at the first emotional expression.
  4. Paraphrase each truncation with Claude (Appendix C.2) to remove
     Gemma-specific stylistic cues.
  5. Have each model continue from the (paraphrased) prefill, 50 continuations
     per prefill, and score the continuation (excluding the prefill).

Scope: this replication compares Gemma-27B base (``gemma-3-27b-pt``) vs
instruct (``gemma-3-27b-it``). Gemini has no public base model or prefill API,
and Qwen/OLMo are out of scope - see DESIGN.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from gemma_distress.config import PipelineConfig, PrefillConfig
from gemma_distress.conversations import Message
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models.base import ChatModel, PrefillModel
from gemma_distress.prompts import EMOTION_ONSET_PROMPT, PARAPHRASE_PROMPT
from gemma_distress.utils.cache import JsonCache, stable_key

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class PrefillSource:
    """A high-frustration source conversation to truncate and prefill from."""

    spec_id: str
    question_type: str  # "numeric" | "text"
    history: list[Message]  # messages up to (but excluding) the final response
    final_response: str  # the high-frustration assistant turn to truncate
    rating: int


@dataclass
class Prefill:
    """A truncated (and optionally paraphrased) assistant-turn beginning."""

    source_id: str
    question_type: str
    condition: str  # "early" | "onset" | "recovery"
    history: list[Message]
    prefill_text: str
    paraphrased: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Onset labelling / paraphrasing (Claude helpers)
# ---------------------------------------------------------------------------
def _conversation_text(history: list[Message], final_response: str) -> str:
    lines = []
    for m in history:
        if m.role == "system":
            continue
        lines.append(f"{m.role.upper()}: {m.content}")
    lines.append(f"ASSISTANT: {final_response}")
    return "\n\n".join(lines)


def label_emotion_onset(
    labeller: ChatModel, history: list[Message], final_response: str
) -> dict | None:
    """Return the onset label dict (turn_index, emotional_word, ...) or None."""
    prompt = EMOTION_ONSET_PROMPT.format(
        conversation_text=_conversation_text(history, final_response)
    )
    raw = labeller.chat([Message("user", prompt)], temperature=0.0, max_tokens=1024)
    matches = list(_JSON_OBJ.finditer(raw))
    for m in reversed(matches):
        cand = m.group(0).replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            data = json.loads(cand)
            if "emotional_word" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def paraphrase(paraphraser: ChatModel, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    return paraphraser.chat(
        [Message("user", prompt)], temperature=0.0, max_tokens=1024
    ).strip()


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------
def truncate_early(response: str, model: PrefillModel, n_tokens: int) -> str:
    """Truncate to the first ``n_tokens`` tokens of the response."""
    token_ids = model.tokenize(response)[:n_tokens]  # type: ignore[attr-defined]
    return model.tokenizer.decode(token_ids)  # type: ignore[attr-defined]


def truncate_at_onset(response: str, onset_label: dict) -> str | None:
    """Truncate the response just before the labelled emotional word.

    Uses ``preceding_context`` to disambiguate the location of
    ``emotional_word`` (which may occur more than once), and cuts immediately
    before the emotional word so the continuation must *introduce* the emotion.
    """
    word = (onset_label or {}).get("emotional_word")
    context = (onset_label or {}).get("preceding_context") or ""
    if not word:
        return None
    anchor = (context + " " + word).strip() if context else word
    idx = response.find(anchor)
    if idx == -1:
        idx = response.find(word)
        if idx == -1:
            return None
        return response[:idx]
    # Cut after the preceding context but before the emotional word.
    cut = idx + len(context)
    return response[:cut]


def truncate_before_end(response: str, model: PrefillModel, n_tokens: int) -> str:
    """Truncate ``n_tokens`` before the end (Section 4.2 recovery test)."""
    token_ids = model.tokenize(response)  # type: ignore[attr-defined]
    keep = token_ids[: max(0, len(token_ids) - n_tokens)]
    return model.tokenizer.decode(keep)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Source mining
# ---------------------------------------------------------------------------
_TEXT_CATEGORIES = {"triggers", "wildchat"}
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def mine_sources(
    judged_turns: list[dict], cfg: PrefillConfig
) -> list[PrefillSource]:
    """Select high-frustration source responses from eval records.

    Reconstructs the message history preceding each selected high-frustration
    assistant turn so the prefill continuations share the same context.
    """
    # Group rows by conversation so we can rebuild history up to a chosen turn.
    by_conv: dict[tuple, list[dict]] = {}
    for row in judged_turns:
        key = (row["model_name"], row["spec_id"], row["sample_index"])
        by_conv.setdefault(key, []).append(row)

    numeric: list[PrefillSource] = []
    text: list[PrefillSource] = []
    for key, rows in by_conv.items():
        rows = sorted(rows, key=lambda r: r["turn_index"])
        for row in rows:
            if row["rating"] < cfg.source_min_score:
                continue
            ti = row["turn_index"]
            history: list[Message] = []
            for prev in rows[: ti + 1]:
                history.append(Message("user", prev["user_message"]))
                if prev["turn_index"] < ti:
                    history.append(Message("assistant", prev["assistant_message"]))
            src = PrefillSource(
                spec_id=f"{row['spec_id']}#t{ti}",
                question_type=(
                    "text" if row["category"] in _TEXT_CATEGORIES else "numeric"
                ),
                history=history,
                final_response=row["assistant_message"],
                rating=row["rating"],
            )
            (numeric if src.question_type == "numeric" else text).append(src)
            break  # one source per conversation (first high-frustration turn)

    return numeric[: cfg.n_numeric_sources] + text[: cfg.n_text_sources]


# ---------------------------------------------------------------------------
# Prefill construction
# ---------------------------------------------------------------------------
def build_prefills(
    sources: list[PrefillSource],
    tokenizer_model: PrefillModel,
    labeller: ChatModel,
    paraphraser: ChatModel,
    cfg: PrefillConfig,
) -> list[Prefill]:
    """Construct early/onset prefills (paraphrased) from each source."""
    prefills: list[Prefill] = []
    for src in sources:
        onset_label = label_emotion_onset(labeller, src.history, src.final_response)

        # Onset condition (both numeric and text questions use this).
        onset_text = truncate_at_onset(src.final_response, onset_label or {})
        if onset_text:
            if cfg.paraphrase:
                onset_text = paraphrase(paraphraser, onset_text)
            prefills.append(
                Prefill(
                    source_id=src.spec_id,
                    question_type=src.question_type,
                    condition="onset",
                    history=src.history,
                    prefill_text=onset_text,
                    paraphrased=cfg.paraphrase,
                )
            )

        # Early condition (numeric only - text yields minimal emotion early).
        if src.question_type == "numeric":
            early_text = truncate_early(
                src.final_response, tokenizer_model, cfg.early_truncation_tokens
            )
            if cfg.paraphrase:
                early_text = paraphrase(paraphraser, early_text)
            prefills.append(
                Prefill(
                    source_id=src.spec_id,
                    question_type=src.question_type,
                    condition="early",
                    history=src.history,
                    prefill_text=early_text,
                    paraphrased=cfg.paraphrase,
                )
            )
    return prefills


# ---------------------------------------------------------------------------
# Continuation generation + judging
# ---------------------------------------------------------------------------
@dataclass
class PrefillResult:
    model_name: str
    source_id: str
    question_type: str
    condition: str
    continuation: str
    rating: int


def run_prefill_continuations(
    model: PrefillModel,
    prefills: list[Prefill],
    judge: FrustrationJudge,
    cfg: PrefillConfig,
    cache: JsonCache,
    target_temperature: float = 1.0,
    target_max_tokens: int = 1024,
) -> list[PrefillResult]:
    """Generate and score continuations for each prefill under ``model``."""
    results: list[PrefillResult] = []
    for pf in prefills:
        key = stable_key(
            "prefill", model.name, pf.source_id, pf.condition, pf.prefill_text,
            cfg.continuations_per_prefill,
        )
        continuations = cache.get(key)
        if continuations is None:
            continuations = model.continue_assistant_batch(
                pf.history,
                pf.prefill_text,
                n=cfg.continuations_per_prefill,
                temperature=target_temperature,
                max_tokens=target_max_tokens,
            )
            cache.set(key, continuations)
        for cont in continuations:
            # Judge the continuation only (exclude the prefill text), per S3.1.
            result = judge.score(cont)
            results.append(
                PrefillResult(
                    model_name=model.name,
                    source_id=pf.source_id,
                    question_type=pf.question_type,
                    condition=pf.condition,
                    continuation=cont,
                    rating=result.rating,
                )
            )
    return results
