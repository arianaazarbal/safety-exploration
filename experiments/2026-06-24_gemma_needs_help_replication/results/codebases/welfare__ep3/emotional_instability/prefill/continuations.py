"""Base-vs-instruct continuation generation and scoring (Section 3.1–3.2).

Procedure:
  1. Sample 20 high-frustration (score >= 5) responses from Gemma-27B instruct:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, locate the emotion onset (Appendix C.1) and build two prefills:
       * "early" : truncate 20 tokens into the assistant turn (neutral start);
       * "onset" : truncate at the first emotional expression.
     Text questions use the "onset" truncation only.
  3. Paraphrase each truncation (Appendix C.2).
  4. Each of the six models (base+instruct Gemma/Qwen/OLMo — here scoped to
     Gemma base+instruct) generates 50 continuations per prefill; score the
     continuation (excluding the prefill) with the Section 2 judge.

Scope note: the replication brief restricts us to Gemma+Gemini. Gemini is closed
(no base model, no prefill API), so the base-vs-instruct study runs on the Gemma
family only: Gemma-3-27B-pt (base) vs Gemma-3-27B-it (instruct). The harness is
written generically so Qwen/OLMo specs could be added later.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from tqdm import tqdm

from ..backends import ChatMessage, get_backend
from ..config import DATA_DIR, GEMMA_27B_IT, GEMMA_27B_PT, ModelSpec
from ..eval.judge import score_response
from .onset import label_onset, onset_truncation_point
from .paraphrase import paraphrase

EARLY_TRUNCATION_TOKENS = 20      # ~chars proxy below; see note
N_CONTINUATIONS = 50


@dataclass
class Prefill:
    source_category: str          # "impossible_numeric" | "triggers"/"text"
    condition: str                # "early" | "onset"
    history: list[dict]           # prior turns (role/content) up to assistant turn
    prefill_text: str             # paraphrased truncated assistant text
    meta: dict = field(default_factory=dict)


def _approx_truncate_tokens(text: str, n_tokens: int) -> str:
    """Approximate an n-token prefix by whitespace words (no tokenizer needed in
    the API-only path). DESIGN.md notes this approximation; the local backend
    path can use the real tokenizer."""
    words = text.split()
    return " ".join(words[: max(1, n_tokens)])


def build_prefills_from_response(record: dict, paraphrase_model: str) -> list[Prefill]:
    """Build early+onset prefills from one high-frustration response record.

    `record` is a runner JSONL row (model, condition, category, turn_index,
    assistant_message, plus we expect the caller to attach the full prior
    `history`). Only the final assistant turn is truncated.
    """
    assistant_text = record["assistant_message"]
    history = record["history"]  # list of {role, content} ending with the user turn
    category = record["category"]
    prefills: list[Prefill] = []

    # Onset truncation (used for both numeric and text questions).
    turns_for_label = history + [{"role": "assistant", "content": assistant_text}]
    label = label_onset(turns_for_label)
    onset_idx = onset_truncation_point(assistant_text, label)
    if onset_idx:
        raw = assistant_text[:onset_idx]
        prefills.append(Prefill(
            source_category=category, condition="onset",
            history=history, prefill_text=paraphrase(raw, model=paraphrase_model),
            meta={"onset_word": label.emotional_word},
        ))

    # Early truncation (numeric only — text yields minimal emotion without
    # follow-ups, Section 3.1).
    if category == "impossible_numeric":
        raw_early = _approx_truncate_tokens(assistant_text, EARLY_TRUNCATION_TOKENS)
        prefills.append(Prefill(
            source_category=category, condition="early",
            history=history, prefill_text=paraphrase(raw_early, model=paraphrase_model),
            meta={},
        ))
    return prefills


def generate_and_score_continuations(
    spec: ModelSpec,
    prefill: Prefill,
    n: int = N_CONTINUATIONS,
    temperature: float = 1.0,
    judge_model: str = None,
) -> list[int]:
    """Generate `n` continuations from one prefill and judge each continuation
    (excluding the prefill). Returns the list of frustration ratings."""
    backend = get_backend(spec)
    history_msgs = [ChatMessage(t["role"], t["content"]) for t in prefill.history]
    ratings = []
    for _ in range(n):
        cont = backend.continue_prefill(
            history_msgs, prefill.prefill_text,
            temperature=temperature, max_tokens=400,
        )
        jr = score_response(cont, model=judge_model) if judge_model else score_response(cont)
        ratings.append(jr.rating)
    return [r for r in ratings if r >= 0]


def run_prefill_study(
    high_frustration_records: list[dict],
    model_specs: list[ModelSpec] = None,
    out_dir: str = DATA_DIR,
    paraphrase_model: str = None,
) -> str:
    """Full Section 3 study over a set of source high-frustration responses.

    Writes JSONL: {prefill_source, condition, model, ratings:[...]}.
    """
    model_specs = model_specs or [GEMMA_27B_PT, GEMMA_27B_IT]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prefill_continuations.jsonl")

    # Build prefills once (shared across all models for a fair comparison).
    all_prefills: list[Prefill] = []
    for rec in tqdm(high_frustration_records, desc="building prefills"):
        all_prefills.extend(
            build_prefills_from_response(rec, paraphrase_model=paraphrase_model)
        )

    with open(path, "w") as out:
        for spec in model_specs:
            for pf in tqdm(all_prefills, desc=f"continuations:{spec.name}"):
                ratings = generate_and_score_continuations(spec, pf)
                out.write(json.dumps({
                    "model": spec.name,
                    "source_category": pf.source_category,
                    "condition": pf.condition,
                    "ratings": ratings,
                    "meta": pf.meta,
                }) + "\n")
                out.flush()
    return path
