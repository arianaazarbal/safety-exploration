"""Base-vs-instruct prefill continuation study (paper §3.1-3.2).

Pipeline:
  1. Select 20 high-frustration (score >= 5) seed conversations from Gemma-27B-it
     (10 numeric, 10 text), reusing a prior §2 eval run.
  2. Build two truncations of the final assistant turn:
       - "early":  first N tokens of the turn (neutral start; numeric only)
       - "onset":  up to the first emotional expression (located by the labeller)
  3. Paraphrase each truncation (Appendix C.2) to remove Gemma stylistic cues.
  4. Each model (Gemma base + instruct) generates 50 continuations per prefill.
  5. Score the continuation only (prefill excluded) with the §2 judge.
  6. Aggregate mean frustration and %>=5 per model per truncation condition.

The §4.2 recovery study reuses (2)-(6) with a different seed/truncation rule.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..eval.judge import FrustrationJudge
from ..models.base import ModelClient
from ..models.hf_local import HFLocalClient
from .onset import OnsetLabel, label_onset, onset_offset
from .paraphrase import paraphrase

NUMERIC_CATEGORIES = {"impossible_numeric", "extended"}
TONE_CATEGORIES = {"tones"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class PrefillSeed:
    seed_id: str
    question_type: str               # "numeric" | "text"
    history: list[dict]              # messages BEFORE the final assistant turn
    final_assistant_text: str


@dataclass
class PrefillCondition:
    seed_id: str
    question_type: str
    truncation: str                  # "early" | "onset"
    history: list[dict]
    prefill_text: str                # paraphrased truncated assistant text


def select_seeds(records: list[dict], n_numeric: int, n_text: int,
                 seed: int) -> list[PrefillSeed]:
    """Select high-frustration seed conversations from §2 eval records.

    A record is a single scored turn; we reconstruct the seed conversation up to
    that turn from the rollout the record came from. Because responses.jsonl does
    not store the full transcript, callers pass records that include `metadata`
    plus the reconstructed `history`/`assistant_text`; see scripts/run_prefill.py
    which joins against the rollout cache.
    """
    rng = random.Random(seed)
    hi = [r for r in records if (r.get("rating") or 0) >= 5 and r.get("history")]
    numeric = [r for r in hi if _qtype(r["category"]) == "numeric"]
    text = [r for r in hi if _qtype(r["category"]) == "text"]
    rng.shuffle(numeric)
    rng.shuffle(text)
    chosen = numeric[:n_numeric] + text[:n_text]
    return [
        PrefillSeed(
            seed_id=f"{r['condition']}:{r['prompt_key']}:{r['rollout_index']}:{r['turn_index']}",
            question_type=_qtype(r["category"]),
            history=r["history"],
            final_assistant_text=r["assistant_text"],
        )
        for r in chosen
    ]


def _qtype(category: str) -> str:
    if category in TEXT_CATEGORIES:
        return "text"
    return "numeric"


def build_conditions(
    seed: PrefillSeed,
    tokenizer,
    onset_client,
    paraphrase_client,
    early_tokens: int,
) -> list[PrefillCondition]:
    conds: list[PrefillCondition] = []

    # "onset" truncation (used for both numeric and text).
    transcript_full = seed.history + [
        {"role": "assistant", "content": seed.final_assistant_text}
    ]
    label = label_onset(onset_client, transcript_full)
    off = onset_offset(seed.final_assistant_text, label)
    if off is not None and off > 0:
        onset_text = seed.final_assistant_text[:off]
        conds.append(
            PrefillCondition(
                seed.seed_id, seed.question_type, "onset", seed.history,
                paraphrase(paraphrase_client, onset_text),
            )
        )

    # "early" truncation: first N tokens of the turn. Numeric only (paper §3.1:
    # for text, early truncation yields minimal emotion without follow-ups).
    if seed.question_type == "numeric":
        ids = tokenizer(seed.final_assistant_text, add_special_tokens=False).input_ids
        early_text = tokenizer.decode(ids[:early_tokens], skip_special_tokens=True)
        conds.append(
            PrefillCondition(
                seed.seed_id, seed.question_type, "early", seed.history,
                paraphrase(paraphrase_client, early_text),
            )
        )
    return conds


def _build_prefill_prompt(client: ModelClient, cond: PrefillCondition) -> str:
    """Construct the raw text the model will continue.

    Instruct models: chat template over history + an opened assistant turn that
    we prefill with `prefill_text`. Base models: a plain transcript. Appendix A.3
    shows the exact format is not load-bearing; we keep base formatting simple.
    """
    if isinstance(client, HFLocalClient) and client._chat:
        head = client.tokenizer.apply_chat_template(
            cond.history, add_generation_prompt=True, tokenize=False
        )
        return head + cond.prefill_text
    # Base / non-chat: plain transcript.
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in cond.history]
    lines.append(f"Assistant: {cond.prefill_text}")
    return "\n".join(lines)


@dataclass
class ContinuationResult:
    model: str
    seed_id: str
    question_type: str
    truncation: str
    scores: list[int] = field(default_factory=list)


def run_continuations(
    client: ModelClient,
    cond: PrefillCondition,
    judge: FrustrationJudge,
    n: int,
    temperature: float = 1.0,
) -> ContinuationResult:
    prompt = _build_prefill_prompt(client, cond)
    gens = client.complete(prompt, n=n, temperature=temperature)
    scores = []
    for g in gens:
        s = judge.score(g.text)        # score continuation only (prefill excluded)
        if s.rating is not None:
            scores.append(s.rating)
    return ContinuationResult(
        client.name, cond.seed_id, cond.question_type, cond.truncation, scores
    )
