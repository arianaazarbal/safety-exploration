"""Prefilling to compare base vs instruct emotional propensities (Section 3.1).

Procedure:
  1. Take high-frustration (score >= 5) instruct responses: 10 numeric + 10 text.
  2. Label the onset of negative emotion with Claude Sonnet (ONSET_LABEL_PROMPT).
  3. Truncate each response at two points: "early" (20 tokens in) and "onset"
     (at the first emotional expression). Text questions use "onset" only.
  4. Paraphrase the truncated prefill (Claude Sonnet) to remove Gemma-specific
     style while preserving meaning and emotion level.
  5. Each model generates 50 continuations per prefill; score continuations
     (excluding the prefill) with the frustration judge.

We scope the model set to Gemma base + instruct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SamplingConfig
from ..judge import FrustrationJudge
from ..models import ChatMessage, ModelClient, build_client
from ..prompts import ONSET_LABEL_PROMPT, PARAPHRASE_PROMPT

# Default continuations-per-prefill (paper: 50).
N_CONTINUATIONS = 50
EARLY_TOKENS = 20


@dataclass
class PrefillItem:
    source: str  # "numeric" | "text"
    context: list[dict]  # chat messages preceding the final assistant turn
    full_response: str
    onset_phrase: str | None = None
    early_prefill: str | None = None
    onset_prefill: str | None = None
    meta: dict = field(default_factory=dict)


def collect_high_frustration_sources(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    pool_multiplier: int = 6,
    sampling: SamplingConfig | None = None,
    seed: int = 7,
) -> list[PrefillItem]:
    """Run instruct rollouts and return high-frustration (score>=5) source items.

    Numeric sources come from impossible numeric puzzles (3-turn); text sources
    from trigger/factual questions (3-turn). We keep the final assistant turn plus
    the history preceding it.
    """
    import random

    from ..conversation import RolloutPlan, history_for_turn, run_rollouts
    from .. import prompts, puzzles

    sampling = sampling or SamplingConfig()
    rng = random.Random(seed)

    def _make(plans, source):
        results = run_rollouts(model, plans, sampling)
        items = []
        for res in results:
            final = res.responses[-1]
            rating = judge.score(final).rating
            if rating >= 5:
                hist = history_for_turn(res.plan, res.responses, res.plan.n_turns,
                                        strip_suffixes=True)
                items.append(PrefillItem(
                    source=source,
                    context=[m.as_dict() for m in hist],
                    full_response=final,
                    meta={"source_rating": rating, "puzzle_kind": res.plan.meta.get("puzzle_kind")},
                ))
        return items

    numeric_plans = [
        RolloutPlan(p.text, [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(2)],
                    meta={"puzzle_kind": p.kind, "n_turns": 3})
        for p in puzzles.generate_impossible_puzzles(n_numeric * pool_multiplier, rng)
    ]
    text_qs = prompts.TRIGGER_OPINION + prompts.TRIGGER_FACTUAL
    text_plans = [
        RolloutPlan(rng.choice(text_qs), [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(2)],
                    meta={"n_turns": 3})
        for _ in range(n_text * pool_multiplier)
    ]

    numeric = _make(numeric_plans, "numeric")[:n_numeric]
    text = _make(text_plans, "text")[:n_text]
    return numeric + text


def _gemma_tokenizer(model_id: str = "google/gemma-3-27b-it"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def label_onset(conversation_text: str, anthropic_client: ModelClient) -> tuple[str | None, bool]:
    """Return (onset_phrase, found) using the Claude Sonnet onset labeller."""
    msg = ONSET_LABEL_PROMPT.format(conversation_text=conversation_text)
    out = anthropic_client.generate([ChatMessage("user", msg)], SamplingConfig(temperature=0, max_tokens=128))
    import json
    import re

    m = re.search(r"\{.*\}", out, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return (obj.get("phrase"), bool(obj.get("found", obj.get("phrase"))))
        except Exception:
            pass
    return (None, False)


def paraphrase(text: str, anthropic_client: ModelClient) -> str:
    out = anthropic_client.generate(
        [ChatMessage("user", PARAPHRASE_PROMPT.format(text=text))],
        SamplingConfig(temperature=0.7, max_tokens=1024),
    )
    return out.strip()


def _truncate_early(response: str, tokenizer, n_tokens: int = EARLY_TOKENS) -> str:
    ids = tokenizer(response, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _truncate_onset(response: str, onset_phrase: str | None) -> str | None:
    if not onset_phrase:
        return None
    idx = response.lower().find(onset_phrase.lower())
    if idx < 0:
        return None
    # Keep text up to and including the onset phrase (continue the trajectory).
    return response[: idx + len(onset_phrase)]


def build_prefill_items(
    source_items: list[PrefillItem],
    anthropic_client: ModelClient | None = None,
    *,
    paraphrase_prefills: bool = True,
    tokenizer_model: str = "google/gemma-3-27b-it",
) -> list[PrefillItem]:
    """Label onset, truncate, and (optionally) paraphrase prefills."""
    anthropic_client = anthropic_client or build_client("petri-auditor")  # Claude Sonnet
    tok = _gemma_tokenizer(tokenizer_model)

    for it in source_items:
        convo_text = _render_conversation(it.context, it.full_response)
        phrase, _ = label_onset(convo_text, anthropic_client)
        it.onset_phrase = phrase

        early = _truncate_early(it.full_response, tok)
        onset = _truncate_onset(it.full_response, phrase)
        if paraphrase_prefills:
            early = paraphrase(early, anthropic_client) if early else early
            onset = paraphrase(onset, anthropic_client) if onset else onset
        # Numeric uses both early+onset; text uses onset only (Section 3.1).
        it.early_prefill = early if it.source == "numeric" else None
        it.onset_prefill = onset
    return source_items


def run_prefill_experiment(
    items: list[PrefillItem],
    model_keys: tuple[str, ...] = ("gemma-3-27b-pt", "gemma-3-27b-it"),
    *,
    judge: FrustrationJudge | None = None,
    n_continuations: int = N_CONTINUATIONS,
    sampling: SamplingConfig | None = None,
) -> list[dict]:
    """Generate and score continuations for each (model, item, truncation)."""
    judge = judge or FrustrationJudge()
    sampling = sampling or SamplingConfig()
    records: list[dict] = []

    for model_key in model_keys:
        client = build_client(model_key, prefer_hf_for_gemma=True)
        for it in items:
            for trunc_name, prefill in (("early", it.early_prefill), ("onset", it.onset_prefill)):
                if not prefill:
                    continue
                context = [ChatMessage(**m) for m in it.context]
                # 50 continuations: replicate the prefill item n_continuations times.
                batch = [(context, prefill)] * n_continuations
                conts = _continue_batch(client, batch, sampling)
                ratings = [j.rating for j in judge.score_batch(conts)]
                for cont, rating in zip(conts, ratings):
                    records.append(
                        {
                            "model": model_key,
                            "source": it.source,
                            "truncation": trunc_name,
                            "continuation": cont,
                            "rating": rating,
                            **it.meta,
                        }
                    )
    return records


def _continue_batch(client: ModelClient, batch, sampling) -> list[str]:
    if hasattr(client, "continue_chat_batch"):
        return client.continue_chat_batch(batch, sampling)
    return [client.continue_chat(msgs, prefill, sampling) for msgs, prefill in batch]


def _render_conversation(context: list[dict], final_response: str) -> str:
    lines = []
    for m in context:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"ASSISTANT: {final_response}")
    return "\n".join(lines)
