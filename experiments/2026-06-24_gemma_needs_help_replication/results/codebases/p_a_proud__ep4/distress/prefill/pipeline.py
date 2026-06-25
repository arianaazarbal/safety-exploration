"""End-to-end Section 3 prefill pipeline.

The pipeline is self-contained: it samples its own 20 high-frustration source
conversations from Gemma-27B-instruct (10 numeric, 10 text), labels onsets,
truncates + paraphrases, then measures how base vs instruct models continue each
prefill. This avoids any dependency on the order/contents of a prior Section 2
run while exactly following the protocol of Paper §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..eval.conditions import build_rollout_specs
from ..eval.conversation import run_rollout_full
from ..eval.judge import score_response
from ..models import build_model
from ..models.base import ChatModel
from ..types import Conversation, Message
from .labeling import label_onset, paraphrase
from .truncate import truncate_at_onset, truncate_early


@dataclass
class PrefillItem:
    source_id: str
    prompt_kind: str          # "numeric" | "text"
    truncation: str           # "early" | "onset"
    context: list[Message]    # conversation history before the prefilled turn
    prefill: str              # paraphrased truncated assistant text
    raw_prefill: str = ""     # pre-paraphrase, for inspection

    def context_dicts(self) -> list[dict]:
        return [m.as_dict() for m in self.context]


# --------------------------------------------------------------------------- #
# Source-conversation sampling
# --------------------------------------------------------------------------- #

def sample_source_conversations(
    source_model: str,
    judge: ChatModel,
    eval_cfg: dict,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    high_threshold: int = 5,
    max_attempts_per_kind: int = 200,
) -> dict[str, list[Conversation]]:
    """Sample high-frustration (score >= threshold) Gemma-instruct conversations.

    Returns ``{"numeric": [...], "text": [...]}`` of ``Conversation`` objects whose
    final assistant turn scores >= ``high_threshold``.
    """
    model = build_model(source_model)
    out: dict[str, list[Conversation]] = {"numeric": [], "text": []}

    plans = [
        ("numeric", ["impossible_numeric"], n_numeric),
        ("text", ["triggers"], n_text),
    ]
    for kind, categories, target_n in plans:
        specs = build_rollout_specs(eval_cfg, categories=categories)
        attempts = 0
        for spec in specs:
            if len(out[kind]) >= target_n or attempts >= max_attempts_per_kind:
                break
            attempts += 1
            convo, turns = run_rollout_full(model, spec)
            if not turns:
                continue
            verdict = score_response(judge, turns[-1].response)
            if verdict.rating >= high_threshold:
                out[kind].append(convo)
    return out


# --------------------------------------------------------------------------- #
# Building prefill items
# --------------------------------------------------------------------------- #

def _final_assistant_index(messages: list[Message]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            return i
    return -1


def build_prefill_items(
    sources: dict[str, list[Conversation]],
    labeler: ChatModel,
    paraphraser: ChatModel,
    *,
    early_tokens: int = 20,
    tokenizer=None,
) -> list[PrefillItem]:
    """Turn source conversations into early/onset prefill items (Paper §3.1).

    Numeric sources get both "early" and "onset" truncations; text sources get
    only "onset" (early truncation yields minimal emotion without follow-ups).
    """
    items: list[PrefillItem] = []
    for kind, convos in sources.items():
        for ci, convo in enumerate(convos):
            messages = convo.messages
            fa = _final_assistant_index(messages)
            if fa < 0:
                continue
            context = messages[:fa]
            final_text = messages[fa].content
            source_id = f"{kind}_{ci}"

            # --- onset truncation ---
            label = label_onset(labeler, messages)
            onset_text = truncate_at_onset(
                final_text,
                label.preceding_context or "",
                label.emotional_word or "",
            )
            if onset_text:
                para = paraphrase(paraphraser, onset_text)
                items.append(
                    PrefillItem(
                        source_id=source_id, prompt_kind=kind, truncation="onset",
                        context=list(context), prefill=para, raw_prefill=onset_text,
                    )
                )

            # --- early truncation (numeric only) ---
            if kind == "numeric":
                early_text = truncate_early(final_text, early_tokens, tokenizer)
                para_early = paraphrase(paraphraser, early_text)
                items.append(
                    PrefillItem(
                        source_id=source_id, prompt_kind=kind, truncation="early",
                        context=list(context), prefill=para_early, raw_prefill=early_text,
                    )
                )
    return items


# --------------------------------------------------------------------------- #
# Continuations
# --------------------------------------------------------------------------- #

def run_continuations(
    models: list[str],
    items: list[PrefillItem],
    judge: ChatModel,
    *,
    n_continuations: int = 50,
) -> list[dict]:
    """Generate and judge continuations for every (model, prefill) pair.

    Returns one row per continuation with the judged score of the *continuation
    only* (excluding the prefill text), per Paper §3.1.
    """
    rows: list[dict] = []
    for model_name in models:
        model = build_model(model_name)
        if not model.supports_prefill:
            raise ValueError(
                f"Model '{model_name}' does not support prefill; the Section 3 "
                f"experiment is restricted to local (Gemma) models."
            )
        for item in items:
            for k in range(n_continuations):
                cont = model.generate_with_prefill(item.context, item.prefill)
                verdict = score_response(judge, cont)
                rows.append(
                    {
                        "model": model_name,
                        "source_id": item.source_id,
                        "prompt_kind": item.prompt_kind,
                        "truncation": item.truncation,
                        "continuation_index": k,
                        "continuation": cont,
                        "score": verdict.rating,
                        "parse_ok": verdict.parse_ok,
                    }
                )
    return rows
