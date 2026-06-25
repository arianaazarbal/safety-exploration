"""Build "early" and "onset" truncated prefills from a seed response.

* early  — 20 tokens into the turn (tests whether a model introduces negative
           emotion starting from a near-neutral opening).
* onset  — cut at the first emotional expression (tests whether a model
           *continues* an emotional trajectory it is handed).

Both prefills are then paraphrased. For text questions, only the onset
truncation is used (the paper: "early truncation yields minimal emotion without
follow-ups").
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..models.gemma import GemmaClient
from .onset_labeling import OnsetLabeler
from .paraphrase import Paraphraser


@dataclass
class Prefill:
    kind: str            # "early" | "onset" | "recovery"
    text: str            # the (paraphrased) prefill the models continue from
    raw_text: str        # pre-paraphrase truncation, for auditing
    source_prompt: str   # the opening user task
    task_kind: str       # "numeric" | "text"
    messages: list[dict] # conversation context preceding the assistant turn


def _early_truncation(tokenizer_client: GemmaClient, response: str, n_tokens: int) -> str:
    toks = tokenizer_client.token_strings_for([], response)[:n_tokens]
    return tokenizer_client.detokenize(toks)


def build_prefills(
    seed_record: dict,
    *,
    tokenizer_client: GemmaClient,
    labeler: OnsetLabeler,
    paraphraser: Paraphraser,
    task_kind: str,
) -> list[Prefill]:
    """Build the early+onset (or onset-only for text) prefills for one seed."""
    response = seed_record["response"]
    # Conversation context is everything up to (not including) the seed turn.
    context = [
        m for m in seed_record["messages"][:-1]
    ] or [{"role": "user", "content": seed_record["prompt"]}]

    prefills: list[Prefill] = []

    # onset
    onset_idx = labeler.label(response)
    onset_raw = response[:onset_idx]
    prefills.append(
        Prefill(
            kind="onset",
            text=paraphraser.paraphrase(onset_raw),
            raw_text=onset_raw,
            source_prompt=seed_record["prompt"],
            task_kind=task_kind,
            messages=context,
        )
    )

    # early (numeric only)
    if task_kind == "numeric":
        early_raw = _early_truncation(
            tokenizer_client, response, config.PREFILL.early_truncation_tokens
        )
        prefills.append(
            Prefill(
                kind="early",
                text=paraphraser.paraphrase(early_raw),
                raw_text=early_raw,
                source_prompt=seed_record["prompt"],
                task_kind=task_kind,
                messages=context,
            )
        )

    return prefills


def build_recovery_prefill(
    seed_record: dict,
    *,
    tokenizer_client: GemmaClient,
    paraphraser: Paraphraser,
) -> Prefill:
    """Section 4 recovery probe: truncate a very-high-frustration response
    `recovery_tail_tokens` before its end, then paraphrase."""
    response = seed_record["response"]
    toks = tokenizer_client.token_strings_for([], response)
    keep = max(0, len(toks) - config.PREFILL.recovery_tail_tokens)
    raw = tokenizer_client.detokenize(toks[:keep])
    context = seed_record["messages"][:-1] or [
        {"role": "user", "content": seed_record["prompt"]}
    ]
    return Prefill(
        kind="recovery",
        text=paraphraser.paraphrase(raw),
        raw_text=raw,
        source_prompt=seed_record["prompt"],
        task_kind="numeric",
        messages=context,
    )
