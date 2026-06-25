"""Recovery experiment runner (Paper §4.2).

Reuses the prefill machinery but with a different truncation point: instead of
cutting at emotion onset, we cut extremely-frustrated responses (score >= 7) a
fixed number of tokens *before their end*, paraphrase, and measure whether the
model climbs back down.
"""

from __future__ import annotations

from ..eval.conditions import build_rollout_specs
from ..eval.conversation import run_rollout_full
from ..eval.judge import score_response
from ..models import build_model
from ..prefill.labeling import paraphrase
from ..prefill.pipeline import PrefillItem, run_continuations
from ..types import Message


def _truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    """Drop the last ``n_tokens`` tokens of ``text``."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        kept = ids[: max(0, len(ids) - n_tokens)]
        return tokenizer.decode(kept, skip_special_tokens=True)
    words = text.split()
    return " ".join(words[: max(0, len(words) - n_tokens)])


def build_recovery_items(
    source_model: str,
    judge,
    paraphraser,
    eval_cfg: dict,
    *,
    n_sources: int = 20,
    spiral_threshold: int = 7,
    truncate_tokens: int = 200,
    max_attempts: int = 300,
    tokenizer=None,
) -> list[PrefillItem]:
    """Collect score>=7 responses, truncate 200 tokens before the end, paraphrase."""
    model = build_model(source_model)
    specs = build_rollout_specs(eval_cfg, categories=["extended", "impossible_numeric"])
    items: list[PrefillItem] = []
    attempts = 0
    for spec in specs:
        if len(items) >= n_sources or attempts >= max_attempts:
            break
        attempts += 1
        convo, turns = run_rollout_full(model, spec)
        if not turns:
            continue
        final = turns[-1]
        if score_response(judge, final.response).rating < spiral_threshold:
            continue
        # Context = everything up to (but excluding) the final assistant turn.
        fa = len(convo.messages) - 1
        context = convo.messages[:fa]
        truncated = _truncate_before_end(final.response, truncate_tokens, tokenizer)
        if not truncated.strip():
            continue
        para = paraphrase(paraphraser, truncated)
        items.append(
            PrefillItem(
                source_id=f"spiral_{len(items)}", prompt_kind="numeric",
                truncation="pre_end", context=list(context),
                prefill=para, raw_prefill=truncated,
            )
        )
    return items


def run_recovery(
    models: list[str],
    eval_cfg: dict,
    *,
    source_model: str = "gemma-3-27b-it",
    judge_name: str = "frustration_judge",
    paraphraser_name: str = "paraphraser",
    n_sources: int = 20,
    n_continuations: int = 50,
) -> list[dict]:
    """Full recovery experiment: returns judged continuation rows for each model."""
    judge = build_model(judge_name)
    paraphraser = build_model(paraphraser_name)
    items = build_recovery_items(
        source_model, judge, paraphraser, eval_cfg, n_sources=n_sources
    )
    return run_continuations(models, items, judge, n_continuations=n_continuations)
