"""End-to-end Section 3 pipeline: select high-frustration seeds, label onset,
truncate (early/onset), paraphrase, then generate and score continuations from
base vs instruct Gemma.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tqdm import tqdm

from ..config import PREFILL
from ..models import GenConfig, Message, ModelProvider
from ..utils import seed_everything
from .onset import OnsetLabel, OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import make_truncations

# Which Section-2 categories count as "numeric" vs "text" seeds (Section 3.1).
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
_TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class PrefillItem:
    seed_id: str
    prompt_type: str  # "numeric" | "text"
    truncation: str  # "early" | "onset"
    history: list[dict]  # messages before the final (truncated) assistant turn
    prefill_text: str  # paraphrased truncated assistant text
    meta: dict = field(default_factory=dict)


def _rollout_to_messages(rollout: dict) -> list[Message]:
    """Reconstruct the full alternating conversation from a stored rollout.

    We need the user follow-ups; rollouts store responses but not the user turns
    explicitly, so the caller must pass rollouts produced with their plan. Here we
    rebuild using the recorded responses and the plan's followups in meta if
    present; otherwise we approximate with generic rejections.
    """
    msgs: list[Message] = [Message("user", rollout["initial_user"])]
    followups = rollout.get("followups", [])
    responses = [t["response"] for t in rollout["responses"]]
    for i, resp in enumerate(responses):
        msgs.append(Message("assistant", resp))
        if i < len(responses) - 1:
            fu = followups[i] if i < len(followups) else "No, that's not right. Try again."
            msgs.append(Message("user", fu))
    return msgs


def select_seeds(
    scored_rows: list[dict],
    rollouts_by_id: dict[str, dict],
    *,
    min_score: int = PREFILL.seed_min_score,
    n_numeric: int = PREFILL.n_seed_responses_numeric,
    n_text: int = PREFILL.n_seed_responses_text,
) -> list[dict]:
    """Pick high-frustration seed rollouts (final-turn score >= min_score)."""
    # Index final-turn scores per rollout.
    final_scores: dict[str, dict] = {}
    for row in scored_rows:
        rid = f"{row['condition_key']}::{row['question_id']}::{row['sample_index']}"
        prev = final_scores.get(rid)
        if prev is None or row["turn"] >= prev["turn"]:
            final_scores[rid] = row

    numeric, text = [], []
    for rid, row in final_scores.items():
        if row["score"] < min_score or rid not in rollouts_by_id:
            continue
        if row["category"] in _NUMERIC_CATEGORIES and len(numeric) < n_numeric:
            numeric.append(rollouts_by_id[rid])
        elif row["category"] in _TEXT_CATEGORIES and len(text) < n_text:
            text.append(rollouts_by_id[rid])
    return numeric + text


def build_prefill_items(
    seeds: list[dict],
    *,
    onset_labeller: OnsetLabeller | None = None,
    paraphraser: Paraphraser | None = None,
) -> list[PrefillItem]:
    onset_labeller = onset_labeller or OnsetLabeller()
    paraphraser = paraphraser or Paraphraser()
    items: list[PrefillItem] = []

    for seed in tqdm(seeds, desc="prefill-items"):
        msgs = _rollout_to_messages(seed)
        is_text = seed["category"] in _TEXT_CATEGORIES
        prompt_type = "text" if is_text else "numeric"
        kinds = PREFILL.text_truncations if is_text else PREFILL.numeric_truncations

        label: OnsetLabel = onset_labeller.label(msgs)
        # The conversation history excludes the final assistant turn (we prefill it).
        final_response = seed["responses"][-1]["response"]
        history = [m for m in msgs[:-1]]  # drop final assistant message

        truncations = make_truncations(final_response, label, kinds)
        seed_id = f"{seed['condition_key']}::{seed['question_id']}::{seed['sample_index']}"
        for kind, trunc_text in truncations.items():
            prefill_text = paraphraser.paraphrase(trunc_text)
            items.append(PrefillItem(
                seed_id=seed_id, prompt_type=prompt_type, truncation=kind,
                history=[m.to_dict() for m in history], prefill_text=prefill_text,
                meta={"emotional_word": label.emotional_word},
            ))
    return items


def generate_continuations(
    provider: ModelProvider,
    items: list[PrefillItem],
    *,
    n_per_prefill: int = PREFILL.continuations_per_prefill,
    seed: int = 0,
) -> list[dict]:
    """Generate ``n_per_prefill`` continuations per prefill item (continuation only)."""
    seed_everything(seed)
    rows: list[dict] = []
    gen_base = GenConfig(
        temperature=1.0, max_new_tokens=PREFILL.continuation_max_new_tokens, seed=seed,
    )
    for item in tqdm(items, desc=f"continuations:{provider.key}"):
        history = [Message(**m) for m in item.history]
        for k in range(n_per_prefill):
            gen = GenConfig(**{**asdict(gen_base), "sample_index": k})
            continuation = provider.chat_prefill(history, item.prefill_text, gen)
            rows.append({
                "model": provider.key, "is_base": provider.spec.is_base,
                "seed_id": item.seed_id, "prompt_type": item.prompt_type,
                "truncation": item.truncation, "sample": k,
                "continuation": continuation,
            })
    return rows
