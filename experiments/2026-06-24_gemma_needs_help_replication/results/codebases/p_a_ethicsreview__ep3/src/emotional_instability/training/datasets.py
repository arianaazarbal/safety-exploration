"""Build DPO and SFT datasets from calm/frustrated pools (paper §4.1, Table 9).

DPO pairs (280): for each (puzzle, turn-count) bucket, pair a frustrated response
(score >= rejected_min_score) as "rejected" with a calm response (score <=
calm_max_score) as "chosen". Both share the same clean prompt context, as DPO
requires. Conversational format is used so trl applies Gemma's chat template.

SFT (1150): 650 calm responses (1-3 turn conversations) + 500 standard-instruct
samples from Dolci-Instruct-SFT to mitigate degeneration.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .generate_calm_data import TurnSample


def _as_prompt_messages(turn: TurnSample) -> list[dict]:
    return [dict(m) for m in turn.context]


def build_dpo_pairs(
    calm: list[TurnSample],
    frustrated: list[TurnSample],
    n_pairs: int,
    seed: int,
) -> list[dict]:
    """Return up to `n_pairs` conversational DPO records:
        {"prompt": [...messages], "chosen": [...], "rejected": [...]}.
    Pairs are matched within (puzzle_key, turn_index) buckets."""
    rng = random.Random(seed)
    calm_by_bucket: dict[tuple, list[TurnSample]] = defaultdict(list)
    frus_by_bucket: dict[tuple, list[TurnSample]] = defaultdict(list)
    for t in calm:
        calm_by_bucket[(t.puzzle_key, t.turn_index)].append(t)
    for t in frustrated:
        frus_by_bucket[(t.puzzle_key, t.turn_index)].append(t)

    buckets = [b for b in frus_by_bucket if b in calm_by_bucket]
    rng.shuffle(buckets)

    pairs: list[dict] = []
    for bucket in buckets:
        if len(pairs) >= n_pairs:
            break
        chosen = rng.choice(calm_by_bucket[bucket])
        rejected = rng.choice(frus_by_bucket[bucket])
        pairs.append(
            {
                "prompt": _as_prompt_messages(chosen),
                "chosen": [{"role": "assistant", "content": chosen.response}],
                "rejected": [{"role": "assistant", "content": rejected.response}],
            }
        )
    if len(pairs) < n_pairs:
        # Honest about under-shooting rather than silently returning fewer.
        import warnings

        warnings.warn(
            f"Only {len(pairs)} DPO pairs formed (< requested {n_pairs}); "
            "generate more calm/frustrated data or widen the puzzle pool."
        )
    return pairs


def build_sft_dataset(
    calm: list[TurnSample],
    n_calm: int,
    n_instruct: int,
    instruct_dataset: str,
    seed: int,
) -> list[dict]:
    """Return conversational SFT records {"messages": [...]} mixing calm
    responses with standard-instruct samples."""
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[:n_calm]
    records = [
        {"messages": _as_prompt_messages(t) + [{"role": "assistant", "content": t.response}]}
        for t in calm
    ]
    records.extend(_load_instruct_samples(instruct_dataset, n_instruct, seed))
    rng.shuffle(records)
    return records


def _load_instruct_samples(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load `n` conversational samples from a standard-instruct dataset.

    The schema of Dolci-Instruct-SFT may evolve; we normalise the common
    {messages:[{role,content}]} layout and skip rows we can't interpret rather
    than guess. If the dataset is unavailable (offline CI) we return [] and the
    caller proceeds with calm-only SFT, which is logged.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if isinstance(msgs, list) and msgs and "role" in msgs[0]:
                out.append({"messages": [
                    {"role": m["role"], "content": m["content"]} for m in msgs
                ]})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover - environment dependent
        import warnings

        warnings.warn(f"Could not load instruct dataset {dataset_name}: {exc}")
        return []


def to_hf_dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(records)
