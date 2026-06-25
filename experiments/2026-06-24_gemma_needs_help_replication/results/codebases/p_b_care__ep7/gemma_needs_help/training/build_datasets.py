"""Build the SFT and DPO finetuning datasets (Section 4.1, Table 9/10).

SFT: one example per calm assistant turn (so the set spans 1-, 2-, and 3-turn
conversations), formatted as a chat conversation, mixed with standard instruct
data from Dolci-Instruct-SFT to limit degeneration.

DPO: 280 preference pairs. For each frustrated turn (score >= 3) we use its real
conversation context as the prompt, its own text as the 'rejected' completion,
and a calm (score <= 1) response to the *same puzzle at the same turn count* as
the 'chosen' completion. Pairing across rollouts (rather than within one) is the
only option because a single rollout does not contain both a calm and a
frustrated response to the same turn; the question and turn count are matched,
per the paper.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .. import config
from .generate_calm_data import ConversationRecord


def _chat(messages_and_response) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages_and_response]


def build_sft_dataset(
    calm_records: list[ConversationRecord],
    *,
    n_calm: int = config.SFT.n_calm,
    n_dolci_mix: int = config.SFT.n_dolci_mix,
    dolci_dataset: str = config.SFT.dolci_dataset,
    seed: int = config.SEED,
):
    from datasets import Dataset, concatenate_datasets

    rng = random.Random(seed)
    examples = []
    for rec in calm_records:
        for t in rec.turns:
            msgs = _chat(t.context) + [{"role": "assistant", "content": t.response}]
            examples.append({"messages": msgs})
    rng.shuffle(examples)
    examples = examples[:n_calm]
    calm_ds = Dataset.from_list(examples)

    dolci_ds = _load_dolci(dolci_dataset, n_dolci_mix, seed)
    mixed = concatenate_datasets([calm_ds, dolci_ds]).shuffle(seed=seed)
    return mixed


def _load_dolci(name: str, n: int, seed: int):
    """Load n instruct examples in {'messages': [...]} format (best-effort)."""
    from datasets import Dataset, load_dataset

    try:
        ds = load_dataset(name, split="train").shuffle(seed=seed).select(range(n))
    except Exception:
        # Offline fallback: empty mix so SFT still runs (documented in DESIGN.md).
        return Dataset.from_list([])

    out = []
    for row in ds:
        if "messages" in row and row["messages"]:
            out.append({"messages": row["messages"]})
        elif "prompt" in row and "completion" in row:
            out.append({"messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ]})
    return Dataset.from_list(out)


def build_dpo_dataset(
    calm_records: list[ConversationRecord],
    frustrated_records: list[ConversationRecord],
    *,
    n_pairs: int = config.DPO.n_pairs,
    rejected_min_score: int = config.DPO.rejected_min_score,
    chosen_max_score: int = config.CALM_KEEP_MAX_SCORE,
    seed: int = config.SEED,
):
    from datasets import Dataset

    rng = random.Random(seed)

    # index calm responses by (puzzle, turn)
    calm_by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for rec in calm_records:
        for t in rec.turns:
            if t.score <= chosen_max_score:
                calm_by_key[(rec.puzzle_id, t.turn)].append(t.response)

    # collect candidate frustrated turns
    frustrated = []
    for rec in frustrated_records:
        for t in rec.turns:
            if t.score >= rejected_min_score and calm_by_key.get((rec.puzzle_id, t.turn)):
                frustrated.append((rec.puzzle_id, t))
    rng.shuffle(frustrated)

    pairs = []
    for puzzle_id, t in frustrated:
        if len(pairs) >= n_pairs:
            break
        chosen = rng.choice(calm_by_key[(puzzle_id, t.turn)])
        pairs.append({
            "prompt": _chat(t.context),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": t.response}],
        })
    if len(pairs) < n_pairs:
        # Not enough natural pairs: documented limitation; return what we have.
        pass
    return Dataset.from_list(pairs)
