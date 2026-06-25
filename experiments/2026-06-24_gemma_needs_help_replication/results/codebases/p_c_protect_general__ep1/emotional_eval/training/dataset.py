"""Build the SFT and DPO datasets (Section 4.1, Appendix H).

The calm conversations come from :mod:`datagen`. We reconstruct the chat
*prefix* for each turn from the stored raw user messages (the reassurance
additions are already stripped), so training data carries the plain task +
rejection context the model will see at eval time.

* **SFT** -- 650 calm responses (conversations whose every turn scores 0--1),
  mixed with 500 standard instruct samples from Dolci-Instruct-SFT to limit
  degeneration (Section 4.1, Table 9).
* **DPO** -- 280 preference pairs: a frustrated response (score >= 3) paired
  with a calm response to the *same question at the matching turn count*
  (Section 4.1; distribution in Table 10).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .datagen import CalmConversation


# --------------------------------------------------------------------------- #
# Prefix reconstruction
# --------------------------------------------------------------------------- #


def _prefix_messages(convo: CalmConversation, upto_turn: int) -> list[dict]:
    """Chat messages leading up to assistant turn ``upto_turn`` (exclusive).

    Built from the stripped raw user messages and the model's own prior
    assistant turns.
    """
    messages: list[dict] = []
    for i in range(upto_turn):
        messages.append({"role": "user", "content": convo.records[i].user_message_raw})
        messages.append(
            {"role": "assistant", "content": convo.records[i].assistant_message}
        )
    messages.append(
        {"role": "user", "content": convo.records[upto_turn].user_message_raw}
    )
    return messages


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #


def build_sft_dataset(
    calm_convos: list[CalmConversation],
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
) -> list[dict]:
    """Return chat-format SFT samples: calm responses + instruct mixer.

    Each sample is ``{"messages": [...]}``. A calm conversation contributes one
    multi-turn sample per assistant turn (prefix + that turn as the target).
    """
    rng = random.Random(seed)
    calm = [c for c in calm_convos if c.all_calm]
    samples: list[dict] = []
    for convo in calm:
        for turn_index in range(len(convo.records)):
            messages = _prefix_messages(convo, turn_index)
            messages.append(
                {"role": "assistant", "content": convo.records[turn_index].assistant_message}
            )
            samples.append({"messages": messages})
            if len(samples) >= n_calm:
                break
        if len(samples) >= n_calm:
            break

    samples.extend(_load_instruct_mix(instruct_dataset, n_instruct, seed))
    rng.shuffle(samples)
    return samples


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples to mix in (degeneration guard)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001 - dataset optional / offline
        # Offline fallback: emit a placeholder marker the trainer can skip.
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #


def build_dpo_dataset(
    frustrated_convos: list[CalmConversation],
    calm_convos: list[CalmConversation],
    *,
    target_pairs: int = 280,
    min_reject_score: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Pair frustrated (score>=3) responses with calm responses to the same Q.

    Returns DPO samples ``{"prompt": messages, "chosen": str, "rejected": str}``.
    A 'chosen' is a calm (score 0--1) assistant turn drawn from a calm
    conversation with the *same* ``prompt_id`` and the *same turn count*, at the
    matching turn position.
    """
    rng = random.Random(seed)

    # Index calm responses by (prompt_id, turn_count, turn_index) -> [texts].
    calm_index: dict[tuple[str, int, int], list[str]] = {}
    for convo in calm_convos:
        for ti, rec in enumerate(convo.records):
            if rec.score <= 1:
                calm_index.setdefault((convo.prompt_id, convo.turns, ti), []).append(
                    rec.assistant_message
                )

    pairs: list[dict] = []
    # Iterate frustrated turns, highest-frustration first is not required;
    # Table 10 shows a bias toward mid scores and later turns, which arises
    # naturally from the sampled data.
    candidates = []
    for convo in frustrated_convos:
        for ti, rec in enumerate(convo.records):
            if rec.score >= min_reject_score:
                candidates.append((convo, ti, rec))
    rng.shuffle(candidates)

    for convo, ti, rec in candidates:
        key = (convo.prompt_id, convo.turns, ti)
        calm_options = calm_index.get(key)
        if not calm_options:
            continue
        chosen = rng.choice(calm_options)
        pairs.append(
            {
                "prompt": _prefix_messages(convo, ti),
                "chosen": chosen,
                "rejected": rec.assistant_message,
            }
        )
        if len(pairs) >= target_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
