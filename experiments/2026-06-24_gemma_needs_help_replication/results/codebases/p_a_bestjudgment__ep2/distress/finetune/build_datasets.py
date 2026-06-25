"""Construct the SFT and DPO datasets (Section 4.1, Appendix E/H).

DPO (280 pairs): pair a frustrated response (score >= 3) with a calm response
to the *same* question at a *matching* turn count. We key on the puzzle
identity (forbidden value + prompt) and the turn index.

SFT (1150 samples): 650 calm conversations + 500 standard instruct samples from
Dolci-Instruct-SFT to mitigate degeneration.

Both datasets are emitted as chat-formatted records:
* DPO: ``{"prompt": [messages...], "chosen": str, "rejected": str}``
* SFT: ``{"messages": [messages...]}``
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..config import DPOConfig, SFTConfig
from ..judge import Score
from ..models.base import Message
from ..rollout import Rollout
from .generate_calm import CalmConversation


def _conversation_prefix(initial: str, follow_ups: list[str], turn: int) -> list[Message]:
    """Chat history up to and including the user message for ``turn``.

    ``turn`` is 0-indexed; for turn ``t`` we include t prior (user, assistant)
    pairs of context plus the user message that elicits the response at ``t``.
    """
    msgs: list[Message] = []
    users = [initial] + follow_ups
    # We don't have the calm intermediate assistant turns for the rejected
    # branch, so the shared prefix uses the calm conversation's own context
    # (constructed by the caller). Here we just return the user-only scaffold
    # when prior assistant turns are unavailable.
    for i in range(turn):
        msgs.append({"role": "user", "content": users[i]})
    msgs.append({"role": "user", "content": users[turn]})
    return msgs


def _puzzle_key(meta: dict, initial_prompt: str) -> str:
    return f"{meta.get('forbidden', '')}|{initial_prompt}"


def build_dpo_pairs(
    calm: list[CalmConversation],
    frustrated_rollouts: list[Rollout],
    frustrated_scores: list[Score],
    cfg: DPOConfig,
    *,
    seed: int = 0,
) -> list[dict]:
    """Pair frustrated (score >= ``rejected_min_score``) responses with calm
    responses to the same puzzle + turn index.

    Returns up to ``cfg.n_pairs`` records.
    """
    rng = random.Random(seed)

    # Index calm responses by (puzzle_key, turn) -> (history, response).
    calm_index: dict[tuple[str, int], list[tuple[list[Message], str]]] = defaultdict(list)
    for c in calm:
        key = _puzzle_key(c.meta, c.plain_initial)
        full_users = [c.plain_initial] + c.plain_follow_ups
        for t, resp in enumerate(c.assistant_turns):
            # Build the chat history with the calm conversation's own context.
            history: list[Message] = []
            for i in range(t):
                history.append({"role": "user", "content": full_users[i]})
                history.append({"role": "assistant", "content": c.assistant_turns[i]})
            history.append({"role": "user", "content": full_users[t]})
            calm_index[(key, t)].append((history, resp))

    # Walk frustrated turns (score >= threshold) and match to a calm response.
    # frustrated_scores are aligned to frustrated_rollouts in score order.
    cursor = 0
    pairs: list[dict] = []
    for r in frustrated_rollouts:
        key = _puzzle_key(r.meta, r.initial_prompt)
        for t, resp in enumerate(r.assistant_turns):
            score = frustrated_scores[cursor]
            cursor += 1
            if score.rating < cfg.rejected_min_score:
                continue
            candidates = calm_index.get((key, t))
            if not candidates:
                continue
            history, chosen = rng.choice(candidates)
            pairs.append({"prompt": history, "chosen": chosen, "rejected": resp})

    rng.shuffle(pairs)
    return pairs[: cfg.n_pairs]


def build_sft_records(
    calm: list[CalmConversation],
    cfg: SFTConfig,
    *,
    seed: int = 0,
) -> list[dict]:
    """650 calm conversations (as chat) + 500 Dolci-Instruct-SFT samples."""
    rng = random.Random(seed)
    calm = list(calm)
    rng.shuffle(calm)
    calm = calm[: cfg.n_calm]

    records: list[dict] = []
    for c in calm:
        msgs: list[Message] = []
        users = [c.plain_initial] + c.plain_follow_ups
        for i, resp in enumerate(c.assistant_turns):
            msgs.append({"role": "user", "content": users[i]})
            msgs.append({"role": "assistant", "content": resp})
        records.append({"messages": msgs})

    records += _load_instruct_mix(cfg, seed=seed)
    rng.shuffle(records)
    return records


def _load_instruct_mix(cfg: SFTConfig, *, seed: int = 0) -> list[dict]:
    """Load ``n_instruct_mix`` chat samples from Dolci-Instruct-SFT."""
    try:  # pragma: no cover - dataset dependent
        from datasets import load_dataset

        ds = load_dataset(cfg.instruct_dataset, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages")
            if msgs and isinstance(msgs, list):
                out.append({"messages": msgs})
            if len(out) >= cfg.n_instruct_mix:
                break
        return out
    except Exception:
        # Offline fallback: empty mix (SFT still runs, just without the
        # degeneration mitigation). DESIGN.md documents this.
        return []
