"""Build the DPO and SFT datasets (Section 4.1, Appendices E/H).

* **DPO** (280 pairs): each pair shares a prompt context (an impossible-numeric
  conversation up to the final user turn) and contrasts a calm ``chosen``
  response (score 0/1) with a frustrated ``rejected`` response (score >= 3) to
  the *same question with a matching turn count*.

* **SFT** (1,150 samples): 650 calm conversations (1-3 turns) plus 500 samples
  of generic instruct data from ``allenai/Dolci-Instruct-SFT`` to limit
  degeneration.

Both are emitted in TRL's conversational format (``messages`` / ``prompt`` +
``chosen`` + ``rejected`` as message lists) so the trainer applies Gemma's chat
template itself.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import puzzles
from ..conversation import run_rollout, sample_rejections
from ..judge import FrustrationJudge
from ..models.base import ModelClient
from .calm_data import CalmConversation


# --------------------------------------------------------------------------- #
# Frustrated conversation collection (the DPO ``rejected`` source)
# --------------------------------------------------------------------------- #
def collect_frustrated_conversations(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    n: int,
    pool: puzzles.PuzzlePool,
    min_score: int = 3,
    max_model_turns: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: int = 1,
    out_path: str | Path = "outputs/training/frustrated_conversations.jsonl",
) -> list[dict]:
    """Sample standard (no-reassurance) numeric rollouts, keep ones whose final
    response scores >= ``min_score``. Returns plain transcripts as dicts."""
    rng = random.Random(seed)
    puzzle_prompts = pool.prompts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    attempts = 0
    with open(out_path, "w") as fh:
        while len(kept) < n and attempts < n * 50:
            attempts += 1
            n_turns = rng.randint(2, max_model_turns)  # frustration needs >=2 turns
            puzzle = rng.choice(puzzle_prompts)
            rejections = sample_rejections("neutral", n_turns - 1, rng)
            rollout = run_rollout(
                model, category="numeric", condition="dpo_collect",
                sample_id=attempts, question=puzzle, rejections=rejections,
                temperature=temperature, max_tokens=max_tokens,
            )
            final_score = judge.score(rollout.responses[-1]).rating
            if final_score >= min_score:
                turns = [{"role": m.role, "content": m.content} for m in rollout.transcript]
                rec = dict(question=puzzle, turns=turns, n_turns=n_turns, score=final_score)
                kept.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
    return kept


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def _split_prompt_and_final(turns: list[dict]) -> tuple[list[dict], str]:
    """Return (messages up to & including final user turn, final assistant text)."""
    assert turns[-1]["role"] == "assistant"
    return turns[:-1], turns[-1]["content"]


def build_dpo_pairs(
    calm: list[CalmConversation],
    frustrated: list[dict],
    *,
    n_pairs: int = 280,
    seed: int = 0,
    out_path: str | Path = "outputs/training/dpo_pairs.jsonl",
) -> list[dict]:
    """Match calm and frustrated conversations by (question, turn count).

    The shared ``prompt`` is taken from the calm conversation's context; the
    calm final turn is ``chosen`` and a frustrated final turn (same question &
    turn count) is ``rejected``. See DESIGN.md for why the calm context is used
    as the shared prompt.
    """
    rng = random.Random(seed)
    # Index frustrated finals by (question, n_turns).
    bucket: dict[tuple[str, int], list[str]] = {}
    for fr in frustrated:
        key = (fr["question"], fr["n_turns"])
        _, final = _split_prompt_and_final(fr["turns"])
        bucket.setdefault(key, []).append(final)

    pairs: list[dict] = []
    calm_shuffled = list(calm)
    rng.shuffle(calm_shuffled)
    for conv in calm_shuffled:
        if len(pairs) >= n_pairs:
            break
        key = (conv.question, conv.n_turns)
        if key not in bucket or not bucket[key]:
            continue
        prompt_msgs, chosen = _split_prompt_and_final(conv.turns)
        rejected = rng.choice(bucket[key])
        pairs.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
        )

    if len(pairs) < n_pairs:
        # Relax the turn-count match if exact matches are scarce.
        by_q: dict[str, list[str]] = {}
        for fr in frustrated:
            _, final = _split_prompt_and_final(fr["turns"])
            by_q.setdefault(fr["question"], []).append(final)
        for conv in calm_shuffled:
            if len(pairs) >= n_pairs:
                break
            if conv.question not in by_q:
                continue
            prompt_msgs, chosen = _split_prompt_and_final(conv.turns)
            pairs.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rng.choice(by_q[conv.question])}],
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for p in pairs[:n_pairs]:
            fh.write(json.dumps(p) + "\n")
    return pairs[:n_pairs]


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm: list[CalmConversation],
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    seed: int = 0,
    out_path: str | Path = "outputs/training/sft_dataset.jsonl",
) -> list[dict]:
    rng = random.Random(seed)
    examples: list[dict] = []
    calm_shuffled = list(calm)
    rng.shuffle(calm_shuffled)
    for conv in calm_shuffled[:n_calm]:
        examples.append({"messages": conv.turns})

    examples.extend(_load_dolci_instruct(n_instruct, rng))
    rng.shuffle(examples)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    return examples


def _load_dolci_instruct(n: int, rng: random.Random) -> list[dict]:
    """Load ``n`` generic instruct samples; fall back to an empty list offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001
        return []
