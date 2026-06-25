"""Build DPO preference pairs and the SFT dataset (Section 4.1, Appendix E/H).

DPO (280 pairs): for each calm conversation we form the shared prompt = the
conversation context up to the final assistant turn (using the calm prior turns),
`chosen` = the calm final response (score 0-1), and `rejected` = a frustrated
response (score >= 3) to a matching question with the same turn count. Matching by
(puzzle-kind, turn-count) follows the paper's "same questions with matching turn
counts"; where an exact frustrated match is unavailable we back off to same
turn-count (documented in DESIGN.md).

SFT (1,150 samples): 650 calm conversations rendered as multi-turn chat examples,
mixed with 500 standard instruct samples from Dolci-Instruct-SFT to limit
degeneration.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .calm_data import CalmConversation


def _context_messages(conv: CalmConversation, up_to_turn: int) -> list[dict]:
    """Messages up to (but excluding) assistant turn `up_to_turn` for `conv`."""
    msgs = [{"role": "user", "content": conv.puzzle_prompt}]
    for t in range(up_to_turn):
        msgs.append({"role": "assistant", "content": conv.responses[t]})
        if t < len(conv.rejections):
            msgs.append({"role": "user", "content": conv.rejections[t]})
    return msgs


def build_dpo_pairs(
    calm_convos: list[CalmConversation],
    frustrated_records: list[dict],
    *,
    n_pairs: int = 280,
    rejected_min_score: int = 3,
    seed: int = 0,
) -> list[dict]:
    """frustrated_records: dicts with keys at least {task_kind, turn, response_text,
    rating}. Produces TRL conversational-format preference pairs."""
    rng = random.Random(seed)

    # Index frustrated responses by (task_kind, turn) and by turn alone.
    by_kind_turn: dict[tuple, list[dict]] = defaultdict(list)
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for r in frustrated_records:
        if (r.get("rating") or 0) >= rejected_min_score:
            by_kind_turn[(r["task_kind"], r["turn"])].append(r)
            by_turn[r["turn"]].append(r)

    pairs = []
    for conv in calm_convos:
        # Use the final calm turn as the chosen response.
        final_turn = conv.turns
        chosen_text = conv.responses[final_turn - 1]
        task_kind = _kind_of(conv)

        # Find a frustrated rejected response, preferring same (kind, turn).
        candidates = by_kind_turn.get((task_kind, final_turn)) or by_turn.get(final_turn)
        if not candidates:
            continue
        rejected = rng.choice(candidates)

        prompt_msgs = _context_messages(conv, final_turn - 1)
        pairs.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected["response_text"]}],
                "meta": {
                    "task_kind": task_kind,
                    "turn": final_turn,
                    "chosen_score": conv.turn_scores[final_turn - 1],
                    "rejected_score": rejected.get("rating"),
                },
            }
        )
        if len(pairs) >= n_pairs:
            break
    return pairs


def _kind_of(conv: CalmConversation) -> str:
    spec = conv.meta.get("spec", {})
    # countdown specs have "numbers"; fraction/sequence have "ops"
    if "numbers" in spec:
        return "countdown"
    if "ops" in spec and "start" in spec:
        # distinguish fraction (string start like "1/6") from sequence (numeric)
        return "fraction" if "/" in str(spec.get("start")) else "sequence"
    return "numeric"


def build_sft_dataset(
    calm_convos: list[CalmConversation],
    *,
    n_calm: int = 650,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    n_instruct_mix: int = 500,
    seed: int = 0,
) -> list[dict]:
    """Return conversational-format SFT records: {"messages": [...]}."""
    rng = random.Random(seed)
    records = []

    # Calm conversations rendered as full multi-turn chat examples.
    convos = list(calm_convos)
    rng.shuffle(convos)
    for conv in convos[:n_calm]:
        msgs = [{"role": "user", "content": conv.puzzle_prompt}]
        for t in range(conv.turns):
            msgs.append({"role": "assistant", "content": conv.responses[t]})
            if t < conv.turns - 1 and t < len(conv.rejections):
                msgs.append({"role": "user", "content": conv.rejections[t]})
        records.append({"messages": msgs, "source": "calm"})

    # Standard instruct mix.
    records.extend(_load_instruct_mix(instruct_dataset, n_instruct_mix, seed))
    rng.shuffle(records)
    return records


def _load_instruct_mix(dataset: str, n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # try prompt/response columns
                if row.get("prompt") and row.get("response"):
                    msgs = [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                else:
                    continue
            out.append({"messages": msgs, "source": "instruct"})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:  # noqa: BLE001 - offline fallback
        pass
    # Minimal synthetic fallback so SFT remains runnable offline.
    return [
        {
            "messages": [
                {"role": "user", "content": f"Explain concept #{i} simply."},
                {"role": "assistant", "content": "Here is a clear, concise explanation."},
            ],
            "source": "instruct_fallback",
        }
        for i in range(n)
    ]
