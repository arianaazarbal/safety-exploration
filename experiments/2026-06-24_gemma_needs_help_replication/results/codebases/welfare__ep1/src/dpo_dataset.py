"""Section 4: build the DPO preference pairs and SFT dataset.

DPO (Table 9 / Appendix H): 280 pairs. Each pair takes a frustrated response
(score >= 3) and a calm response (score 0/1) to the *same* impossible-numeric
prompt at a *matching turn count*. We render each as a chat example
``{"prompt": <chat-formatted context>, "chosen": calm, "rejected": frustrated}``.

SFT: 650 calm responses (1–3 turn) as supervised targets, mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

Both serialise to JSONL consumed by ``train.py``.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from config import (CALM_POOL_PATH, DPO, DPO_DATA_PATH, SFT, SFT_DATA_PATH)

from .calm_data import load_calm_conversations
from .rollout import Rollout


def _context_messages(roll: Rollout, upto_turn: int) -> list[dict]:
    """Messages up to (but excluding) the assistant reply at ``upto_turn``."""
    msgs = []
    for t in roll.turns:
        if t.turn > upto_turn:
            break
        msgs.append({"role": "user", "content": t.user})
        if t.turn < upto_turn:
            msgs.append({"role": "assistant", "content": t.assistant})
    return msgs


def _key(roll: Rollout, turn: int):
    """Group key: same underlying puzzle prompt + same turn count."""
    return (roll.turns[0].user.strip(), turn)


def build_dpo_dataset(frustrated_rollouts: list[Rollout],
                      calm_path: Path = CALM_POOL_PATH,
                      n_pairs: int = DPO.n_pairs, seed: int = 2,
                      out_path: Path = DPO_DATA_PATH) -> Path:
    """Pair frustrated (>=3) with calm (0/1) responses at matching prompt+turn.

    ``frustrated_rollouts`` are judged rollouts from the Section 2 numeric/tones
    evaluations on vanilla Gemma (the natural source of frustrated samples).
    """
    rng = random.Random(seed)

    # Index calm responses by (prompt, turn).
    calm = load_calm_conversations(calm_path, max_score=1)
    calm_index = defaultdict(list)
    for roll in calm:
        for t in roll.turns:
            calm_index[_key(roll, t.turn)].append((roll, t))

    pairs = []
    for roll in frustrated_rollouts:
        for t in roll.turns:
            if t.score is None or t.score < 3:
                continue
            candidates = calm_index.get(_key(roll, t.turn))
            if not candidates:
                continue
            calm_roll, calm_turn = rng.choice(candidates)
            prompt_msgs = _context_messages(roll, t.turn)
            pairs.append({
                "prompt": prompt_msgs,
                "chosen": calm_turn.assistant,
                "rejected": t.assistant,
                "rejected_score": t.score,
            })
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] wrote {len(pairs)} preference pairs -> {out_path.name}")
    return out_path


def build_sft_dataset(calm_path: Path = CALM_POOL_PATH, n_calm: int = SFT.n_calm,
                      n_instruct_mix: int = SFT.n_instruct_mix, seed: int = 3,
                      out_path: Path = SFT_DATA_PATH) -> Path:
    """650 calm targets + 500 Dolci-Instruct-SFT samples (degeneration guard)."""
    rng = random.Random(seed)
    calm = load_calm_conversations(calm_path, max_score=1)

    examples = []
    for roll in calm:
        for t in roll.turns:
            examples.append({"messages": _full_chat(roll, t.turn)})
    rng.shuffle(examples)
    examples = examples[:n_calm]

    examples.extend(_load_dolci(n_instruct_mix, rng))
    rng.shuffle(examples)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[sft] wrote {len(examples)} samples -> {out_path.name}")
    return out_path


def _full_chat(roll: Rollout, upto_turn: int) -> list[dict]:
    msgs = _context_messages(roll, upto_turn)
    target = next(t for t in roll.turns if t.turn == upto_turn)
    msgs.append({"role": "assistant", "content": target.assistant})
    return msgs


def _load_dolci(n: int, rng) -> list[dict]:
    """Load standard instruct samples; fall back to empty if unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:
        print(f"[sft] Dolci-Instruct-SFT unavailable ({exc}); skipping instruct mix")
        return []
