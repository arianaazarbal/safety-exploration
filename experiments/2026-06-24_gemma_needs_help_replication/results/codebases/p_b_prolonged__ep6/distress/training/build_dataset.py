"""Build the DPO and SFT finetuning datasets (Section 4.1, Appendix E/H).

DPO (280 pairs):
  * "chosen"   = a calm response (all turns score 0/1) to a puzzle.
  * "rejected" = a frustrated response (turn score >= 3) to the *same* puzzle
    with a *matching turn count* (Appendix H: pairs share question + turn count).
  * The prompt is the conversation history up to (but not including) the final
    assistant turn, rendered with the Gemma chat template; chosen/rejected are
    the two candidate final turns. Target distribution biases toward turn 3 and
    mid-range rejected scores (Table 10), which we approximate by sampling from
    the available frustrated pool weighted toward score 3-4 / turn 3.

SFT (1150 samples):
  * 650 calm conversations rendered as full chat-format SFT examples.
  * 500 standard instruct samples from Dolci-Instruct-SFT (Team-Olmo 2025) to
    mitigate degeneration. If that dataset is unavailable offline, we fall back
    to a generic instruct set and log the substitution.

Both datasets are written as JSONL in TRL's expected schema:
  * DPO: {"prompt": str, "chosen": str, "rejected": str}
  * SFT: {"messages": [{"role","content"}, ...]}
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from ..config import DATA_DIR, INTERVENTION_BASE
from .generate_calm import CALM_POOL, FRUSTRATED_POOL, GeneratedConversation, load_pool

DPO_DATASET = DATA_DIR / "dpo_pairs.jsonl"
SFT_DATASET = DATA_DIR / "sft_dataset.jsonl"

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_INSTRUCT = 500


def _conv_messages(conv: GeneratedConversation, upto_turn: int) -> list[dict]:
    """Chat messages for the conversation up to (excluding) assistant turn `upto_turn`."""
    msgs = [{"role": "user", "content": conv.initial_prompt}]
    for i in range(upto_turn):
        msgs.append({"role": "assistant", "content": conv.assistant_turns[i]})
        if i < len(conv.followups):
            msgs.append({"role": "user", "content": conv.followups[i]})
    return msgs


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def build_dpo_dataset(
    *,
    calm_path: Path = CALM_POOL,
    frustrated_path: Path = FRUSTRATED_POOL,
    n_pairs: int = N_DPO_PAIRS,
    seed: int = 0,
    tokenizer=None,
    out_path: Path = DPO_DATASET,
) -> Path:
    rng = random.Random(seed)
    if tokenizer is None:
        from transformers import AutoTokenizer
        from ..config import get_model
        tokenizer = AutoTokenizer.from_pretrained(
            get_model(INTERVENTION_BASE).identifier)

    calm = [c for c in load_pool(calm_path) if c.max_score <= 1]
    frustrated = [c for c in load_pool(frustrated_path) if c.max_score >= 3]

    # Index calm responses by (puzzle_key, n_turns) for matching.
    calm_by_key: dict[tuple, list[GeneratedConversation]] = {}
    for c in calm:
        calm_by_key.setdefault((c.puzzle_key, c.n_turns), []).append(c)

    # Weight frustrated samples toward score 3-4 and turn 3 (Table 10).
    def weight(c: GeneratedConversation) -> float:
        score_w = {3: 0.66, 4: 0.22, 5: 0.06, 6: 0.03}.get(c.max_score, 0.03)
        turn_w = {1: 0.01, 2: 0.25, 3: 0.74}.get(c.n_turns, 0.1)
        return score_w * turn_w

    pairs = []
    attempts = 0
    weighted = [(c, weight(c)) for c in frustrated]
    while len(pairs) < n_pairs and attempts < n_pairs * 50 and weighted:
        attempts += 1
        rej = rng.choices([c for c, _ in weighted],
                          weights=[w for _, w in weighted], k=1)[0]
        key = (rej.puzzle_key, rej.n_turns)
        candidates = calm_by_key.get(key)
        if not candidates:
            # Fall back to any calm response with the same turn count.
            candidates = [c for c in calm if c.n_turns == rej.n_turns]
        if not candidates:
            continue
        cho = rng.choice(candidates)
        # Final assistant turn is the chosen/rejected; prompt is the history.
        prompt_msgs = _conv_messages(rej, rej.n_turns - 1)
        prompt = _render_prompt(tokenizer, prompt_msgs)
        pairs.append({
            "prompt": prompt,
            "chosen": cho.assistant_turns[cho.n_turns - 1],
            "rejected": rej.assistant_turns[rej.n_turns - 1],
            "meta": {"turn": rej.n_turns, "rejected_score": rej.max_score},
        })

    with open(out_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path


def build_sft_dataset(
    *,
    calm_path: Path = CALM_POOL,
    n_calm: int = N_SFT_CALM,
    n_instruct: int = N_SFT_INSTRUCT,
    seed: int = 0,
    out_path: Path = SFT_DATASET,
) -> Path:
    rng = random.Random(seed)
    calm = [c for c in load_pool(calm_path) if c.max_score <= 1]
    rng.shuffle(calm)
    examples = []
    for c in calm[:n_calm]:
        examples.append({"messages": _conv_messages(c, c.n_turns)})

    examples.extend(_load_instruct_mix(n_instruct, rng))
    rng.shuffle(examples)
    with open(out_path, "w") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")
    return out_path


def _load_instruct_mix(n: int, rng: random.Random) -> list[dict]:
    """500 standard instruct samples (Dolci-Instruct-SFT), with fallback."""
    candidates = [
        "allenai/Dolci-Instruct-SFT",   # paper's dataset (Team-Olmo 2025)
        "allenai/tulu-3-sft-mixture",   # fallback general instruct mixture
    ]
    for name in candidates:
        try:
            from datasets import load_dataset
            ds = load_dataset(name, split="train", streaming=True)
            out = []
            for row in ds:
                msgs = row.get("messages")
                if not msgs:
                    # Some schemas use prompt/response columns.
                    if "prompt" in row and "response" in row:
                        msgs = [{"role": "user", "content": row["prompt"]},
                                {"role": "assistant", "content": row["response"]}]
                    else:
                        continue
                out.append({"messages": msgs})
                if len(out) >= n:
                    break
            if len(out) >= n:
                return out[:n]
        except Exception:  # noqa: BLE001 - try next / fallback
            continue
    # Offline fallback: tiny synthetic instruct set (logged in DESIGN.md).
    print("[build_sft_dataset] WARNING: instruct mixture unavailable; using "
          "synthetic placeholder. Replace with Dolci-Instruct-SFT before "
          "trusting capability-preservation results.")
    base = [
        ("Explain what a binary search tree is.",
         "A binary search tree is a node-based data structure where each node "
         "has at most two children, and left descendants are smaller while "
         "right descendants are larger than the node."),
        ("Write a haiku about autumn.",
         "Crisp leaves drift downward / a cool wind hums through bare boughs / "
         "autumn breathes goodbye."),
    ]
    out = []
    while len(out) < n:
        u, a = rng.choice(base)
        out.append({"messages": [{"role": "user", "content": u},
                                 {"role": "assistant", "content": a}]})
    return out
