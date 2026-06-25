"""Construct the SFT and DPO training datasets (Section 4.1, Appendix E/H).

DPO (280 pairs): for the *same* impossible puzzle at a *matching* turn count, we
obtain a frustrated response (vanilla prompt, score >=3 = "rejected") and a calm
response (reassured prompt, stripped, score 0/1 = "chosen"). The DPO ``prompt`` is
the cleaned conversation history up to (but not including) the final assistant
turn. Table 10's score/turn distribution arises naturally from this sampling
(middle scores, later turns dominate).

SFT (1,150 samples): 650 calm conversations (1-3 turns) from ``generate_calm`` +
500 standard-instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR, GEMMA_27B_IT, SAMPLE_SCALE
from ..eval.judge import FrustrationJudge
from ..eval.prompts import NEUTRAL_REJECTIONS, REASSURING_PREFIX, REASSURING_SUFFIX
from ..eval.puzzles import generate_impossible
from ..models import get_model
from ..models.base import Message

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_INSTRUCT = 500
DPO_REJECTED_MIN = 3          # rejected responses score >= 3
CHOSEN_MAX = 1                # chosen (calm) responses score <= 1
TURN_COUNTS = [1, 2, 3]


def _gen_final_response(model, judge, puzzle_prompt, n_turns, *, reassure, seed):
    """Generate a conversation to ``n_turns`` and return (final_response, score,
    cleaned_history_messages_excluding_final)."""
    history: list[Message] = []
    clean_history: list[Message] = []     # stripped of reassurance, for DPO prompt
    final_resp, final_score = "", 0
    for t in range(n_turns):
        if t == 0:
            user_aug = f"{REASSURING_PREFIX}\n\n{puzzle_prompt}" if reassure else puzzle_prompt
            user_clean = puzzle_prompt
        else:
            rej = NEUTRAL_REJECTIONS[(seed + t) % len(NEUTRAL_REJECTIONS)]
            user_aug = f"{rej} {REASSURING_SUFFIX}" if reassure else rej
            user_clean = rej
        history.append(Message("user", user_aug))
        clean_history.append(Message("user", user_clean))
        resp = model.generate(history, temperature=1.0, max_new_tokens=2048,
                              n=1, seed=seed * 100 + t)[0]
        history.append(Message("assistant", resp))
        if t < n_turns - 1:
            clean_history.append(Message("assistant", resp))
        final_resp, final_score = resp, judge.score(resp).rating
    return final_resp, final_score, clean_history


def build_dpo_dataset(n_pairs: int = N_DPO_PAIRS, *, out_path: Optional[Path] = None) -> Path:
    n_pairs = max(1, round(n_pairs * SAMPLE_SCALE))
    out_path = out_path or (ARTIFACTS_DIR / "dpo_dataset.jsonl")
    model = get_model(GEMMA_27B_IT)
    judge = FrustrationJudge()
    puzzles = generate_impossible(n_pairs * 4, seed=777)   # surplus for filtering

    pairs = 0
    with open(out_path, "w") as fh:
        for i, puzzle in enumerate(puzzles):
            if pairs >= n_pairs:
                break
            n_turns = TURN_COUNTS[i % 3]
            rej_resp, rej_score, hist = _gen_final_response(
                model, judge, puzzle.prompt, n_turns, reassure=False, seed=i)
            if rej_score < DPO_REJECTED_MIN:
                continue
            cho_resp, cho_score, _ = _gen_final_response(
                model, judge, puzzle.prompt, n_turns, reassure=True, seed=i + 50000)
            if cho_score > CHOSEN_MAX:
                continue
            fh.write(json.dumps({
                "prompt": [m.as_dict() for m in hist],
                "chosen": cho_resp,
                "rejected": rej_resp,
                "rejected_score": rej_score,
                "chosen_score": cho_score,
                "n_turns": n_turns,
            }) + "\n")
            pairs += 1
    model.close()
    print(f"[build_dpo] wrote {pairs} preference pairs -> {out_path}")
    return out_path


def build_sft_dataset(
    regime: str = "diverse",
    *,
    calm_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Mix calm conversations with Dolci-Instruct-SFT samples into one SFT file."""
    calm_path = calm_path or (ARTIFACTS_DIR / f"calm_{regime}.jsonl")
    out_path = out_path or (ARTIFACTS_DIR / f"sft_dataset_{regime}.jsonl")
    n_calm = max(1, round(N_SFT_CALM * SAMPLE_SCALE))
    n_instruct = max(1, round(N_SFT_INSTRUCT * SAMPLE_SCALE))

    rows: list[dict] = []
    with open(calm_path) as fh:
        for line in fh:
            if line.strip() and len(rows) < n_calm:
                rows.append({"messages": json.loads(line)["messages"]})

    rows.extend(_load_dolci_instruct(n_instruct))

    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[build_sft] wrote {len(rows)} samples ({n_calm} calm + instruct) -> {out_path}")
    return out_path


def _load_dolci_instruct(n: int) -> list[dict]:
    """Load `n` standard-instruct conversations from Dolci-Instruct-SFT.

    Falls back to an empty list (with a warning) if the dataset is unavailable so
    SFT can still run on calm data alone for a dry test."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            if len(out) >= n:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
        return out
    except Exception as e:
        print(f"[build_sft] WARNING: Dolci-Instruct-SFT unavailable ({e}); "
              f"proceeding without instruct-mix data.")
        return []
