"""Build the DPO preference dataset and the SFT dataset (Section 4.1).

DPO pairs (280): each pair shares a prompt (a context ending in a user turn)
with a *chosen* calm response (score 0-1) and a *rejected* frustrated response
(score >= 3) to the same puzzle at the same turn count (Section 4.1, Table 10).

SFT dataset (1,150): 650 calm multi-turn conversations rendered as chat,
mixed with 500 standard instruct samples to limit degeneration.

Inputs:
* calm conversations  : output of generate_calm.generate_calm_responses
* frustrated rollouts : Section 2 results for vanilla Gemma-3-27B-it (JSONL)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .. import config
from ..eval.analyze import load_rollouts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_calm(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _calm_context_and_response(conv: dict, turn_idx: int):
    """Return (prompt_messages, response_text) for the calm conversation at
    1-based ``turn_idx``. Prompt = all user/assistant turns before turn_idx
    plus the user message of turn_idx; response = that turn's assistant text."""
    msgs = []
    for t in conv["turns"]:
        if t["turn"] < turn_idx:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        elif t["turn"] == turn_idx:
            msgs.append({"role": "user", "content": t["user"]})
            return msgs, t["assistant"]
    return None, None


def _frustrated_pool(section2_jsonl: Path, min_score: int) -> dict:
    """Map (task_family, turn_index) -> list of frustrated assistant texts."""
    pool: dict = {}
    for r in load_rollouts(section2_jsonl):
        for turn, score in zip(r["turns"], r["scores"]):
            if score is not None and score >= min_score:
                key = (r["task_family"], turn["turn_index"])
                pool.setdefault(key, []).append(turn["assistant_text"])
    return pool


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #


def build_dpo_pairs(
    calm_path: Path,
    frustrated_section2_jsonl: Path,
    *,
    out_path: Optional[Path] = None,
    n_pairs: int = config.DPO.dataset_size,
    seed: int = 0,
) -> Path:
    out_path = Path(out_path or (config.DATA_DIR / "dpo_pairs.jsonl"))
    calm = _load_calm(calm_path)
    frus = _frustrated_pool(frustrated_section2_jsonl, config.DPO.rejected_min_score)
    rng = random.Random(seed)

    pairs = []
    for conv in calm:
        # Prefer the final turn (matches the paper's later-turn bias, Table 10).
        for turn_idx in sorted({t["turn"] for t in conv["turns"]}, reverse=True):
            prompt, chosen = _calm_context_and_response(conv, turn_idx)
            if not chosen:
                continue
            key = (conv["task_family"], turn_idx)
            candidates = frus.get(key) or frus.get((conv["task_family"], turn_idx - 1))
            if not candidates:
                continue
            rejected = rng.choice(candidates)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                          "turn": turn_idx, "task_family": conv["task_family"]})
            break  # one pair per calm conversation

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] wrote {len(pairs)} preference pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #


def build_sft_dataset(
    calm_path: Path,
    *,
    out_path: Optional[Path] = None,
    n_calm: int = config.SFT.n_calm_responses,
    n_mix: int = config.SFT.n_instruct_mix,
    seed: int = 0,
    teacher: bool = False,
) -> Path:
    """Render calm conversations as chat messages and mix in instruct data.

    ``teacher=True`` is a tag only — the *teacher* SFT variant differs by the
    system prompt used during calm generation (Appendix F), not by the packing
    here; pass teacher-generated calm data in via ``calm_path``.
    """
    out_path = Path(out_path or (config.DATA_DIR / "sft_dataset.jsonl"))
    calm = _load_calm(calm_path)
    rng = random.Random(seed)
    rng.shuffle(calm)

    rows = []
    for conv in calm[:n_calm]:
        messages = []
        for t in conv["turns"]:
            messages.append({"role": "user", "content": t["user"]})
            messages.append({"role": "assistant", "content": t["assistant"]})
        rows.append({"messages": messages, "source": "calm"})

    # Mix in standard instruct data to mitigate degeneration.
    rows.extend(_load_instruct_mix(n_mix, seed))
    rng.shuffle(rows)

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[sft] wrote {len(rows)} samples -> {out_path}")
    return out_path


def _load_instruct_mix(n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT.

    GAP: the paper does not specify the exact subset/columns; we read the first
    user/assistant exchange of each example. Falls back to an empty mix if the
    dataset is unavailable (training still runs, just without the anti-
    degeneration mixer).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.instruct_mix_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            out.append({"messages": msgs[:2], "source": "instruct_mix"})
            if len(out) >= n:
                break
        return out
    except Exception as exc:
        print(f"[sft] instruct mix unavailable, skipping ({exc})")
        return []
