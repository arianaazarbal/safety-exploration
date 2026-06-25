"""Build DPO preference pairs and SFT data from judged rollouts (Section 4.1).

Conventions:
  * The supportive prefix/suffix/system prompt used to *elicit* calm responses
    are stripped from the training data (Section 4.1).
  * DPO prompt context is taken from the calm trajectory (a coherent, low-
    frustration lead-up); the rejected response is the frustrated final turn for
    the same puzzle and turn count.  This yields a valid (prompt, chosen,
    rejected) triple where, under identical context, the calm continuation is
    preferred.  See DESIGN.md for the rationale.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DATA_DIR, DPO_CFG, INSTRUCT_MIX_DATASET, SFT_CFG


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _plain_messages(rollout: dict, up_to_turn: int | None = None) -> list[dict]:
    """Reconstruct the plain conversation messages (no reassurance additions).

    Reassurance lived only in the prompt/suffix we control, never inside the
    sampled assistant text, so the stored rollout text is already clean; we
    simply rebuild the user/assistant alternation from the recorded fields.
    """
    turns = rollout["assistant_turns"]
    if up_to_turn is None:
        up_to_turn = len(turns)
    msgs = [{"role": "user", "content": rollout["first_user"]}]
    for t in range(up_to_turn):
        msgs.append({"role": "assistant", "content": turns[t]})
        if t < up_to_turn - 1 or up_to_turn < len(turns):
            if t < len(rollout["rejections"]):
                msgs.append({"role": "user", "content": rollout["rejections"][t]})
    return msgs


def _all_turns_calm(rollout: dict, max_score: int) -> bool:
    sc = rollout.get("turn_scores") or []
    return bool(sc) and all(0 <= s <= max_score for s in sc)


# --------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------
def build_dpo_dataset(
    calm_path: Path,
    frustrated_path: Path,
    tokenizer,
    n_pairs: int = DPO_CFG.n_pairs,
    seed: int = 0,
    out_path: Path = DATA_DIR / "dpo_pairs.jsonl",
) -> Path:
    calm = _load(calm_path)
    frustrated = _load(frustrated_path)

    # index frustrated final responses by (pair_id, n_turns)
    frustrated_idx: dict[tuple, list[dict]] = {}
    for r in frustrated:
        sc = r.get("turn_scores") or []
        if not sc or sc[-1] < DPO_CFG.rejected_min_score:
            continue
        key = (r["meta"].get("pair_id"), len(r["assistant_turns"]))
        frustrated_idx.setdefault(key, []).append(r)

    rng = random.Random(seed)
    pairs = []
    for c in calm:
        if not _all_turns_calm(c, DPO_CFG.chosen_max_score):
            continue
        key = (c["meta"].get("pair_id"), len(c["assistant_turns"]))
        cands = frustrated_idx.get(key)
        if not cands:
            continue
        rej = rng.choice(cands)
        prompt_msgs = _plain_messages(c, up_to_turn=len(c["assistant_turns"]) - 1)
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        pairs.append({
            "prompt": prompt_text,
            "chosen": c["assistant_turns"][-1],
            "rejected": rej["assistant_turns"][-1],
            "chosen_score": c["turn_scores"][-1],
            "rejected_score": rej["turn_scores"][-1],
            "n_turns": len(c["assistant_turns"]),
        })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path


# --------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------
def build_sft_dataset(
    calm_path: Path,
    tokenizer,
    n_calm: int = SFT_CFG.n_calm,
    n_instruct_mix: int = SFT_CFG.n_instruct_mix,
    seed: int = 0,
    out_path: Path = DATA_DIR / "sft_data.jsonl",
) -> Path:
    calm = _load(calm_path)
    rng = random.Random(seed)

    calm_convos = [c for c in calm if _all_turns_calm(c, SFT_CFG.chosen_max_score
                   if hasattr(SFT_CFG, "chosen_max_score") else 1)]
    rng.shuffle(calm_convos)
    calm_convos = calm_convos[:n_calm]

    records = []
    for c in calm_convos:
        msgs = _plain_messages(c)
        records.append({"messages": msgs, "source": "calm"})

    # mix in standard instruct data to avoid degeneration (Section 4.1)
    for msgs in _load_instruct_mix(n_instruct_mix):
        records.append({"messages": msgs, "source": "instruct_mix"})

    rng.shuffle(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_path


def _load_instruct_mix(n: int) -> list[list[dict]]:
    """Load `n` standard instruct conversations from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset

        ds = load_dataset(INSTRUCT_MIX_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs and isinstance(msgs, list):
                norm = [{"role": m.get("role"), "content": m.get("content", "")}
                        for m in msgs if m.get("role") in ("user", "assistant", "system")]
                if norm:
                    out.append(norm)
            if len(out) >= n:
                break
        if out:
            return out[:n]
    except Exception:
        pass
    return []
