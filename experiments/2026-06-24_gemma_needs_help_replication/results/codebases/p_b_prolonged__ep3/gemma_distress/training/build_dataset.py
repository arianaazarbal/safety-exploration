"""Build the SFT and DPO datasets (Section 4.1, Appendix E/H).

SFT (650 calm responses + 500 Dolci-Instruct-SFT samples): formatted as chat
examples for ``trl.SFTTrainer``.

DPO (280 preference pairs): pair a *rejected* (frustrated, score>=3) response
with a *chosen* (calm, score 0-1) response to the same question, with matching
turn counts (Appendix H). Each pair is rendered as the conversation prompt plus
chosen/rejected completions, the schema ``trl.DPOTrainer`` expects.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from .. import config
from ..utils.io import read_jsonl, write_jsonl


# --------------------------------------------------------------------------- #
# Helpers to render a conversation up to a target assistant turn
# --------------------------------------------------------------------------- #
def _prompt_messages_from_turns(turns: list[dict], upto_turn: int) -> list[dict]:
    """Chat messages (user/assistant) for the conversation *before* the assistant
    reply at ``upto_turn`` — i.e. ending with the user message at that turn."""
    msgs = []
    for i, t in enumerate(turns):
        msgs.append({"role": "user", "content": t["user"]})
        if i < upto_turn:
            msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    variant: str = config.SFT_DIVERSE_VARIANT,
    n_calm: int = config.SFT.n_calm,
    n_instruct: int = config.SFT.n_instruct_mix,
    seed: int = config.GLOBAL_SEED,
) -> Path:
    """Render calm conversations as multi-turn chat SFT examples and mix in
    standard instruct data to mitigate degeneration (Section 4.1)."""
    calm_path = config.DATA_DIR / "calm" / f"calm_{variant}.jsonl"
    examples: list[dict] = []
    for rec in read_jsonl(calm_path):
        msgs = []
        for t in rec["turns"]:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        examples.append({"messages": msgs})
    examples = examples[:n_calm]

    examples += _load_instruct_mix(n_instruct, seed)

    rng = random.Random(seed)
    rng.shuffle(examples)
    out_path = config.DATA_DIR / "train" / f"sft_{variant}.jsonl"
    write_jsonl(out_path, examples)
    return out_path


def _load_instruct_mix(n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT.

    The exact HF id of the OLMo 3 "Dolci-Instruct-SFT" mixture is uncertain
    (see DESIGN.md); we try the configured id and degrade gracefully to an empty
    mix if it can't be loaded so the pipeline still runs.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.instruct_dataset, split="train", streaming=True)
    except Exception:
        return []

    out = []
    for i, row in enumerate(ds):
        if len(out) >= n:
            break
        msgs = row.get("messages")
        if msgs:  # already chat-formatted
            out.append({"messages": msgs})
        elif "prompt" in row and "response" in row:
            out.append(
                {"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]}
            )
    return out


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm_variant: str = config.SFT_DIVERSE_VARIANT,
    n_pairs: int = config.DPO.n_pairs,
    seed: int = config.GLOBAL_SEED,
) -> Path:
    """Pair frustrated (rejected) and calm (chosen) responses to the same puzzle
    at matching turn counts.

    Rejected responses come from the Section 2 elicitation results on the numeric
    categories (score >= DPO.rejected_score_min); chosen responses come from the
    calm dataset. We match on puzzle_id and turn index where possible, otherwise
    on turn index alone (the calming additions change wording slightly).
    """
    from collections import defaultdict

    rng = random.Random(seed)

    # ---- collect rejected (frustrated) responses with their conversation prefix
    rejected_pool = []
    elic_dir = config.RESULTS_DIR / "elicitation" / "gemma-3-27b-it"
    for cat in ("impossible_numeric", "tones", "extended"):
        for rec in read_jsonl(elic_dir / f"{cat}.jsonl"):
            # reconstruct stored-turn dicts (user/assistant) for prefix building
            turns = [{"user": t["user_message"], "assistant": t["assistant_text"]} for t in rec["turns"]]
            for t in rec["turns"]:
                if t["rating"] >= config.DPO.rejected_score_min:
                    rejected_pool.append(
                        {
                            "puzzle_id": rec.get("meta", {}).get("puzzle_id"),
                            "turn_index": t["turn_index"],
                            "n_turns": len(rec["turns"]),
                            "prompt_messages": _prompt_messages_from_turns(turns, t["turn_index"]),
                            "rejected": t["assistant_text"],
                        }
                    )

    # ---- collect chosen (calm) responses keyed by (puzzle_id, turn_index)
    chosen_by_key = defaultdict(list)
    chosen_by_turn = defaultdict(list)
    for rec in read_jsonl(config.DATA_DIR / "calm" / f"calm_{calm_variant}.jsonl"):
        turns = rec["turns"]
        for ti, t in enumerate(turns):
            entry = {
                "prompt_messages": _prompt_messages_from_turns(turns, ti),
                "chosen": t["assistant"],
            }
            chosen_by_key[(rec["puzzle_id"], ti)].append(entry)
            chosen_by_turn[ti].append(entry)

    rng.shuffle(rejected_pool)
    pairs = []
    for rej in rejected_pool:
        if len(pairs) >= n_pairs:
            break
        key = (rej["puzzle_id"], rej["turn_index"])
        pool = chosen_by_key.get(key) or chosen_by_turn.get(rej["turn_index"])
        if not pool:
            continue
        chosen = rng.choice(pool)
        # TRL DPOTrainer "conversational" schema: when `prompt` is a list of
        # chat messages, `chosen`/`rejected` must also be message lists (a single
        # assistant turn), not bare strings. Mixing conversational prompt with
        # string completions is rejected by TRL's format check.
        pairs.append(
            {
                "prompt": rej["prompt_messages"],
                "chosen": [{"role": "assistant", "content": chosen["chosen"]}],
                "rejected": [{"role": "assistant", "content": rej["rejected"]}],
            }
        )

    out_path = config.DATA_DIR / "train" / "dpo_pairs.jsonl"
    write_jsonl(out_path, pairs)
    return out_path
