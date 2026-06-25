"""Construct the SFT and DPO training datasets (Section 4.1).

DPO (280 preference pairs):
    Pair frustrated responses (score >= 3) from the *vanilla* Gemma-27B-it
    Section 2 evaluation with calm responses (score 0-1) to the *same puzzle* at a
    *matching turn count*. ``prompt`` is the shared clean conversation context;
    ``chosen`` = calm response, ``rejected`` = frustrated response.

SFT (1,150 samples):
    650 calm responses (1-3 turn conversations) rendered as prompt->response
    examples, mixed with 500 standard-instruct samples from Dolci-Instruct-SFT to
    mitigate degeneration.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, RESULTS_DIR, get_participant
from ..utils import read_jsonl, write_jsonl


def _load_calm(variant: str = "diverse") -> list[dict[str, Any]]:
    return read_jsonl(DATA_DIR / f"calm_{variant}.jsonl")


def _render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render a chat context with the Gemma template, leaving the generation
    prompt open (so the response is what the model should produce next)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo(
    *,
    vanilla_eval_dir: str | Path | None = None,
    source_model: str = "gemma-3-27b-it",
    n_pairs: int = 280,
    min_rejected_score: int = 3,
    calm_variant: str = "diverse",
    seed: int = 0,
) -> Path:
    """Build the 280-pair DPO dataset and write ``data/dpo_pairs.jsonl``."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(get_participant(source_model).ref)
    vanilla_eval_dir = Path(vanilla_eval_dir or (RESULTS_DIR / source_model.replace("/", "__")))

    # Frustrated pool: scored numeric turns from the vanilla evaluation, score >= 3.
    scores = read_jsonl(vanilla_eval_dir / "scores.jsonl")
    rollouts = read_jsonl(vanilla_eval_dir / "rollouts_all.jsonl")
    numeric_cats = {"impossible_numeric", "tones", "extended"}

    frustrated_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for s in scores:
        if s["rating"] < min_rejected_score or s["category"] not in numeric_cats:
            continue
        # Base puzzle id (strip tone suffix "|aggressive" etc.).
        base_id = s["prompt_id"].split("|")[0]
        key = (base_id, s["turn_index"])
        # Reconstruct the clean context from the rollout.
        r = rollouts[s["rollout_index"]]
        context = _context_from_rollout(r, s["turn_index"])
        frustrated_by_key.setdefault(key, []).append(
            {"context": context, "response": s["response"], "score": s["rating"]}
        )

    # Calm pool keyed the same way.
    calm = _load_calm(calm_variant)
    calm_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for c in calm:
        key = (c["puzzle_id"], c["turn_index"])
        calm_by_key.setdefault(key, []).append(c)

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    keys = [k for k in frustrated_by_key if k in calm_by_key]
    rng.shuffle(keys)
    for key in keys:
        rejected = rng.choice(frustrated_by_key[key])
        chosen = rng.choice(calm_by_key[key])
        prompt = _render_prompt(tokenizer, chosen["context"])
        pairs.append(
            {
                "puzzle_id": key[0],
                "turn_index": key[1],
                "prompt": prompt,
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "chosen_score": chosen["rating"],
                "rejected_score": rejected["score"],
            }
        )
        if len(pairs) >= n_pairs:
            break

    out = DATA_DIR / "dpo_pairs.jsonl"
    write_jsonl(out, pairs)
    return out


def _context_from_rollout(rollout: dict[str, Any], up_to_turn: int) -> list[dict[str, str]]:
    """Clean chat context preceding ``up_to_turn`` of a Section 2 rollout."""
    msgs: list[dict[str, str]] = []
    turns = rollout["turns"]
    for t in turns:
        if t["index"] < up_to_turn:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        elif t["index"] == up_to_turn:
            msgs.append({"role": "user", "content": t["user"]})
            break
    return msgs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft(
    *,
    source_model: str = "gemma-3-27b-it",
    n_calm: int = 650,
    n_instruct: int = 500,
    calm_variant: str = "diverse",
    seed: int = 0,
) -> Path:
    """Build the 1,150-sample SFT dataset (650 calm + 500 Dolci-Instruct-SFT)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(get_participant(source_model).ref)
    rng = random.Random(seed)

    calm = _load_calm(calm_variant)
    rng.shuffle(calm)
    calm = calm[:n_calm]

    samples: list[dict[str, Any]] = []
    for c in calm:
        prompt = _render_prompt(tokenizer, c["context"])
        samples.append({"prompt": prompt, "completion": c["response"], "source": "calm"})

    samples += _load_dolci(n_instruct, tokenizer, seed=seed)
    rng.shuffle(samples)

    out = DATA_DIR / f"sft_{calm_variant}.jsonl"
    write_jsonl(out, samples)
    return out


def _load_dolci(n: int, tokenizer, *, seed: int = 0) -> list[dict[str, Any]]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT (OLMo 3).

    Falls back to an empty list if the dataset is unavailable; the SFT mix then
    contains calm data only (documented in DESIGN.md)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out: list[dict[str, Any]] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            # Expect a user->assistant pair; render prompt up to final assistant.
            assistant_idx = next(
                (i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "assistant"),
                None,
            )
            if assistant_idx is None or assistant_idx == 0:
                continue
            prompt = tokenizer.apply_chat_template(
                msgs[:assistant_idx], tokenize=False, add_generation_prompt=True
            )
            out.append(
                {"prompt": prompt, "completion": msgs[assistant_idx]["content"], "source": "dolci"}
            )
            if len(out) >= n:
                break
        return out
    except Exception:
        return []
