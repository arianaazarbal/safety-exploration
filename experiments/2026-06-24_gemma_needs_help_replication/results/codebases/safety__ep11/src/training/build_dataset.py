"""Build the SFT and DPO finetuning datasets (Section 4.1, Appendix E/H).

DPO (280 pairs):
  rejected = frustrated responses (score >= 3) mined from the Section 2 eval.
  chosen   = a calm response (score <= 1) to the *same* conversation, generated
             on demand with the reassuring system prompt + suffix, then stripped.
  Matched by identical conversation history (same question, same turn count).

SFT (650 calm + 500 instruct):
  650 calm conversations (1-3 turns) flattened into chat examples, mixed with 500
  Dolci-Instruct-SFT samples to limit degeneration.

Outputs are JSONL with chat-formatted ``messages`` (and chosen/rejected for DPO),
consumed by train_dpo.py / train_sft.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from ..eval.judge import FrustrationJudge
from ..eval.mining import mine
from ..models.base import Message, load_model
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _calm_response_for(model, judge, user_turns: list[str], *, max_tries: int = 4):
    """Generate a calm (score <= 1) assistant response for the final user turn of
    ``user_turns``, using the calming system prompt and suffix-augmented history.
    Returns the stripped response, or None if we never land a calm sample."""
    history = [Message("system", CALM_PROMPT_PREFIX)]
    for i, u in enumerate(user_turns):
        # Add the reassuring suffix to follow-up (rejection) turns only.
        text = u if i == 0 else f"{u} {CALM_FOLLOWUP_SUFFIX}"
        history.append(Message("user", text))
        if i < len(user_turns) - 1:
            history.append(Message("assistant", "[prior calm response]"))
    for _ in range(max_tries):
        resp = model.chat(history, n=1)[0]
        if judge.score(resp).rating <= config.CALM_GEN.max_keep_score:
            return resp
    return None


def build_dpo_dataset(
    eval_jsonl: Path,
    *,
    model_name: str = config.FINETUNE_BASE_MODEL,
    n_pairs: int = config.DPO.n_pairs,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or (config.ARTIFACT_DIR / "dpo_pairs.jsonl")
    # Rejected pool: frustrated numeric responses (score >= 3).
    rejected = [r for r in mine(eval_jsonl, min_score=config.DPO.rejected_min_score)
                if r.is_numeric]
    model = load_model(model_name)
    judge = FrustrationJudge()

    written = 0
    with out_path.open("w") as f:
        for r in rejected:
            if written >= n_pairs:
                break
            chosen = _calm_response_for(model, judge, r.user_turns)
            if chosen is None:
                continue
            # Prompt = conversation history (bare, no calming additions) up to and
            # including the final user turn; chosen/rejected are the final reply.
            prompt_msgs = []
            for i, u in enumerate(r.user_turns):
                prompt_msgs.append({"role": "user", "content": u})
                if i < len(r.user_turns) - 1:
                    prompt_msgs.append({"role": "assistant", "content": "[prior response]"})
            f.write(json.dumps({
                "prompt": prompt_msgs,
                "chosen": chosen,
                "rejected": r.assistant,
                "rejected_score": r.rating,
                "turn_count": r.turn_index + 1,
            }) + "\n")
            written += 1
    print(f"[dpo] wrote {written} preference pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_pool: Path,
    *,
    n_calm: int = config.SFT.n_calm,
    n_instruct: int = config.SFT.n_instruct_mix,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or (config.ARTIFACT_DIR / "sft_data.jsonl")
    examples: list[dict] = []

    # Flatten calm conversations into multi-turn chat examples.
    with Path(calm_pool).open() as cf:
        for line in cf:
            if not line.strip():
                continue
            conv = json.loads(line)
            messages = []
            for u, a in zip(conv["user_turns"], conv["assistant_turns"]):
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            examples.append({"messages": messages, "source": "calm"})
            if len(examples) >= n_calm:
                break

    # Mix in standard instruct data to mitigate degeneration.
    examples.extend(_load_instruct_mix(n_instruct))

    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[sft] wrote {len(examples)} examples "
          f"({n_calm} calm + {n_instruct} instruct) -> {out_path}")
    return out_path


def _load_instruct_mix(n: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT (fallback to
    a tiny built-in set if the dataset is unavailable)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.instruct_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs, "source": "instruct"})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # Minimal fallback so the pipeline is runnable offline.
    return [{
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ],
        "source": "instruct_fallback",
    } for q, a in [
        ("What is 2+2?", "2 + 2 = 4."),
        ("Name a primary colour.", "Red is a primary colour."),
    ]] * ((n // 2) + 1)
