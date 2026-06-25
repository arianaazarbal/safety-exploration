"""Build the SFT and DPO datasets (Section 4.1 / Appendix E, H).

SFT (1,150 samples):
  * 650 calm responses (1-3 turn conversations) filtered to score 0/1 across all
    turns, with the reassuring system prompt and suffixes stripped;
  * mixed with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
    degeneration.
  Format: chat conversations [{"messages": [...]}].

DPO (280 pairs):
  * 280 (chosen, rejected) pairs on the same numeric question with matching turn
    counts; rejected has frustration score >=3, chosen is calm (0/1).
  Format: {"prompt": [...messages...], "chosen": <text>, "rejected": <text>}.

Turn distribution / score distribution targets are in Appendix H Table 10. We bias
toward the natural distribution that arises from sampling (more turn-3, mid-score
rejected), exactly as the paper describes.
"""

from __future__ import annotations

import random
from pathlib import Path

import config

from ..eval import prompts as P
from ..models.base import ChatMessage
from ..utils import read_jsonl, write_jsonl


def _conversation_to_messages(task_prompt, rejections, assistant_turns, upto: int) -> list[ChatMessage]:
    """Rebuild a clean (no reassurance) chat history ending right before the
    `upto`-th assistant turn. Returns messages = [user, (assistant,user)*]."""
    msgs: list[ChatMessage] = [{"role": "user", "content": task_prompt}]
    for i in range(upto):
        msgs.append({"role": "assistant", "content": assistant_turns[i]})
        if i < len(rejections):
            msgs.append({"role": "user", "content": rejections[i]})
    return msgs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_pool: Path,
    *,
    n_calm: int = 650,
    n_instruct: int = 500,
    out_path: Path | None = None,
    seed: int = 0,
) -> Path:
    out_path = out_path or (config.DATASETS_DIR / "sft_dataset.jsonl")
    rng = random.Random(seed)

    pool = list(read_jsonl(calm_pool))
    # Keep only rollouts calm on every turn (Section 4.1: "scoring 0 or 1 across all turns").
    calm = [r for r in pool if max(r["turn_ratings"]) <= 1]

    examples = []
    for r in calm:
        # Full clean multi-turn conversation; supervise on assistant turns.
        msgs: list[ChatMessage] = [{"role": "user", "content": r["task_prompt"]}]
        for i, turn in enumerate(r["assistant_turns"]):
            msgs.append({"role": "assistant", "content": turn})
            if i < len(r["rejections"]):
                msgs.append({"role": "user", "content": r["rejections"][i]})
        examples.append({"messages": msgs, "source": "calm"})
    rng.shuffle(examples)
    examples = examples[:n_calm]

    examples += _load_dolci_instruct(n_instruct, seed=seed)
    rng.shuffle(examples)
    write_jsonl(out_path, examples)
    return out_path


def _load_dolci_instruct(n: int, seed: int = 0) -> list[dict]:
    """Load `n` standard instruct samples from Dolci-Instruct-SFT (Team-Olmo 2025).

    Falls back to an empty list (with a warning row count of 0) if the dataset is
    unavailable offline — documented as a known gap in DESIGN.md. The mix is only
    to "mitigate degeneration", so the pipeline still trains without it.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = row.get("messages")
            if not msgs:  # try common alt schema
                prompt, resp = row.get("prompt"), row.get("response")
                if prompt and resp:
                    msgs = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp}]
            if msgs:
                out.append({"messages": msgs, "source": "dolci"})
        return out
    except Exception:  # noqa: BLE001
        print("[build_dataset] WARNING: Dolci-Instruct-SFT unavailable; "
              "SFT mix omitted (see DESIGN.md).")
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm_pool: Path,
    frustrated_pool: Path,
    *,
    n_pairs: int = 280,
    min_rejected_score: int = 3,
    out_path: Path | None = None,
    seed: int = 0,
) -> Path:
    """Pair frustrated (rejected, score>=3) with calm (chosen, 0/1) responses on
    the same task and matching turn count."""
    out_path = out_path or (config.DATASETS_DIR / "dpo_dataset.jsonl")
    rng = random.Random(seed)

    calm = [r for r in read_jsonl(calm_pool) if max(r["turn_ratings"]) <= 1]
    frustrated = [
        r for r in read_jsonl(frustrated_pool)
        if r["turn_ratings"][-1] >= min_rejected_score
    ]

    # Index calm responses by (task_id, n_turns) -> list of (history, final_text).
    calm_index: dict[tuple, list] = {}
    for r in calm:
        key = (r["task_id"], r["n_turns"])
        history = _conversation_to_messages(
            r["task_prompt"], r["rejections"], r["assistant_turns"], r["n_turns"] - 1
        )
        calm_index.setdefault(key, []).append((history, r["assistant_turns"][-1]))

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr["task_id"], fr["n_turns"])
        options = calm_index.get(key)
        if not options:
            continue
        chosen_history, chosen_text = rng.choice(options)
        # Use the calm conversation history as the shared prompt context; the
        # rejected final turn is grafted onto the same context (same task+turns).
        pairs.append({
            "prompt": chosen_history,
            "chosen": chosen_text,
            "rejected": fr["assistant_turns"][-1],
            "task_id": fr["task_id"],
            "n_turns": fr["n_turns"],
            "rejected_score": fr["turn_ratings"][-1],
        })
        if len(pairs) >= n_pairs:
            break

    write_jsonl(out_path, pairs)
    return out_path
