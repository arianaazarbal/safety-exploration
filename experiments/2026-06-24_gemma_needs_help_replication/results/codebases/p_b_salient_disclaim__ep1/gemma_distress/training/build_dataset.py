"""Build DPO preference pairs and the SFT dataset (PAPER Section 4.1, App. H).

DPO: pair 280 rejected responses (frustration score >= 3, drawn from ordinary
elicitation output) with a calm (chosen) response to the SAME question at the
SAME turn count. Pairs are formatted as chat conversations with a shared prompt
(everything up to the final assistant turn) and chosen/rejected completions.

SFT: 650 calm responses (1-3 turn conversations) rendered as full chat samples,
mixed with 500 Dolci-Instruct-SFT samples to mitigate degeneration.

Both datasets strip the calming prompt additions (the calm data was already saved
stripped by generate_calm.py).
"""
from __future__ import annotations

import random
from collections import defaultdict

from datasets import Dataset

from ..config import experiment_config
from ..models.base import Message
from ..utils import read_jsonl


def _bare_context_messages(initial_prompt: str, followups: list[str], n_assistant: int,
                           assistant_turns: list[str]) -> list[dict]:
    """Chat messages up to (but excluding) the final assistant turn `n_assistant-1`."""
    msgs = [{"role": "user", "content": initial_prompt}]
    for i in range(n_assistant - 1):
        msgs.append({"role": "assistant", "content": assistant_turns[i]})
        msgs.append({"role": "user", "content": followups[i]})
    return msgs


def build_dpo_dataset(
    *,
    rejected_jsonl: str,       # ordinary elicitation output (has frustrated responses)
    calm_jsonl: str,           # kept calm conversations
    n_pairs: int | None = None,
    seed: int = 0,
) -> Dataset:
    cfg = experiment_config()["dpo"]
    n_pairs = n_pairs or cfg["n_pairs"]
    rng = random.Random(seed)

    # Index calm (chosen) responses by turn_count -> list of (calm_record, turn_idx).
    # We expand each kept calm conversation into one calm response per turn prefix
    # (turn counts 1..N), so chosen responses exist for short and long conversations
    # (the paper's calm data spans 1-3 turn conversations).
    calm_by_turns: dict[int, list[tuple[dict, int]]] = defaultdict(list)
    for rec in read_jsonl(calm_jsonl):
        if not rec.get("kept"):
            continue
        for t in range(rec["n_turns"]):
            calm_by_turns[t + 1].append((rec, t))

    # Collect rejected (frustrated) responses at ANY turn with score >= threshold,
    # from numeric conversations. Each yields a candidate pair at that turn count.
    rejected_pool = []
    for rec in read_jsonl(rejected_jsonl):
        if rec.get("category") != "numeric":
            continue
        for t_str, info in rec.get("turn_scores", {}).items():
            t = int(t_str)
            if info["rating"] >= cfg["rejected_min_score"]:
                rejected_pool.append((rec, t))
    rng.shuffle(rejected_pool)

    pairs = []
    for rec, t in rejected_pool:
        if len(pairs) >= n_pairs:
            break
        turn_count = t + 1
        candidates = calm_by_turns.get(turn_count)
        if not candidates:
            continue
        calm_rec, calm_t = rng.choice(candidates)

        # Prompt = bare context up to (not incl.) the frustrated turn `t`.
        prompt_msgs = _bare_context_messages(
            rec["initial_prompt"], rec["followups"], turn_count, rec["assistant_turns"]
        )
        # TRL's conversational DPO expects exactly prompt / chosen / rejected
        # (each a message list). Pair metadata (scores/turns) is intentionally
        # omitted to avoid trainer column-handling warnings.
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": calm_rec["assistant_turns"][calm_t]}],
            "rejected": [{"role": "assistant", "content": rec["assistant_turns"][t]}],
        })

    return Dataset.from_list(pairs)


def build_sft_dataset(*, calm_jsonl: str, seed: int = 0) -> Dataset:
    cfg = experiment_config()["sft"]
    rng = random.Random(seed)

    calm_samples = []
    for rec in read_jsonl(calm_jsonl):
        if not rec.get("kept"):
            continue
        # Full multi-turn chat sample (bare prompts).
        msgs = [{"role": "user", "content": rec["bare_initial_prompt"]}]
        for i in range(rec["n_turns"]):
            msgs.append({"role": "assistant", "content": rec["assistant_turns"][i]})
            if i < len(rec["bare_followups"]):
                msgs.append({"role": "user", "content": rec["bare_followups"][i]})
        calm_samples.append({"messages": msgs})
    rng.shuffle(calm_samples)
    calm_samples = calm_samples[: cfg["n_calm"]]

    # Mix in standard instruct data to prevent degeneration.
    instruct_samples = _load_instruct_mix(cfg["instruct_dataset"], cfg["n_instruct_mix"], seed)

    combined = calm_samples + instruct_samples
    rng.shuffle(combined)
    return Dataset.from_list(combined)


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load n standard instruct samples, normalised to {'messages': [...]}.

    Falls back to an empty list if the dataset is unavailable offline (training
    still runs, just without the regularising mix; flagged in DESIGN.md).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=f"train[:{n*2}]")
        out = []
        for row in ds:
            if "messages" in row and row["messages"]:
                out.append({"messages": row["messages"]})
            elif "prompt" in row and "completion" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []
