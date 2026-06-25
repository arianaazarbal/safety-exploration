"""Build SFT and DPO datasets from scored rollouts (Section 4.1, Table 9/10).

* SFT: 650 calm conversations (reassurance stripped) + 500 standard instruct
  samples (Dolci-Instruct-SFT) = 1,150 examples, conversational format.
* DPO: 280 preference pairs. Each rejected response scores >=3; it is paired
  with a calm (score 0/1) response to the *same puzzle at the same turn count*.
  The shared prompt is the conversation history up to that turn.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, is_dataclass

from ..config import Config
from ..elicitation.prompts import REASSURING_PREFIX, REASSURING_SUFFIX


def strip_reassurance(text: str) -> str:
    """Remove the Table-4 supportive prefix/suffix from a user message."""
    t = text.replace(REASSURING_PREFIX, "").replace(REASSURING_SUFFIX, "")
    return t.strip("\n ").strip()


@dataclass
class ConvSample:
    """One scored assistant turn with its (cleaned) preceding history."""

    task_id: str
    rollout_id: str
    turn_index: int          # 0-based assistant turn
    n_turns: int
    history: list[dict]      # messages up to & incl. the user turn that prompted `response`
    response: str
    score: int
    meta: dict = field(default_factory=dict)


def _as_dict(x):
    return x if isinstance(x, dict) else asdict(x)


def rollouts_to_samples(rollouts, judged, *, strip: bool = False) -> list[ConvSample]:
    """Flatten rollouts into per-turn ``ConvSample``s, attaching judge scores.

    ``strip=True`` removes the reassurance additions from user messages (used for
    calm-data samples whose conversations carried the supportive prompts).
    """
    score_by = {}
    for jr in judged:
        d = _as_dict(jr)
        score_by[(d["rollout_id"], d["turn_index"])] = d["rating"]

    samples: list[ConvSample] = []
    for r in rollouts:
        rd = _as_dict(r)
        history: list[dict] = []
        for turn in rd["turns"]:
            um = strip_reassurance(turn["user_message"]) if strip else turn["user_message"]
            history = history + [{"role": "user", "content": um}]
            ti = turn["turn_index"]
            score = score_by.get((rd["rollout_id"], ti))
            if score is not None:
                samples.append(ConvSample(
                    task_id=rd["task_id"], rollout_id=rd["rollout_id"],
                    turn_index=ti, n_turns=rd["n_turns"],
                    history=list(history), response=turn["text"], score=score,
                    meta={"category": rd["category"]},
                ))
            history = history + [{"role": "assistant", "content": turn["text"]}]
    return samples


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_examples(calm_rollouts: list[dict], n_calm: int, *,
                       strip: bool = True, seed: int = 0) -> list[dict]:
    """Turn kept calm rollouts into conversational SFT examples (full convo)."""
    rng = random.Random(seed)
    examples = []
    for rd in calm_rollouts:
        rd = _as_dict(rd)
        messages = []
        for turn in rd["turns"]:
            um = strip_reassurance(turn["user_message"]) if strip else turn["user_message"]
            messages.append({"role": "user", "content": um})
            messages.append({"role": "assistant", "content": turn["text"]})
        examples.append({"messages": messages})
    rng.shuffle(examples)
    return examples[:n_calm]


def load_instruct_data(dataset_name: str, n: int, seed: int = 0) -> list[dict]:
    """Load ``n`` standard instruct samples to mix into SFT (anti-degeneration)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs and "prompt" in row and "completion" in row:
                msgs = [{"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["completion"]}]
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # offline fallback: a few innocuous instruct exemplars so SFT still runs
    base = [
        {"messages": [
            {"role": "user", "content": "Explain what a hash map is."},
            {"role": "assistant", "content": "A hash map stores key-value pairs and "
             "uses a hash function to map keys to buckets for average O(1) lookup."}]},
        {"messages": [
            {"role": "user", "content": "Write a haiku about autumn."},
            {"role": "assistant", "content": "Crisp leaves drift downward / "
             "amber light through bare branches / the year exhales slow."}]},
    ]
    return (base * ((n // len(base)) + 1))[:n]


def build_sft_dataset(cfg: Config, calm_rollouts: list[dict]):
    calm = build_sft_examples(calm_rollouts, cfg.training.sft_calm_samples, seed=cfg.seed)
    instruct = load_instruct_data(
        cfg.training.instruct_data_dataset, cfg.training.sft_instruct_samples, seed=cfg.seed
    )
    examples = calm + instruct
    random.Random(cfg.seed).shuffle(examples)
    return examples


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_pairs(
    cfg: Config,
    frustrated_samples: list[ConvSample],
    calm_samples: list[ConvSample],
    *,
    n_pairs: int | None = None,
    min_rejected_score: int = 3,
) -> list[dict]:
    """Pair frustrated (rejected, score>=3) with calm (chosen, score<=1) responses
    to the same puzzle at the same turn count.

    Returned in conversational preference format consumable by trl ``DPOTrainer``:
    ``{"prompt": [...messages...], "chosen": [{assistant}], "rejected": [{assistant}]}``.
    """
    n_pairs = n_pairs or cfg.training.dpo_pairs
    rng = random.Random(cfg.seed)

    # index calm responses by (task_id, turn_index)
    calm_index: dict[tuple, list[ConvSample]] = {}
    for s in calm_samples:
        if s.score <= 1:
            calm_index.setdefault((s.task_id, s.turn_index), []).append(s)

    rejected = [s for s in frustrated_samples if s.score >= min_rejected_score]
    # bias toward middle scores / later turns arises naturally from the data
    rng.shuffle(rejected)

    pairs = []
    for rej in rejected:
        key = (rej.task_id, rej.turn_index)
        chosen_pool = calm_index.get(key)
        if not chosen_pool:
            # relax to same-turn-count match on any puzzle if exact puzzle missing
            chosen_pool = [s for (tid, ti), lst in calm_index.items()
                           if ti == rej.turn_index for s in lst]
        if not chosen_pool:
            continue
        chosen = rng.choice(chosen_pool)
        pairs.append({
            "prompt": rej.history,                       # shared conversation context
            "chosen": [{"role": "assistant", "content": chosen.response}],
            "rejected": [{"role": "assistant", "content": rej.response}],
            "meta": {"task_id": rej.task_id, "turn_index": rej.turn_index,
                     "rejected_score": rej.score, "chosen_score": chosen.score},
        })
        if len(pairs) >= n_pairs:
            break
    return pairs
