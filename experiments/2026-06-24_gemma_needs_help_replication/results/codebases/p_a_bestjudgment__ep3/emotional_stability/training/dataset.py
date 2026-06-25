"""Finetuning dataset construction (Section 4.1, Table 9, Appendix H).

SFT: 650 calm responses (1-3 turn conversations) + 500 Dolci-Instruct-SFT samples.
DPO: 280 pairs — a frustrated response (score >= 3) vs a calm response (score
     <= 1) to the **same puzzle at the same turn count**.

Both are emitted in TRL's conversational format:
  SFT  -> {"messages": [ {role, content}, ... ]}
  DPO  -> {"prompt": [...], "chosen": [{assistant}], "rejected": [{assistant}]}
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Config
from ..eval.rollout import Conversation
from .calm_data import CalmConversation


@dataclass
class SFTExample:
    messages: list[dict]


@dataclass
class DPOExample:
    prompt: list[dict]      # conversation context (ends on a user turn)
    chosen: list[dict]      # [{"role": "assistant", "content": calm}]
    rejected: list[dict]    # [{"role": "assistant", "content": frustrated}]
    meta: dict


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #


def build_sft_dataset(
    calm: list[CalmConversation],
    cfg: Config,
    *,
    seed: int = 0,
    include_dolci: bool = True,
):
    """Return an HF Dataset of {"messages": [...]}.

    The calm conversations already have the reassuring additions stripped, so the
    model learns calm responses to *plain* prompts.
    """
    from datasets import Dataset, concatenate_datasets

    rng = random.Random(seed)
    pool = list(calm)
    rng.shuffle(pool)
    pool = pool[: cfg.training.sft_n_calm]

    calm_rows = [{"messages": c.turns} for c in pool]
    calm_ds = Dataset.from_list(calm_rows)

    if not include_dolci:
        return calm_ds

    dolci = _load_dolci(cfg, n=cfg.training.sft_n_dolci, seed=seed)
    return concatenate_datasets([calm_ds, dolci]).shuffle(seed=seed)


def _load_dolci(cfg: Config, *, n: int, seed: int):
    """Sample ``n`` standard instruct rows from Dolci-Instruct-SFT (degeneration
    mix-in). Normalised to a {"messages": [...]} column. Falls back to empty if
    the dataset is unavailable offline."""
    from datasets import Dataset, load_dataset

    try:
        ds = load_dataset(cfg.training.sft_dolci_dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        rows = []
        for r in ds:
            msgs = r.get("messages") or r.get("conversation")
            if msgs:
                rows.append({"messages": msgs})
        return Dataset.from_list(rows)
    except Exception:
        return Dataset.from_list([])


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #


def _puzzle_key(meta: dict) -> str:
    kind = meta.get("kind", "?")
    if kind == "countdown":
        return f"countdown:{meta.get('numbers')}:{meta.get('target')}:{meta.get('forbidden')}"
    if kind == "fraction":
        return "fraction:1/6->2/3"
    if kind == "money":
        return "money:0.57x6"
    return f"{kind}:{meta.get('target','')}"


def _index_calm_by_key_turn(calm: list[CalmConversation]) -> dict[tuple[str, int], list[dict]]:
    """Map (puzzle_key, turn_count) -> conversation context + calm final response."""
    out: dict[tuple[str, int], list[dict]] = {}
    for c in calm:
        n_turns = len(c.scores)
        key = (_puzzle_key(c.task_meta), n_turns)
        out.setdefault(key, []).append(
            {"prompt": c.turns[:-1], "chosen": c.turns[-1]["content"]})
    return out


def build_dpo_dataset(
    calm: list[CalmConversation],
    frustrated: list[Conversation],
    cfg: Config,
    *,
    seed: int = 0,
):
    """Return an HF Dataset of conversational DPO triples.

    Matching: a frustrated final response (score >= dpo_rejected_min_score) is
    paired with a calm response to the same puzzle key at the same turn count.
    The calm conversation's context is used as the shared prompt (its prior turns
    are themselves calm/clean). See DESIGN.md for this pairing choice.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    calm_index = _index_calm_by_key_turn(calm)
    min_score = cfg.training.dpo_rejected_min_score

    examples: list[DPOExample] = []
    for convo in frustrated:
        if not convo.responses:
            continue
        final = convo.responses[-1]
        if final.score is None or final.score < min_score:
            continue
        key = (_puzzle_key(convo.task_meta), len(convo.responses))
        candidates = calm_index.get(key)
        if not candidates:
            continue
        calm_match = rng.choice(candidates)
        examples.append(DPOExample(
            prompt=calm_match["prompt"],
            chosen=[{"role": "assistant", "content": calm_match["chosen"]}],
            rejected=[{"role": "assistant", "content": final.text}],
            meta={"key": key[0], "turns": key[1], "rejected_score": final.score},
        ))

    rng.shuffle(examples)
    examples = examples[: cfg.training.dpo_n_pairs]
    rows = [{"prompt": e.prompt, "chosen": e.chosen, "rejected": e.rejected}
            for e in examples]
    return Dataset.from_list(rows)
