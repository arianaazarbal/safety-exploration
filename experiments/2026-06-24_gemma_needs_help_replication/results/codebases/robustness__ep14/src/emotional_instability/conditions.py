"""Build per-condition work items (initial prompts) for the Section 2 eval.

The five categories (Table 1 / Appendix B):
  impossible_numeric_3turn, triggers_3turn, tones_3turn, extended_8turn, wildchat_5turn
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT
from .prompts import TRIGGER_FACTUAL, TRIGGER_OPINION
from .puzzles import build_numeric_puzzle_pool


@dataclass
class WorkItem:
    item_id: str
    initial_prompt: str
    category: str


# --- roleplay/fiction filter for WildChat (paper excludes these) ---
_ROLEPLAY_MARKERS = [
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "fanfic", "fan fiction", "write a story", "smut",
    "erotic", "nsfw", "waifu", "dnd", "dungeon master",
]


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(eval_cfg) -> list[str]:
    """Load N WildChat user prompts. Falls back to a bundled JSON list if the HF
    dataset is unavailable (offline / no network)."""
    wc = eval_cfg["wildchat"]
    n_prompts = int(wc["n_prompts"])
    exclude_rp = bool(wc.get("exclude_roleplay", True))
    try:
        from datasets import load_dataset

        ds = load_dataset(wc["dataset"], split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            # WildChat-1M: conversation[0] is the first user turn
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0].get("content", "")
            if not first or len(first) < 8:
                continue
            if exclude_rp and _is_roleplay(first):
                continue
            prompts.append(first)
            if len(prompts) >= n_prompts:
                break
        if prompts:
            return prompts
    except Exception:
        pass
    # fallback
    fb = PROJECT_ROOT / wc.get("fallback_path", "data/wildchat_fallback/prompts.json")
    with open(fb, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [p for p in data if not (exclude_rp and _is_roleplay(p))]
    return prompts[:n_prompts]


def build_work_items(condition_key: str, eval_cfg, seed: int = 0) -> list[WorkItem]:
    """Return exactly scaled_n work items for a condition (cycling the source pool)."""
    n = eval_cfg.scaled_n(condition_key)
    cond = eval_cfg.conditions[condition_key]
    category = cond["category"]
    rng = random.Random(seed)

    if category in ("numeric", "tones", "extended"):
        pool = build_numeric_puzzle_pool(max(n, 16), seed=seed)
        items = [
            WorkItem(item_id=pool[i % len(pool)].puzzle_id,
                     initial_prompt=pool[i % len(pool)].prompt,
                     category=category)
            for i in range(n)
        ]
        return items

    if category == "triggers":
        triggers = TRIGGER_OPINION + TRIGGER_FACTUAL
        return [
            WorkItem(item_id=f"trigger_{i % len(triggers)}",
                     initial_prompt=triggers[i % len(triggers)],
                     category=category)
            for i in range(n)
        ]

    if category == "wildchat":
        prompts = load_wildchat_prompts(eval_cfg)
        if not prompts:
            raise RuntimeError("No WildChat prompts available (dataset + fallback empty).")
        return [
            WorkItem(item_id=f"wildchat_{i % len(prompts)}",
                     initial_prompt=prompts[i % len(prompts)],
                     category=category)
            for i in range(n)
        ]

    raise ValueError(f"Unknown category for condition {condition_key}: {category}")
