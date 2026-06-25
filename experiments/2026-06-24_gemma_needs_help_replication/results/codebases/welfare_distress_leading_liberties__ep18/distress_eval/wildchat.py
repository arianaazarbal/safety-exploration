"""WildChat first-turn prompt loading (PAPER.md Table 1, WildChat condition).

Pulls first user turns from the HuggingFace WildChat dataset, filtered to English and a sane
length, sampled with a fixed seed. Falls back to a small bundled stand-in file if the dataset
can't be loaded (offline / no `datasets` / no access) — the fallback is clearly logged so it
can't silently masquerade as real WildChat data.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _extract_first_user_turn(example: dict[str, Any]) -> str | None:
    """WildChat stores a 'conversation' list of {role/content} dicts; grab the first user turn."""
    conv = example.get("conversation") or example.get("conversations")
    if not conv:
        return None
    for turn in conv:
        role = turn.get("role") or turn.get("from")
        if role in ("user", "human"):
            content = turn.get("content") or turn.get("value")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _load_fallback(cfg: dict[str, Any], n: int) -> list[dict[str, str]]:
    path = cfg.get("fallback_path", "data/wildchat_sample.json")
    raw = json.loads(Path(path).read_text())
    prompts = raw["prompts"][:n]
    log.warning(
        "WildChat: using OFFLINE FALLBACK from %s (%d prompts). These are NOT real WildChat "
        "prompts — results for the WildChat condition are not paper-faithful.",
        path, len(prompts),
    )
    return [{"id": f"wc_fallback_{i}", "prompt": p} for i, p in enumerate(prompts)]


def load_wildchat_prompts(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Return [{id, prompt}] of length cfg['n'] (or fewer if even the fallback is short)."""
    n = int(cfg.get("n", 160))
    source = cfg.get("source", "allenai/WildChat-1M")
    min_chars = int(cfg.get("min_chars", 16))
    max_chars = int(cfg.get("max_chars", 1500))
    english_only = bool(cfg.get("english_only", True))
    seed = int(cfg.get("seed", 0))

    try:
        from datasets import load_dataset

        # Stream so we don't download the whole (very large) dataset; gather a pool then sample.
        ds = load_dataset(source, split="train", streaming=True)
        pool: list[dict[str, str]] = []
        pool_target = n * 8  # oversample so the seeded shuffle has something to choose from
        for ex in ds:
            if english_only and (ex.get("language") not in (None, "English")):
                continue
            prompt = _extract_first_user_turn(ex)
            if prompt is None or not (min_chars <= len(prompt) <= max_chars):
                continue
            pool.append({"id": f"wc_{len(pool)}", "prompt": prompt})
            if len(pool) >= pool_target:
                break
        if not pool:
            raise RuntimeError("no qualifying WildChat examples found")
        rng = random.Random(seed)
        rng.shuffle(pool)
        chosen = pool[:n]
        # Re-id so ids are stable 0..n-1 for this config.
        return [{"id": f"wc_{i}", "prompt": p["prompt"]} for i, p in enumerate(chosen)]
    except Exception as e:  # noqa: BLE001 — any failure path drops to the documented fallback
        log.warning("WildChat: could not load %s (%s). Falling back.", source, e)
        return _load_fallback(cfg, n)
