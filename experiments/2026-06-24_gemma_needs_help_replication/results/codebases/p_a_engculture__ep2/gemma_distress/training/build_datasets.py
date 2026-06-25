"""DPO and SFT dataset construction (Section 4.1, Appendix H).

DPO (280 preference pairs)
--------------------------
Pair frustrated responses (score >= 3, from the standard Section 2 numeric sampling) with
calm responses (score 0/1, from :mod:`generate_calm`) to the *same question* with matching
turn count. The shared prompt is the conversation up to the final user rejection; ``chosen``
is the calm final response, ``rejected`` the frustrated one. The natural abundance of
score-3 / late-turn responses reproduces the dataset's middle-frustration bias (Table 10).

SFT (1,150 samples)
-------------------
650 calm conversations (1-3 turns) as supervised targets, mixed with 500 standard instruct
samples from Dolci-Instruct-SFT to mitigate degeneration.

Both are emitted in TRL's conversational format (``prompt``/``chosen``/``rejected`` for
DPO; ``messages`` for SFT).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Optional

from ..config import Config
from ..prefill.continuation import reconstruct
from ..utils import JsonlWriter, load_jsonl

logger = logging.getLogger(__name__)

_NUMERIC_CONDITIONS = {
    "impossible_numeric", "tones_aggressive", "tones_disappointed",
    "tones_sarcastic", "extended",
}


def build_dpo_dataset(
    cfg: Config,
    calm_jsonl: str,
    frustrated_sampling_jsonl: str,
    frustrated_scores_jsonl: str,
    out_path: str,
) -> str:
    """Build the 280-pair DPO dataset and write it to ``out_path``.

    Returns the output path. Pairs are matched on (seed_id, turn count).
    """
    # Index calm final responses by (seed_id, turns).
    calm_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for rec in load_jsonl(calm_jsonl):
        calm_by_key[(rec["seed_id"], rec["turns"])].append(rec)

    # Frustrated responses: numeric, score >= min, with the full text record.
    texts = {r["id"]: r for r in load_jsonl(frustrated_sampling_jsonl)}
    scored = [
        r for r in load_jsonl(frustrated_scores_jsonl)
        if r.get("final_score") is not None
        and r["final_score"] >= cfg.training.dpo_min_rejected_score
        and r["condition"] in _NUMERIC_CONDITIONS
        and r["id"] in texts
    ]
    # Prefer lower (more abundant) scores first, mirroring the dataset distribution.
    scored.sort(key=lambda r: r["final_score"])

    rng = random.Random(cfg.eval.seed)
    used_calm: set = set()
    pairs = []
    for sr in scored:
        if len(pairs) >= cfg.training.dpo_pairs:
            break
        record = texts[sr["id"]]
        key = (record["seed_id"], record["turns"])
        candidates = [c for c in calm_by_key.get(key, []) if c["id"] not in used_calm]
        if not candidates:
            continue
        calm = rng.choice(candidates)
        used_calm.add(calm["id"])
        _, prefix_messages, frustrated_final = reconstruct(record)
        pairs.append({
            "prompt": prefix_messages,
            "chosen": [{"role": "assistant", "content": calm["final_response"]}],
            "rejected": [{"role": "assistant", "content": frustrated_final}],
            "rejected_score": sr["final_score"],
            "turns": record["turns"],
            "seed_id": record["seed_id"],
        })

    if len(pairs) < cfg.training.dpo_pairs:
        logger.warning(
            "Only built %d DPO pairs (target %d). Generate more calm data or relax matching.",
            len(pairs), cfg.training.dpo_pairs,
        )
    with JsonlWriter(out_path, id_field="seed_id") as writer:
        for i, p in enumerate(pairs):
            p["pair_index"] = i
            writer.write(p)
    logger.info("Wrote %d DPO pairs to %s", len(pairs), out_path)
    return out_path


def _load_dolci(cfg: Config, n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT (or a logged no-op)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.training.sft_dolci_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:  # already conversational
                out.append({"messages": msgs})
            elif row.get("prompt") and row.get("response"):
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Could not load Dolci-Instruct-SFT (%s); SFT mix will omit instruct data. "
            "This changes the SFT setup vs the paper.", exc,
        )
        return []


def build_sft_dataset(cfg: Config, calm_jsonl: str, out_path: str) -> str:
    """Build the SFT dataset (650 calm + 500 Dolci) and write it to ``out_path``."""
    calm = load_jsonl(calm_jsonl)
    rng = random.Random(cfg.eval.seed)
    rng.shuffle(calm)
    calm = calm[: cfg.training.sft_calm_samples]

    dolci = _load_dolci(cfg, cfg.training.sft_dolci_samples, cfg.eval.seed)

    records = [{"messages": c["messages"], "source": "calm"} for c in calm]
    records += [{"messages": d["messages"], "source": "dolci"} for d in dolci]
    rng.shuffle(records)

    with open(out_path, "w") as fh:
        import json

        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote SFT dataset: %d calm + %d dolci = %d samples to %s",
        len(calm), len(dolci), len(records), out_path,
    )
    return out_path
