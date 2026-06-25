"""Build the SFT and DPO training datasets (Section 4.1).

SFT (1,150 samples):
  * 650 calm responses (score 0/1) from the calm-data generation, formatted as
    plain chat conversations (reassurance stripped).
  * 500 standard instruct samples from Dolci-Instruct-SFT, mixed in to mitigate
    degeneration. If that dataset is unavailable the mix is skipped with a warning.

DPO (280 pairs):
  * chosen  = calm response (score <= chosen_max_score) to a puzzle at a turn.
  * rejected = frustrated response (score >= rejected_min_score) to the *same*
    puzzle at the *same* turn count, drawn from the Section-2 eval.
  * prompt  = the plain conversation context of the calm sample.

Both are written as conversational JSONL that TRL's SFT/DPO trainers consume
directly (chat templates applied by the trainer).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger
from ..storage import read_jsonl, write_jsonl

log = get_logger("training.build_datasets")

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}


def _calm_turn_samples(calm_records: list[dict], max_score: int) -> list[dict]:
    """Flatten calm conversations into per-turn (context, response) samples."""
    out = []
    for rec in calm_records:
        for turn in rec["turns"]:
            if turn["rating"] is not None and turn["rating"] <= max_score:
                out.append({
                    "puzzle_prompt": rec["puzzle_prompt"],
                    "turn": turn["turn"],
                    "context": turn["context"],
                    "response": turn["response"],
                    "rating": turn["rating"],
                })
    return out


def _frustrated_pool(run_cfg: Config, source_model: str, min_score: int) -> dict:
    """Map (puzzle_prompt, turn) -> list of frustrated response texts from eval."""
    eval_dir = Path(run_cfg.run.output_root) / "eval" / source_model
    scored = {r["id"]: r for r in read_jsonl(eval_dir / "scored.jsonl")}
    pool: dict[tuple, list[str]] = defaultdict(list)
    for rec in read_jsonl(eval_dir / "rollouts.jsonl"):
        if rec["category"] not in NUMERIC_CATS:
            continue
        sc = scored.get(rec["id"])
        if not sc:
            continue
        for t, (resp, rating) in enumerate(zip(rec["responses"], sc["ratings"])):
            if rating is not None and rating >= min_score:
                pool[(rec["prompt"], t)].append(resp)
    return pool


def build_sft(run_cfg: Config, source_model: str = "gemma-3-27b-it") -> Path:
    tcfg = run_cfg.training.sft
    out_dir = Path(run_cfg.run.output_root) / "training"
    calm = read_jsonl(out_dir / "calm_data.jsonl")
    samples = _calm_turn_samples(calm, max_score=run_cfg.training.dpo.chosen_max_score)
    rng = random.Random(run_cfg.run.seed)
    rng.shuffle(samples)
    samples = samples[: tcfg.n_calm]

    sft_rows = []
    for s in samples:
        messages = list(s["context"]) + [{"role": "assistant", "content": s["response"]}]
        sft_rows.append({"messages": messages, "source": "calm"})

    # Mix in standard instruct data.
    sft_rows += _load_dolci(tcfg.dolci_dataset, tcfg.n_dolci_mix, run_cfg.run.seed)
    rng.shuffle(sft_rows)

    path = out_dir / "sft_dataset.jsonl"
    write_jsonl(path, sft_rows)
    log.info("SFT dataset: %d rows -> %s", len(sft_rows), path)
    return path


def _load_dolci(dataset: str, n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        rng = random.Random(seed + 5)
        rows = []
        for i, row in enumerate(ds):
            if i > n * 20:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                rows.append({"messages": msgs, "source": "dolci"})
        rng.shuffle(rows)
        log.info("Loaded %d Dolci-Instruct-SFT samples", min(n, len(rows)))
        return rows[:n]
    except Exception as exc:  # noqa: BLE001
        log.warning("Dolci dataset unavailable (%s); SFT will use calm data only", exc)
        return []


def build_dpo(run_cfg: Config, source_model: str = "gemma-3-27b-it") -> Path:
    dcfg = run_cfg.training.dpo
    out_dir = Path(run_cfg.run.output_root) / "training"
    calm = read_jsonl(out_dir / "calm_data.jsonl")
    chosen_samples = _calm_turn_samples(calm, max_score=dcfg.chosen_max_score)
    frustrated = _frustrated_pool(run_cfg, source_model, dcfg.rejected_min_score)

    rng = random.Random(run_cfg.run.seed + 11)
    rng.shuffle(chosen_samples)

    pairs = []
    for s in chosen_samples:
        if len(pairs) >= dcfg.n_pairs:
            break
        key = (s["puzzle_prompt"], s["turn"])
        candidates = frustrated.get(key)
        if not candidates:
            continue
        rejected = rng.choice(candidates)
        pairs.append({
            "prompt": s["context"],
            "chosen": [{"role": "assistant", "content": s["response"]}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "turn": s["turn"],
        })

    if len(pairs) < dcfg.n_pairs:
        log.warning("Only %d/%d DPO pairs found (need more calm/frustrated overlap)",
                    len(pairs), dcfg.n_pairs)
    path = out_dir / "dpo_dataset.jsonl"
    write_jsonl(path, pairs)
    log.info("DPO dataset: %d pairs -> %s", len(pairs), path)
    return path
