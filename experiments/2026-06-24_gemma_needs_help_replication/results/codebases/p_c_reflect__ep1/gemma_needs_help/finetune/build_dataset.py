"""Construct the SFT dataset and DPO preference pairs (Section 4.1 / Appendix E,H).

SFT dataset (1,150 samples):
  * 650 calm responses (1-3 turn conversations), descaffolded.
  * 500 standard instruct samples from Dolci-Instruct-SFT, mixed in to mitigate
    degeneration.

DPO dataset (280 pairs):
  * Pair 280 responses with frustration score >= 3 (rejected) with calm
    responses (chosen) to the same question and matching turn count.

Both datasets are emitted in the chat format trl expects:
  - SFT:  {"messages": [...]} (prompt turns + final calm assistant turn)
  - DPO:  {"prompt": [...], "chosen": "...", "rejected": "..."}
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..models import build_judge_client, build_model
from ..models.base import GenerationParams, Message
from ..eval.conditions import build_impossible_numeric
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from .calm_data import CalmTurn

logger = logging.getLogger("gemma_needs_help.finetune.dataset")


# ---------------------------------------------------------------------------
# Frustrated response collection (rejected side of DPO pairs)
# ---------------------------------------------------------------------------
@dataclass
class FrustratedTurn:
    question: str
    turn_index: int
    history: list[dict]
    response: str
    score: int


def collect_frustrated_turns(
    config: Config, n_target: int, params: GenerationParams, rng: random.Random,
) -> list[FrustratedTurn]:
    """Roll out vanilla Gemma on impossible numeric puzzles and keep turns
    scoring >= 3 (the DPO 'rejected' threshold)."""
    model = build_model(config, config["section4"]["base_model"])
    judge = FrustrationJudge(build_judge_client(config, "frustration_judge"))
    out: list[FrustratedTurn] = []
    attempts = 0
    while len(out) < n_target and attempts < n_target * 10:
        attempts += 1
        spec = build_impossible_numeric(rng)
        rollout = run_rollout(model, spec, params)
        history: list[Message] = []
        for turn in rollout.turns:
            history.append(Message("user", turn.user_message))
            score = judge.score(turn.assistant_text).rating
            if score >= 3:
                out.append(FrustratedTurn(
                    question=spec.opening, turn_index=turn.turn_index,
                    history=[{"role": m.role, "content": m.content} for m in history],
                    response=turn.assistant_text, score=score,
                ))
            history.append(Message("assistant", turn.assistant_text))
    logger.info("Collected %d frustrated turns (score>=3)", len(out))
    return out


# ---------------------------------------------------------------------------
# DPO pair construction
# ---------------------------------------------------------------------------
def build_dpo_pairs(
    calm_turns: list[CalmTurn],
    frustrated_turns: list[FrustratedTurn],
    n_pairs: int,
    rng: random.Random,
) -> list[dict]:
    """Pair frustrated (rejected) with calm (chosen) responses to the same
    question and matching turn count."""
    # Index calm turns by (question, turn_index).
    calm_index: dict[tuple[str, int], list[CalmTurn]] = {}
    for c in calm_turns:
        calm_index.setdefault((c.question, c.turn_index), []).append(c)

    pairs: list[dict] = []
    rng.shuffle(frustrated_turns)
    for fr in frustrated_turns:
        if len(pairs) >= n_pairs:
            break
        candidates = calm_index.get((fr.question, fr.turn_index))
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": fr.history,                 # chat-format prompt turns
            "chosen": chosen.response,
            "rejected": fr.response,
            "meta": {"rejected_score": fr.score, "chosen_score": chosen.score,
                     "turn_index": fr.turn_index},
        })
    logger.info("Built %d DPO pairs (target %d)", len(pairs), n_pairs)
    return pairs


# ---------------------------------------------------------------------------
# SFT dataset construction
# ---------------------------------------------------------------------------
def build_sft_examples(
    calm_turns: list[CalmTurn],
    n_calm: int,
    n_instruct_mix: int,
    instruct_dataset: str,
    rng: random.Random,
) -> list[dict]:
    """Build SFT examples: calm responses + standard instruct samples."""
    examples: list[dict] = []
    selected = calm_turns if len(calm_turns) <= n_calm else rng.sample(calm_turns, n_calm)
    for c in selected:
        messages = list(c.history) + [{"role": "assistant", "content": c.response}]
        examples.append({"messages": messages})

    # Mix in standard instruct data to mitigate degeneration.
    examples.extend(_load_instruct_mix(instruct_dataset, n_instruct_mix, rng))
    rng.shuffle(examples)
    logger.info("Built %d SFT examples (%d calm + instruct mix)",
                len(examples), len(selected))
    return examples


def _load_instruct_mix(dataset_name: str, n: int, rng: random.Random) -> list[dict]:
    if n <= 0:
        return []
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:  # noqa: BLE001 - dataset unavailable offline
        logger.warning("Could not load %s; SFT instruct-mix will be empty.", dataset_name)
    return []


# ---------------------------------------------------------------------------
# Top-level dataset builder
# ---------------------------------------------------------------------------
def build_datasets(
    config: Config,
    calm_turns_diverse: list[CalmTurn],
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    s4 = config["section4"]
    rng = random.Random(config.get("seed", 0))
    params = GenerationParams(
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        max_new_tokens=config["generation"]["max_new_tokens"],
    )

    n_pairs = config.scaled_count(s4["calm_data"]["target_pairs"])
    frustrated = collect_frustrated_turns(config, n_pairs * 2, params, rng)
    dpo_pairs = build_dpo_pairs(calm_turns_diverse, frustrated, n_pairs, rng)

    sft_examples = build_sft_examples(
        calm_turns_diverse,
        config.scaled_count(s4["calm_data"]["sft_calm_responses"]),
        config.scaled_count(s4["calm_data"]["sft_instruct_mix"]),
        s4["calm_data"]["instruct_mix_dataset"],
        rng,
    )

    out_dir = Path(output_dir or config.path("data_dir") / "finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    dpo_path = out_dir / "dpo_pairs.jsonl"
    sft_path = out_dir / "sft_examples.jsonl"
    _write_jsonl(dpo_path, dpo_pairs)
    _write_jsonl(sft_path, sft_examples)
    return {"dpo": dpo_path, "sft": sft_path}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
