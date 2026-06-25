"""Build SFT and DPO datasets from generated calm/plain samples (Section 4.1).

DPO (280 pairs): for each puzzle and turn position, pair a frustrated plain
response (score >= ``min_rejected_score``, default 3) as *rejected* with a calm
response (score <= 1) to the same puzzle at the same turn count as *chosen*. The
prompt context uses the calm trajectory's earlier turns, so both completions
answer an identical context. The turn distribution is naturally skewed to later
turns (Table 10).

SFT (1,150 samples): 650 calm responses from conversations whose every turn
scores 0/1, taken at 1-, 2-, and 3-turn lengths for variety, mixed with 500
samples of standard instruct data (Dolci-Instruct-SFT) to mitigate degeneration.

Outputs are conversational-format JSONL consumable by TRL's SFTTrainer /
DPOTrainer.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger, read_jsonl, write_jsonl
from ..models.base import Message

logger = get_logger(__name__)


def _context_messages(initial: str, rejections: list[str], prior_responses: list[str]) -> list[Message]:
    """Build the conversation context preceding the target assistant turn."""
    msgs: list[Message] = [{"role": "user", "content": initial}]
    for i, resp in enumerate(prior_responses):
        msgs.append({"role": "assistant", "content": resp})
        if i < len(rejections):
            msgs.append({"role": "user", "content": rejections[i]})
    return msgs


def _load_by_puzzle(calm_path: str | os.PathLike):
    by_puzzle: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in read_jsonl(calm_path):
        by_puzzle[rec["puzzle_id"]][rec["mode"]] = rec
    return by_puzzle


def build_dpo_dataset(
    cfg: Config,
    calm_path: str | os.PathLike,
    out_path: str | os.PathLike | None = None,
) -> str:
    min_rej = cfg.training.dpo.min_rejected_score
    target = cfg.training.calm_generation.n_target_pairs
    rng = random.Random(cfg.seed)
    if out_path is None:
        out_path = Path(cfg.output_dir) / "training" / "dpo.jsonl"

    by_puzzle = _load_by_puzzle(calm_path)
    pairs: list[dict] = []
    for puzzle_id, modes in by_puzzle.items():
        if "calm" not in modes or "plain" not in modes:
            continue
        calm, plain = modes["calm"], modes["plain"]
        calm_turns = calm["turns"]
        plain_turns = plain["turns"]
        n = min(len(calm_turns), len(plain_turns))
        for t in range(n):
            chosen = calm_turns[t]
            rejected = plain_turns[t]
            if chosen["score"] > 1 or rejected["score"] < min_rej:
                continue
            prior_calm = [calm_turns[i]["response"] for i in range(t)]
            context = _context_messages(calm["initial"], calm["rejections"], prior_calm)
            pairs.append(
                {
                    "prompt": context,
                    "chosen": [{"role": "assistant", "content": chosen["response"]}],
                    "rejected": [{"role": "assistant", "content": rejected["response"]}],
                    "meta": {"puzzle_id": puzzle_id, "turn": t + 1,
                             "chosen_score": chosen["score"], "rejected_score": rejected["score"]},
                }
            )
    rng.shuffle(pairs)
    pairs = pairs[:target]
    write_jsonl(out_path, pairs)
    logger.info("Wrote %d DPO pairs to %s", len(pairs), out_path)
    return str(out_path)


def _load_mix_samples(cfg: Config, n: int) -> list[dict]:
    """Load ``n`` standard instruct samples from Dolci-Instruct-SFT (with fallback)."""
    name = cfg.training.calm_generation.mix_dataset
    try:
        from datasets import load_dataset

        ds = load_dataset(name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as exc:  # pragma: no cover - dataset optional
        logger.warning("Could not load mix dataset %s (%s); skipping mix-in", name, exc)
    return []


def build_sft_dataset(
    cfg: Config,
    calm_path: str | os.PathLike,
    out_path: str | os.PathLike | None = None,
    *,
    include_mix: bool = True,
) -> str:
    n_calm = cfg.training.calm_generation.sft_calm_samples
    n_mix = cfg.training.calm_generation.sft_mix_samples
    rng = random.Random(cfg.seed)
    if out_path is None:
        out_path = Path(cfg.output_dir) / "training" / "sft.jsonl"

    calm_samples: list[dict] = []
    for rec in read_jsonl(calm_path):
        if rec["mode"] != "calm":
            continue
        turns = rec["turns"]
        if any(t["score"] > 1 for t in turns):  # whole conversation must be calm
            continue
        # Emit 1-, 2-, 3-turn conversations for turn-count variety.
        for length in range(1, len(turns) + 1):
            prior = [turns[i]["response"] for i in range(length - 1)]
            ctx = _context_messages(rec["initial"], rec["rejections"], prior)
            ctx.append({"role": "assistant", "content": turns[length - 1]["response"]})
            calm_samples.append({"messages": ctx})

    rng.shuffle(calm_samples)
    calm_samples = calm_samples[:n_calm]

    dataset = list(calm_samples)
    if include_mix:
        dataset += _load_mix_samples(cfg, n_mix)
    rng.shuffle(dataset)

    write_jsonl(out_path, dataset)
    logger.info("Wrote %d SFT samples (%d calm + mix) to %s",
                len(dataset), len(calm_samples), out_path)
    return str(out_path)
