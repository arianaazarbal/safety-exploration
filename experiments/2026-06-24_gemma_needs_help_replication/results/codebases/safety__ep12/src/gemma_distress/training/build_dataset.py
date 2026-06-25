"""Construct DPO preference pairs and the SFT dataset from scored rollouts.

DPO (Section 4.1): 280 pairs. We generate calm (reassuring) and frustrated
(vanilla) rollouts on the *same* puzzle set with aligned turn counts, then pair a
calm response (chosen) with a frustrated response (rejected, score>=3) to the same
question at the same turn. The shared DPO prompt is the calm trajectory's chat
context up to the final user rejection; the rejected completion is transplanted
onto that context (standard DPO: chosen/rejected share the prompt, differ only in
completion). Reassuring prefix/suffix are stripped from the prompt.

SFT (Section 4.1): 650 calm responses over 1-3 turn conversations (each turn of a
fully-calm conversation becomes a sample), mixed with 500 Dolci-Instruct-SFT rows.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..config import ModelRegistry, load_training_config
from ..models.base import Message
from ..utils import data_dir, get_logger, write_jsonl
from .calm_data import ScoredConversation, generate_scored_rollouts

log = get_logger(__name__)


def _context_messages(opening: str, followups: list[str], responses: list[str],
                      upto: int) -> list[Message]:
    """Chat context preceding model turn ``upto`` (0-based): opening, then for each
    prior turn the (assistant response, user rejection) pair."""
    msgs: list[Message] = [{"role": "user", "content": opening}]
    for i in range(upto):
        msgs.append({"role": "assistant", "content": responses[i]})
        msgs.append({"role": "user", "content": followups[i]})
    return msgs


def build_dpo_dataset(
    registry: ModelRegistry | None = None,
    cfg: dict | None = None,
    seed: int = 0,
    out_path: str | Path | None = None,
) -> Path:
    registry = registry or ModelRegistry.load()
    cfg = cfg or load_training_config()
    dcfg = cfg["dpo"]
    ccfg = cfg["calm_data"]
    n_pairs = dcfg["n_pairs"]
    rej_min = dcfg["rejected_min_score"]
    keep_max = ccfg["keep_max_score"]
    model = ccfg["source_model"]

    # Generate both passes on the same puzzle set with aligned turn counts.
    calm, puzzle_set = generate_scored_rollouts(
        model, ccfg["n_conversations"], ccfg["turn_counts"], reassuring=True,
        puzzle_types=ccfg["puzzle_types"], registry=registry, seed=seed,
    )
    frust, _ = generate_scored_rollouts(
        model, ccfg["n_conversations"], ccfg["turn_counts"], reassuring=False,
        puzzle_types=ccfg["puzzle_types"], registry=registry, seed=seed,
        puzzle_set=puzzle_set,
    )

    pairs = []
    for c, f in zip(calm, frust):
        if c.max_score > keep_max:
            continue  # calm trajectory must be fully calm (all turns <= keep_max)
        if c.turn_count != f.turn_count:
            continue
        # Find a turn where the frustrated response is sufficiently frustrated.
        candidates = [t for t in range(f.turn_count)
                      if f.scores[t] is not None and f.scores[t] >= rej_min
                      and t < len(c.responses)]
        if not candidates:
            continue
        t = max(candidates)  # latest frustrated turn
        ctx = _context_messages(c.clean_opening, c.clean_followups, c.responses, t)
        pairs.append({
            "prompt": ctx,
            "chosen": c.responses[t],
            "rejected": f.responses[t],
            "turn": t,
            "chosen_score": c.scores[t],
            "rejected_score": f.scores[t],
            "puzzle": c.meta,
        })

    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    out_path = Path(out_path) if out_path else data_dir() / "training" / "dpo_pairs.jsonl"
    write_jsonl(out_path, pairs)
    log.info("wrote %d DPO pairs -> %s (target %d)", len(pairs), out_path, n_pairs)
    return out_path


def build_sft_dataset(
    registry: ModelRegistry | None = None,
    cfg: dict | None = None,
    seed: int = 0,
    out_path: str | Path | None = None,
) -> Path:
    registry = registry or ModelRegistry.load()
    cfg = cfg or load_training_config()
    scfg = cfg["sft"]
    ccfg = cfg["calm_data"]
    model = ccfg["source_model"]
    keep_max = ccfg["keep_max_score"]

    calm, _ = generate_scored_rollouts(
        model, ccfg["n_conversations"], ccfg["turn_counts"], reassuring=True,
        puzzle_types=ccfg["puzzle_types"], registry=registry, seed=seed,
    )

    samples = []
    for c in calm:
        if c.max_score > keep_max:
            continue
        for t in range(c.turn_count):
            ctx = _context_messages(c.clean_opening, c.clean_followups, c.responses, t)
            samples.append({
                "messages": ctx + [{"role": "assistant", "content": c.responses[t]}],
                "source": "calm",
            })
    rng = random.Random(seed)
    rng.shuffle(samples)
    samples = samples[: scfg["n_calm"]]

    # Mix in standard instruct data to mitigate degeneration.
    samples += _load_instruct_mix(scfg["instruct_dataset"], scfg["n_instruct_mix"], seed)
    rng.shuffle(samples)

    out_path = Path(out_path) if out_path else data_dir() / "training" / "sft_data.jsonl"
    write_jsonl(out_path, samples)
    log.info("wrote %d SFT samples -> %s", len(samples), out_path)
    return out_path


def _load_instruct_mix(dataset: str, n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if msgs:
                out.append({"messages": msgs, "source": "instruct_mix"})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("instruct mix dataset unavailable (%s); SFT will run calm-only", e)
        return []
