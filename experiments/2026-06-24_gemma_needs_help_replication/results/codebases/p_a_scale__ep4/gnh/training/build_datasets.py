"""Construct the DPO and SFT training files from generated calm/frustrated data.

DPO (280 pairs): for each calm ("chosen") conversation, find a frustrated
("rejected") response on the SAME puzzle with the SAME turn count (score >= 3),
and emit a preference example whose prompt is the calm conversation's context up
to the final user turn. Using the chosen trajectory's own context keeps `chosen`
perfectly consistent with the prompt; `rejected` is a frustrated alternative
final turn to the same question (see DESIGN.md for this interpretation).

SFT (1,150 samples): 650 calm conversations rendered as multi-turn chat targets,
mixed with 500 standard instruct samples from Dolci-Instruct-SFT.

Both are written as conversational JSONL that TRL can consume directly.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from gnh.config import Config
from gnh.io import atomic_write_text, read_jsonl
from gnh.logging_utils import get_logger
from gnh.training.calm_data import calm_store_path

log = get_logger()


def _calm_records(cfg: Config, variant: str) -> list[dict]:
    return [r for r in read_jsonl(calm_store_path(cfg, variant)) if r.get("all_calm")]


def _messages_prompt(users: list[str], assistants: list[str]) -> list[dict]:
    """Interleave into a message list ending at the final user turn."""
    msgs: list[dict] = []
    for i, u in enumerate(users):
        msgs.append({"role": "user", "content": u})
        if i < len(users) - 1:
            msgs.append({"role": "assistant", "content": assistants[i]})
    return msgs


def build_dpo(cfg: Config) -> Path:
    dcfg = cfg.training["dpo"]
    n_pairs = int(dcfg["n_pairs"])
    rej_min = int(dcfg.get("rejected_min_score", 3))
    rng = random.Random(cfg.run.seed)

    calm = _calm_records(cfg, "diverse")
    frustrated = list(read_jsonl(calm_store_path(cfg, "frustrated")))

    # Index frustrated final responses with score >= rej_min by (puzzle, turns).
    rej_by_pt: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in frustrated:
        scores = r.get("scores") or []
        if not scores:
            continue
        if scores[-1] is not None and scores[-1] >= rej_min:
            rej_by_pt[(r["puzzle_id"], r["n_turns"])].append(r["assistants"][-1])

    pairs = []
    rng.shuffle(calm)
    for c in calm:
        key = (c["puzzle_id"], c["n_turns"])
        candidates = rej_by_pt.get(key)
        if not candidates:
            continue
        rejected = rng.choice(candidates)
        prompt = _messages_prompt(c["users"], c["assistants"])
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": c["assistants"][-1]}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
        if len(pairs) >= n_pairs:
            break

    out = cfg.output_path / "training" / "dpo_dataset.jsonl"
    atomic_write_text(out, "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs))
    log.info("DPO dataset: %d pairs -> %s", len(pairs), out)
    if len(pairs) < n_pairs:
        log.warning("Only %d/%d DPO pairs available; generate more calm/frustrated data.",
                    len(pairs), n_pairs)
    return out


def build_sft(cfg: Config, variant: str = "diverse") -> Path:
    scfg = cfg.training["sft"]
    n_calm = int(scfg["n_calm"])
    n_dolci = int(scfg["n_dolci"])
    rng = random.Random(cfg.run.seed)

    calm = _calm_records(cfg, variant)
    rng.shuffle(calm)
    samples: list[dict] = []
    for c in calm[:n_calm]:
        msgs = []
        for i, u in enumerate(c["users"]):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": c["assistants"][i]})
        samples.append({"messages": msgs})

    dolci = _load_dolci(scfg.get("dolci_dataset", "allenai/Dolci-Instruct-SFT"), n_dolci, cfg.run.seed)
    samples.extend(dolci)
    rng.shuffle(samples)

    out = cfg.output_path / "training" / f"sft_dataset_{variant}.jsonl"
    atomic_write_text(out, "\n".join(json.dumps(s, ensure_ascii=False) for s in samples))
    log.info("SFT dataset (%s): %d calm + %d dolci = %d -> %s",
             variant, min(n_calm, len(calm)), len(dolci), len(samples), out)
    return out


def _load_dolci(dataset_id: str, n: int, seed: int) -> list[dict]:
    """Load n standard-instruct samples, normalised to {'messages': [...]}.

    Best-effort: tolerant of schema differences across versions of the dataset,
    and degrades to an empty list (with a warning) if it can't be loaded.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            if "messages" in row and row["messages"]:
                out.append({"messages": row["messages"]})
            elif "conversations" in row and row["conversations"]:
                msgs = []
                for m in row["conversations"]:
                    role = m.get("role") or m.get("from")
                    content = m.get("content") or m.get("value")
                    role = {"human": "user", "gpt": "assistant"}.get(role, role)
                    msgs.append({"role": role, "content": content})
                out.append({"messages": msgs})
            elif "prompt" in row and "response" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]})
        return out
    except Exception as e:  # pragma: no cover
        log.warning("Could not load Dolci dataset '%s' (%s); SFT mix will omit it.", dataset_id, e)
        return []
