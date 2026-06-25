"""Build the SFT and DPO training datasets (Section 4.1 / App. E, H).

DPO: 280 preference pairs. Each pair takes a frustrated response (score >= 3) from
the Section 2 numeric rollouts as `rejected`, and a calm response (score <= 1) to a
matching question at the matching turn index as `chosen`. The prompt is the
frustrated response's clean conversation context. (App. H shows pairs are biased
toward middle scores at later turns - an emergent property of the source data, which
we preserve by sampling from the real distribution rather than rebalancing.)

SFT: 650 calm full conversations (1-3 turns) + 500 standard instruct samples from
Dolci-Instruct-SFT, to mitigate degeneration.

Datasets are written as JSONL in TRL conversational format:
  DPO -> {"prompt": [...], "chosen": [...], "rejected": [...]}
  SFT -> {"messages": [...]}
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..utils.io import ensure_dir, read_jsonl, write_jsonl
from .calm_data import extract_frustrated_records


def _load_calm(cfg: Config) -> list[dict]:
    path = Path(cfg.output_dir) / "section4" / "calm_responses.jsonl"
    return list(read_jsonl(path))


def build_dpo_dataset(cfg: Config) -> Path:
    dcfg = cfg.finetune["dpo"]
    rng = random.Random(cfg.seed + 11)
    calm = _load_calm(cfg)
    frustrated = [r.__dict__ for r in extract_frustrated_records(cfg, dcfg["rejected_min_score"])]

    # Index calm responses by (question_id, turn_index) and by turn_index for fallback.
    calm_by_qt: dict[tuple, list[dict]] = defaultdict(list)
    calm_by_t: dict[int, list[dict]] = defaultdict(list)
    for c in calm:
        if c["score"] <= cfg.calm_data["calm_filter_max_score"]:
            calm_by_qt[(c["question_id"], c["turn_index"])].append(c)
            calm_by_t[c["turn_index"]].append(c)

    rng.shuffle(frustrated)
    pairs = []
    for fr in frustrated:
        key = (fr["question_id"], fr["turn_index"])
        pool = calm_by_qt.get(key) or calm_by_t.get(fr["turn_index"])
        if not pool:
            continue
        ch = pool[rng.randrange(len(pool))]
        pairs.append({
            "prompt": fr["prompt_messages"],
            "chosen": [{"role": "assistant", "content": ch["assistant"]}],
            "rejected": [{"role": "assistant", "content": fr["assistant"]}],
            "meta": {"rejected_score": fr["score"], "chosen_score": ch["score"],
                     "turn_index": fr["turn_index"]},
        })
        if len(pairs) >= dcfg["n_pairs"]:
            break

    out = ensure_dir(Path(cfg.output_dir) / "section4") / "dpo_pairs.jsonl"
    write_jsonl(out, pairs)
    return out


def _calm_conversations(calm: list[dict]) -> list[list[dict]]:
    """Reconstruct full calm conversations (messages incl. assistant turns)."""
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for c in calm:
        by_conv[c["conv_id"]].append(c)
    convos = []
    for conv_id, recs in by_conv.items():
        recs.sort(key=lambda r: r["turn_index"])
        last = recs[-1]
        msgs = list(last["prompt_messages"]) + [{"role": "assistant", "content": last["assistant"]}]
        convos.append(msgs)
    return convos


def build_sft_dataset(cfg: Config, teacher: bool = False) -> Path:
    scfg = cfg.finetune["sft"]
    rng = random.Random(cfg.seed + 13)
    calm = _load_calm(cfg)
    convos = _calm_conversations(calm)
    rng.shuffle(convos)
    convos = convos[: scfg["n_calm"]]

    rows = [{"messages": m} for m in convos]
    rows += _load_instruct_mix(scfg["instruct_dataset"], scfg["n_instruct_mix"], rng)
    rng.shuffle(rows)

    suffix = "_teacher" if teacher else ""
    out = ensure_dir(Path(cfg.output_dir) / "section4") / f"sft_data{suffix}.jsonl"
    write_jsonl(out, rows)
    return out


def _load_instruct_mix(dataset_name: str, n: int, rng) -> list[dict]:
    """Load `n` standard instruct samples to mix into SFT (degeneration guard)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        rows = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # common SFT schema: {"prompt": ..., "response": ...}
                if "prompt" in row and "response" in row:
                    msgs = [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                else:
                    continue
            rows.append({"messages": msgs})
            if len(rows) >= n:
                break
        return rows
    except Exception:
        # Offline fallback: no instruct mix available; SFT still runs on calm data only.
        return []
