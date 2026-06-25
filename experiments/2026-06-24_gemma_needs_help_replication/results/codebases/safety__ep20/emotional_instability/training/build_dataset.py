"""Build the DPO preference pairs and the SFT dataset (Section 4.1, Table 9).

DPO: pair each frustrated response (score >= 3) with a calm response (score 0-1)
to the *same puzzle and turn count*; the prompt is the frustrated sample's
conversation context. Target 280 pairs (Table 10's distribution is honoured
loosely by drawing in natural score/turn proportions).

SFT: 650 calm responses as conversational examples, mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .. import config
from .generate_calm_data import Sample


def _assistant(text: str) -> List[dict]:
    return [{"role": "assistant", "content": text}]


def build_dpo_pairs(
    calm: List[Sample],
    frustrated: List[Sample],
    n_pairs: int = config.DPO.n_pairs,
    seed: int = 0,
) -> List[dict]:
    rng = random.Random(seed)
    calm_by_key: Dict[Tuple[str, int], List[Sample]] = defaultdict(list)
    for s in calm:
        calm_by_key[(s.puzzle_id, s.turn)].append(s)

    pairs: List[dict] = []
    frustrated = frustrated[:]
    rng.shuffle(frustrated)
    for fr in frustrated:
        candidates = calm_by_key.get((fr.puzzle_id, fr.turn))
        if not candidates:
            continue
        ca = rng.choice(candidates)
        pairs.append({
            "prompt": fr.context,
            "chosen": _assistant(ca.response),
            "rejected": _assistant(fr.response),
            "meta": {"puzzle_id": fr.puzzle_id, "turn": fr.turn,
                     "chosen_score": ca.score, "rejected_score": fr.score},
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def _load_instruct_mix(n: int, dataset_name: str, seed: int) -> List[dict]:
    """Load `n` standard instruct samples as conversational examples."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        rng = random.Random(seed)
        out: List[dict] = []
        for row in ds:
            msgs = row.get("messages")
            if not msgs:
                # try common (instruction, response) schemas
                instr = row.get("instruction") or row.get("prompt")
                resp = row.get("response") or row.get("output") or row.get("completion")
                if instr and resp:
                    msgs = [{"role": "user", "content": instr},
                            {"role": "assistant", "content": resp}]
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n * 3:
                break
        rng.shuffle(out)
        return out[:n]
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] could not load instruct mix {dataset_name!r} ({exc!r}); "
              "proceeding without it.")
        return []


def build_sft_dataset(
    calm: List[Sample],
    n_calm: int = config.SFT.n_calm,
    n_instruct: int = config.SFT.n_instruct_mix,
    instruct_dataset: str = config.SFT.instruct_dataset,
    seed: int = 0,
) -> List[dict]:
    rng = random.Random(seed)
    calm = calm[:]
    rng.shuffle(calm)
    examples = [
        {"messages": s.context + _assistant(s.response)}
        for s in calm[:n_calm]
    ]
    examples.extend(_load_instruct_mix(n_instruct, instruct_dataset, seed))
    rng.shuffle(examples)
    return examples


def save_jsonl(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[dataset] wrote {len(records)} -> {path}")


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_samples(path: str) -> List[Sample]:
    with open(path) as f:
        return [Sample(**json.loads(line)) for line in f]
