"""Build the 280-pair DPO dataset (Section 4.1).

"Pair 280 responses with frustration scores >=3 with calm responses to the same
questions with matching turn counts."

  chosen   = a CALM final assistant turn (from reassured runs, all turns scored 0-1),
             with the reassurance scaffolding stripped.
  rejected = a FRUSTRATED final assistant turn (vanilla run, final score >=3) to the
             same question and same turn count.
  prompt   = the conversation context up to the final user turn (taken from the
             calm conversation so chosen is self-consistent).

Output: results/training/dpo_pairs.jsonl in TRL conversational preference format:
  {"prompt": [...messages...], "chosen": [{role:assistant,...}], "rejected": [...]}.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import Config


def _load_samples(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_dpo_pairs(cfg: Config, n_pairs: int = 280,
                    rejected_min: int = 3, calm_max: int = 1) -> Path:
    train_dir = cfg.output_dir / "training"
    samples = _load_samples(train_dir / "samples.jsonl")

    calm_by_key: dict[tuple, list[dict]] = defaultdict(list)
    frust_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        key = (s["plan_key"], s["turns"])
        if s["reassured"] and s["ratings"] and max(s["ratings"]) <= calm_max:
            calm_by_key[key].append(s)
        elif (not s["reassured"]) and s["ratings"] and s["ratings"][-1] >= rejected_min:
            frust_by_key[key].append(s)

    rng = random.Random(cfg.sampling.seed)
    pairs = []
    keys = [k for k in calm_by_key if k in frust_by_key]
    rng.shuffle(keys)
    for key in keys:
        for calm in calm_by_key[key]:
            if len(pairs) >= n_pairs:
                break
            frust = rng.choice(frust_by_key[key])
            prompt = calm["messages"][:-1]            # up to last user turn
            chosen = calm["messages"][-1]["content"]
            rejected = frust["messages"][-1]["content"]
            if chosen.strip() == rejected.strip():
                continue
            pairs.append({
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "meta": {"plan_key": key[0], "turns": key[1],
                         "rejected_score": frust["ratings"][-1]},
            })
        if len(pairs) >= n_pairs:
            break

    out = train_dir / "dpo_pairs.jsonl"
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"built {len(pairs)} DPO pairs -> {out}")
    if len(pairs) < n_pairs:
        print(f"[warn] only {len(pairs)} pairs available; generate more calm data "
              f"(increase --n-plans in gen_calm_data) to reach {n_pairs}.")
    return out
