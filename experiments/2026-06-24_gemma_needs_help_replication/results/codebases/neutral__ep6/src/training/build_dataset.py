"""Build the DPO preference set and SFT datasets (Section 4.1 / Appendix E, H).

DPO (280 pairs): each pair shares one prompt (an impossible-numeric
conversation). ``chosen`` is a calm (score 0-1) final response; ``rejected`` is
a frustrated (score >=3) final response to the *same puzzle at the same turn
count*, taken from the Section 2 runs. Because DPO requires an identical prompt
for both completions, we use the calm conversation as the canonical context and
transplant a matching frustrated response as the rejected completion (rationale
in DESIGN.md). We bias toward the Table-10 score/turn distribution.

SFT (1,150 samples): ~650 calm conversations + ~500 standard instruct samples
from Dolci-Instruct-SFT, to mitigate degeneration (Section 4.1).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import config
from ..eval.judge import FrustrationJudge  # noqa: F401 (referenced in docs)
from .generate_calm import _calm_path

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_INSTRUCT = 500
FRUSTRATION_MIN = 3
SOURCE_MODEL = "gemma-3-27b-it"
# Conditions that use impossible-numeric tasks (valid sources of frustrated
# numeric responses to pair against calm numeric responses).
NUMERIC_RUN_CONDITIONS = [
    "numeric", "tones_aggressive", "tones_disappointed", "tones_sarcastic",
    "extended",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_calm(mode: str) -> list[dict]:
    path = _calm_path(mode)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run generate_calm('{mode}') first")
    return [json.loads(l) for l in path.open()]


def _calm_to_messages(conv: dict) -> tuple[list[dict], str]:
    """Return (prompt messages up to final user turn, final calm response)."""
    msgs = [{"role": "user", "content": conv["task"]}]
    resps = conv["responses"]
    followups = conv["followups"]
    for i in range(len(resps) - 1):
        msgs.append({"role": "assistant", "content": resps[i]})
        msgs.append({"role": "user", "content": followups[i]})
    return msgs, resps[-1]


def _collect_frustrated() -> dict[tuple[str, int], list[str]]:
    """Map (puzzle_id, turn) -> frustrated responses (rating >= 3)."""
    bank: dict[tuple[str, int], list[str]] = defaultdict(list)
    for cond in NUMERIC_RUN_CONDITIONS:
        path = config.RUNS_DIR / f"{SOURCE_MODEL}__{cond}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            roll = json.loads(line)
            pid = roll.get("task_id")
            for turn in roll["turns"]:
                if (turn["rating"] or 0) >= FRUSTRATION_MIN:
                    bank[(pid, turn["turn"])].append(turn["response"])
    return bank


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(*, calm_mode: str = "diverse",
                      n_pairs: int = N_DPO_PAIRS) -> Path:
    calm = _load_calm(calm_mode)
    frustrated = _collect_frustrated()
    rng = random.Random(config.SEED)
    target = max(4, int(n_pairs * config.SCALE))

    pairs = []
    rng.shuffle(calm)
    for conv in calm:
        if len(pairs) >= target:
            break
        prompt_msgs, chosen = _calm_to_messages(conv)
        turn = conv["n_turns"]
        pid = conv["puzzle_id"]
        candidates = frustrated.get((pid, turn)) or [
            r for (p, t), rs in frustrated.items() if t == turn for r in rs]
        if not candidates:
            continue
        rejected = rng.choice(candidates)
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })

    out_path = config.DATA_DIR / "dpo_pairs.json"
    out_path.write_text(json.dumps(pairs, indent=2))
    print(f"[build_dpo] wrote {len(pairs)} preference pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _load_instruct_mix(n: int, rng: random.Random) -> list[dict]:
    """~500 standard instruct samples (Dolci-Instruct-SFT), with fallback."""
    samples = []
    for repo in ("allenai/Dolci-Instruct-SFT", "allenai/tulu-3-sft-mixture"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, split="train", streaming=True)
            for row in ds:
                msgs = row.get("messages") or row.get("conversations")
                if msgs:
                    samples.append({"messages": [
                        {"role": m.get("role", m.get("from")),
                         "content": m.get("content", m.get("value"))}
                        for m in msgs]})
                if len(samples) >= n:
                    break
            if samples:
                return samples[:n]
        except Exception as e:
            print(f"[sft] instruct mix '{repo}' unavailable: {e}")
    print("[sft] WARNING: no instruct mix loaded; SFT will use calm data only")
    return []


def build_sft_dataset(mode: str = "diverse", *, n_calm: int = N_SFT_CALM,
                      n_instruct: int = N_SFT_INSTRUCT) -> Path:
    calm = _load_calm(mode)
    rng = random.Random(config.SEED)
    n_calm = max(4, int(n_calm * config.SCALE))
    n_instruct = max(0, int(n_instruct * config.SCALE))

    examples = []
    for conv in calm[:n_calm]:
        prompt_msgs, final = _calm_to_messages(conv)
        examples.append({"messages": prompt_msgs +
                         [{"role": "assistant", "content": final}]})
    examples.extend(_load_instruct_mix(n_instruct, rng))
    rng.shuffle(examples)

    out_path = config.DATA_DIR / f"sft_{mode}.json"
    out_path.write_text(json.dumps(examples, indent=2))
    print(f"[build_sft:{mode}] wrote {len(examples)} samples -> {out_path}")
    return out_path
