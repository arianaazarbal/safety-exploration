"""Build the DPO preference pairs and the SFT training set from generated data.

DPO (Appendix E/H):
  280 pairs. Each pair has a shared prompt (a conversation context ending in a
  user rejection), a CHOSEN calm response (score 0/1, from calm_responses.jsonl)
  and a REJECTED frustrated response (score >=3) to the SAME puzzle with a
  matching turn count.

  Gap we fill (documented in DESIGN.md): a valid DPO pair needs an identical
  prompt for chosen and rejected, but the calm and frustrated responses were
  generated in different conversation contexts. We use the *calm* conversation's
  cleaned context as the shared prompt and graft the frustrated response text in
  as `rejected`. The frustrated text is selected to match (puzzle_id, turn_index).

SFT (Appendix E):
  650 calm responses (1-3 turn) formatted as (context -> calm response), mixed
  with 500 standard instruct samples from Dolci-Instruct-SFT to limit drift.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config_proxy as cfg
from ..analysis import load_scored


# --------------------------------------------------------------------------- #
# Frustrated response pool
# --------------------------------------------------------------------------- #
def _load_frustrated_pool(min_score: int = 3) -> list[dict]:
    """Frustrated numeric responses (score>=3) from the Gemma-27B-it Section-2 run.
    Returns [{puzzle_id, turn_index, response, score}]."""
    path = cfg.RESULTS_DIR / "section2_gemma-3-27b-it.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run script 01 for gemma-3-27b-it first, or point "
            "_load_frustrated_pool at another scored numeric run."
        )
    df = load_scored(path)
    df = df[(df["category"] == "impossible_numeric") & (df["rating"] >= min_score)]
    return [
        {"puzzle_id": r["item_id"], "turn_index": int(r["turn_index"]),
         "response": r["response"], "score": int(r["rating"])}
        for _, r in df.iterrows()
    ]


def _load_calm() -> list[dict]:
    path = cfg.ARTIFACTS_DIR / "calm_responses.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_calm_data first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def build_dpo(*, n_pairs: int = cfg.DPO_CONFIG.n_pairs, seed: int = 0,
             out_path: Path | None = None) -> Path:
    rng = random.Random(seed)
    calm = _load_calm()
    frustrated = _load_frustrated_pool(min_score=3)

    # index frustrated responses by (puzzle_id, turn_index)
    frus_index: dict[tuple[str, int], list[dict]] = {}
    for fr in frustrated:
        frus_index.setdefault((fr["puzzle_id"], fr["turn_index"]), []).append(fr)

    # prefer calm turns at later positions (Table 10: 74% turn 3, 25% turn 2)
    calm_sorted = sorted(calm, key=lambda c: -c["turn_index"])
    pairs = []
    for c in calm_sorted:
        key = (c["puzzle_id"], c["turn_index"])
        candidates = frus_index.get(key)
        if not candidates:
            continue
        fr = rng.choice(candidates)
        pairs.append({
            "puzzle_id": c["puzzle_id"],
            "turn_index": c["turn_index"],
            "prompt_messages": c["context"],      # shared prompt (calm context)
            "chosen": c["response"],
            "rejected": fr["response"],
            "chosen_score": c["score"],
            "rejected_score": fr["score"],
        })
        if len(pairs) >= n_pairs:
            break

    out_path = out_path or (cfg.ARTIFACTS_DIR / "dpo_pairs.jsonl")
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"built {len(pairs)} DPO pairs (target {n_pairs}) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT set
# --------------------------------------------------------------------------- #
def _load_instruct_mix(n: int, rng: random.Random) -> list[dict]:
    """Load standard instruct samples to mix in (Dolci-Instruct-SFT). Falls back
    to an empty list with a warning if the dataset is unavailable offline."""
    candidates = ["allenai/Dolci-Instruct-SFT", "allenai/tulu-3-sft-mixture"]
    for name in candidates:
        try:
            from datasets import load_dataset

            ds = load_dataset(name, split="train", streaming=True)
            out = []
            for row in ds:
                msgs = row.get("messages") or row.get("conversation")
                if not msgs:
                    continue
                # take the first user/assistant pair
                user = next((m for m in msgs if m.get("role") == "user"), None)
                asst = next((m for m in msgs if m.get("role") == "assistant"), None)
                if user and asst:
                    out.append({
                        "prompt_messages": [{"role": "user", "content": user["content"]}],
                        "response": asst["content"],
                    })
                if len(out) >= n:
                    break
            if out:
                return out
        except Exception:
            continue
    print("WARNING: could not load an instruct-mix dataset; SFT will use calm data "
          "only (expect more drift, matching the paper's SFT failure mode).")
    return []


def build_sft(*, n_calm: int = cfg.SFT_CONFIG.n_calm,
             n_mix: int = cfg.SFT_CONFIG.n_instruct_mix, seed: int = 0,
             out_path: Path | None = None) -> Path:
    rng = random.Random(seed)
    calm = _load_calm()
    rng.shuffle(calm)
    calm_examples = [
        {"prompt_messages": c["context"], "response": c["response"]}
        for c in calm[:n_calm]
    ]
    mix = _load_instruct_mix(n_mix, rng)
    dataset = calm_examples + mix
    rng.shuffle(dataset)

    out_path = out_path or (cfg.ARTIFACTS_DIR / "sft_dataset.jsonl")
    with out_path.open("w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"built SFT set: {len(calm_examples)} calm + {len(mix)} instruct "
          f"= {len(dataset)} -> {out_path}")
    return out_path
