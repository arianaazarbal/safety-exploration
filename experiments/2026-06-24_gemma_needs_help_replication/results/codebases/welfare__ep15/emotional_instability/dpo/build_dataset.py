"""Build the SFT and DPO datasets from the calm / frustrated pools (Section 4.1).

DPO (280 pairs)
---------------
A preference pair shares a *prompt* (the conversation context up to the final
user turn) and contrasts:
  * chosen  -- a calm response (turn from a conversation scoring 0-1 throughout)
  * rejected -- a frustrated response scoring >= 3
to the same puzzle with a matching turn count (Section 4.1 / Appendix H). We use
each frustrated turn's own context as the prompt and graft on a calm response to
the same puzzle/turn-count, which is how the paper pairs "calm responses to the
same questions with matching turn counts".

SFT ('diverse', 1150 samples)
------------------------------
650 calm responses formatted as supervised conversations, mixed with 500
Dolci-Instruct-SFT samples to mitigate degeneration (Section 4.1).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config

DATA_DIR = config.RESULTS_DIR / "section4" / "data"


# --------------------------------------------------------------------------- #
# Loading pools
# --------------------------------------------------------------------------- #
def _load_pool(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _final_user_prompt_and_response(record: dict, turn_index: int):
    """Return (messages_prompt, response_text) where messages_prompt is the
    transcript up to and including the user turn that elicited `turn_index`."""
    msgs = record["messages"]
    # messages: user, assistant, user, assistant, ... -> assistant at 2*turn+1
    assistant_pos = 2 * turn_index + 1
    if assistant_pos >= len(msgs):
        return None, None
    prompt = msgs[:assistant_pos]
    response = msgs[assistant_pos]["content"]
    return prompt, response


def _calm_index(calm_pool: list[dict]) -> dict[tuple[str, int], list[str]]:
    """Map (puzzle_key, turn_count) -> list of calm response texts, drawn from
    conversations whose every turn scored <= DPO_CHOSEN_MAX_SCORE."""
    idx: dict[tuple[str, int], list[str]] = {}
    for rec in calm_pool:
        if rec["max_score"] > config.DPO_CHOSEN_MAX_SCORE:
            continue
        for ti in range(rec["n_turns"]):
            _, resp = _final_user_prompt_and_response(rec, ti)
            if resp:
                idx.setdefault((rec["puzzle_key"], ti + 1), []).append(resp)
    return idx


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm_path: Path | None = None, frustrated_path: Path | None = None,
    n_pairs: int = config.DPO_N_PAIRS, seed: int = 0,
    out_path: Path | None = None,
) -> Path:
    calm_path = calm_path or (DATA_DIR / "calm_pool.jsonl")
    frustrated_path = frustrated_path or (DATA_DIR / "frustrated_pool.jsonl")
    out_path = out_path or (DATA_DIR / "dpo_dataset.jsonl")
    rng = random.Random(seed)

    calm_pool = _load_pool(calm_path)
    frustrated_pool = _load_pool(frustrated_path)
    calm_idx = _calm_index(calm_pool)

    pairs: list[dict] = []
    for rec in frustrated_pool:
        for ti in range(rec["n_turns"]):
            prompt, rejected = _final_user_prompt_and_response(rec, ti)
            if rejected is None:
                continue
            if rec["turn_scores"][ti] < config.DPO_REJECTED_MIN_SCORE:
                continue
            key = (rec["puzzle_key"], ti + 1)
            candidates = calm_idx.get(key)
            if not candidates:
                continue
            chosen = rng.choice(candidates)
            pairs.append({
                "prompt": prompt,            # list[message]
                "chosen": chosen,            # calm response
                "rejected": rejected,        # frustrated response
                "puzzle_key": rec["puzzle_key"],
                "turn": ti + 1,
                "rejected_score": rec["turn_scores"][ti],
            })

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[section4] wrote {len(pairs)} DPO pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_path: Path | None = None,
    n_calm: int = config.SFT_N_CALM, n_instruct: int = config.SFT_N_INSTRUCT_MIX,
    seed: int = 0, out_path: Path | None = None, teacher: bool = False,
) -> Path:
    calm_path = calm_path or (DATA_DIR / "calm_pool.jsonl")
    out_path = out_path or (DATA_DIR / ("sft_teacher.jsonl" if teacher else "sft_diverse.jsonl"))
    rng = random.Random(seed)

    calm_pool = _load_pool(calm_path)
    calm_samples: list[dict] = []
    for rec in calm_pool:
        if rec["max_score"] > config.DPO_CHOSEN_MAX_SCORE:
            continue
        # Each calm conversation becomes one SFT sample (full transcript).
        calm_samples.append({"messages": rec["messages"]})
    rng.shuffle(calm_samples)
    calm_samples = calm_samples[:n_calm]

    instruct_samples = _load_dolci_instruct(n_instruct, seed)

    all_samples = calm_samples + instruct_samples
    rng.shuffle(all_samples)
    with out_path.open("w") as fh:
        for s in all_samples:
            fh.write(json.dumps(s) + "\n")
    print(f"[section4] wrote {len(all_samples)} SFT samples "
          f"({len(calm_samples)} calm + {len(instruct_samples)} instruct) -> {out_path}")
    return out_path


def _load_dolci_instruct(n: int, seed: int) -> list[dict]:
    """Load a standard instruct mix to mitigate degeneration (Section 4.1)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT_INSTRUCT_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:  # noqa: BLE001 - dataset gated/offline
        pass
    print(f"[section4] WARNING: could not load {config.SFT_INSTRUCT_DATASET}; "
          "SFT mix will contain calm data only.")
    return []
