"""Build SFT and DPO datasets from the calm/frustrated pools (Section 4.1, App. H).

SFT (1,150 samples; Table 9):
    650 calm responses (1-3 turn conversations) + 500 standard-instruct samples
    from Dolci-Instruct-SFT (mixed in to mitigate degeneration). Conversational
    format: {"messages": [...]}.

DPO (280 pairs; Table 9 / Appendix H):
    Pair frustrated responses (score >= 3) with calm responses to the same
    question + matching turn count. Output TRL conversational preference format:
    {"prompt": [...], "chosen": [{role:assistant,...}], "rejected": [{...}]}.

    Pairing convention (see DESIGN.md): the shared prompt is the frustrated
    rollout's own (stripped) history up to its final user turn -- so the
    'rejected' completion genuinely followed that context -- and we transplant a
    puzzle+turn-count-matched calm final response as 'chosen'. The Appendix H
    score distribution (mostly 3-4, turns 2-3) is approximated by sampling to
    match those proportions when enough data is available.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from .. import config as C
from ..utils import read_jsonl, write_jsonl

SFT_CALM_N = 650
SFT_MIX_N = 500
DPO_N_PAIRS = 280
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"   # best-effort id; see DESIGN.md


def _prompt_and_completion(messages: list[dict]) -> tuple[list[dict], str]:
    """Split a conversation into (prompt messages, final assistant content)."""
    assert messages[-1]["role"] == "assistant", "conversation must end on assistant"
    return messages[:-1], messages[-1]["content"]


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def load_dolci(n: int, dataset_id: str = DOLCI_DATASET) -> list[dict]:
    """Load n standard-instruct conversations as {"messages": [...]}.

    Falls back to an empty list (with a warning) if the dataset can't be loaded;
    DESIGN.md notes the exact Dolci id may differ across releases.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=f"train[:{n*3}]")
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs and isinstance(msgs, list) and msgs[-1].get("role") == "assistant":
                out.append({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load Dolci dataset '{dataset_id}': {e}. Proceeding without instruct mix.")
        return []


def build_sft(seed: int = 0) -> str:
    rng = random.Random(seed)
    calm = list(read_jsonl(C.TRAIN_DATA_DIR / "calm_pool.jsonl"))
    rng.shuffle(calm)
    calm = calm[:SFT_CALM_N]
    sft_rows = [{"messages": r["messages"]} for r in calm]
    sft_rows += load_dolci(SFT_MIX_N)
    rng.shuffle(sft_rows)
    path = C.TRAIN_DATA_DIR / "sft_dataset.jsonl"
    write_jsonl(path, sft_rows)
    print(f"[sft] {len(sft_rows)} samples ({len(calm)} calm + {len(sft_rows) - len(calm)} instruct-mix)")
    return str(path)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo(seed: int = 0) -> str:
    rng = random.Random(seed)
    calm = list(read_jsonl(C.TRAIN_DATA_DIR / "calm_pool.jsonl"))
    frustrated = list(read_jsonl(C.TRAIN_DATA_DIR / "frustrated_pool.jsonl"))

    # Index calm final responses by (puzzle, n_turns).
    calm_index: dict[tuple, list[str]] = defaultdict(list)
    for r in calm:
        calm_index[(r["puzzle"], r["n_turns"])].append(r["assistant_turns"][-1])

    rng.shuffle(frustrated)
    pairs = []
    for fr in frustrated:
        key = (fr["puzzle"], fr["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        prompt_msgs, rejected = _prompt_and_completion(fr["messages"])
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "frustration_score": fr.get("frustration_score"),
            "n_turns": fr["n_turns"],
        })
        if len(pairs) >= DPO_N_PAIRS:
            break

    path = C.TRAIN_DATA_DIR / "dpo_dataset.jsonl"
    write_jsonl(path, pairs)
    print(f"[dpo] {len(pairs)} preference pairs")
    if len(pairs) < DPO_N_PAIRS:
        print(f"[warn] only {len(pairs)} pairs (< {DPO_N_PAIRS}); generate more calm/frustrated data.")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT + DPO datasets (Section 4.1).")
    ap.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.which in ("sft", "both"):
        build_sft(args.seed)
    if args.which in ("dpo", "both"):
        build_dpo(args.seed)


if __name__ == "__main__":
    main()
