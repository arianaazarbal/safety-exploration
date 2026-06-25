"""Build DPO and SFT datasets from the calm/frustrated pools (Section 4.1).

DPO (280 pairs): for matching (question, turn-count), pair a frustrated
response (score >=3, "rejected") with a calm response (score 0-1, "chosen").
Prompt = the standard (un-reassured) conversation context preceding the turn.

SFT (1,150 samples): 650 calm responses (full stripped conversations) + 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
Two variants: diverse (DPO's calm data) and teacher (Appendix F).

Datasets are written in TRL's conversational format so the trainer applies the
chat template itself.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import Config, load_config

DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"  # best-effort id; see DESIGN.md


def _read_pool(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run generate_calm first")
    return [json.loads(l) for l in open(path) if l.strip()]


# -- DPO ----------------------------------------------------------------------
def build_dpo(cfg: Config, *, seed: int = 0) -> Path:
    rng = random.Random(seed)
    calm = _read_pool(cfg.paths["data_dir"] / "calm_diverse_pool.jsonl")
    frustrated = _read_pool(cfg.paths["data_dir"] / "frustrated_pool.jsonl")
    target = cfg.preset["training"]["dpo_pairs"]

    calm_by_key = defaultdict(list)
    for r in calm:
        calm_by_key[(r["question_id"], r["turn_count"])].append(r)
    calm_by_turn = defaultdict(list)
    for r in calm:
        calm_by_turn[r["turn_count"]].append(r)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr["question_id"], fr["turn_count"])
        candidates = calm_by_key.get(key) or calm_by_turn.get(fr["turn_count"])
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        # prompt is the frustrated context (no reassurance -> matches inference)
        prompt_msgs = fr["history"]
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "assistant", "content": fr["response"]}],
            "meta": {"question_id": fr["question_id"], "turn_count": fr["turn_count"],
                     "rejected_score": fr["score"], "chosen_score": chosen["score"]},
        })
        if len(pairs) >= target:
            break

    out = cfg.paths["data_dir"] / "dpo_dataset.jsonl"
    with open(out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[dpo] built {len(pairs)} preference pairs -> {out}")
    return out


# -- SFT ----------------------------------------------------------------------
def _load_dolci(n: int, seed: int) -> list[dict]:
    """Return `n` instruct samples in conversational format from Dolci, or [] if
    unavailable (the run still proceeds with calm-only SFT, see DESIGN.md)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] Dolci unavailable ({exc!r}); proceeding without instruct mix")
        return []


def _calm_to_sft(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        messages = list(r["history"]) + [{"role": "assistant", "content": r["response"]}]
        out.append({"messages": messages})
    return out


def build_sft(cfg: Config, variant: str = "diverse", *, seed: int = 0) -> Path:
    rng = random.Random(seed)
    pool_name = "calm_teacher_pool.jsonl" if variant == "teacher" else "calm_diverse_pool.jsonl"
    calm = _read_pool(cfg.paths["data_dir"] / pool_name)
    n_calm = cfg.preset["training"]["calm_target_sft"]
    n_dolci = cfg.preset["training"]["dolci_mix"]

    rng.shuffle(calm)
    calm_sft = _calm_to_sft(calm[:n_calm])
    dolci = _load_dolci(n_dolci, seed)
    dataset = calm_sft + dolci
    rng.shuffle(dataset)

    out = cfg.paths["data_dir"] / f"sft_{variant}.jsonl"
    with open(out, "w") as fh:
        for ex in dataset:
            fh.write(json.dumps(ex) + "\n")
    print(f"[sft:{variant}] {len(calm_sft)} calm + {len(dolci)} dolci = {len(dataset)} -> {out}")
    return out


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    build_dpo(cfg)
    build_sft(cfg, "diverse")
    build_sft(cfg, "teacher")


if __name__ == "__main__":
    main()
