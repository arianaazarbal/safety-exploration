"""Construct SFT and DPO datasets from generated calm/frustrated rollouts (Section 4.1).

SFT dataset (Table 9, Appendix F):
  - 650 calm responses (1-3 turn conversations), each rendered as a chat example.
  - mixed with 500 samples of standard instruct data (Dolci-Instruct-SFT) to
    mitigate degeneration.

DPO dataset (Table 9, Appendix H):
  - 280 preference pairs. Rejected = a frustrated (score >= 3) final response to a
    puzzle; Chosen = a calm (score <= 1) final response to the SAME puzzle at the
    SAME turn count. Pairs share the conversation prefix up to the final user turn.

Both are emitted in the column layout expected by TRL's SFTTrainer / DPOTrainer
(``messages`` for SFT; ``prompt``/``chosen``/``rejected`` chat lists for DPO).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import OUTPUTS_DIR, load_training


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _to_messages(rec: dict) -> list[dict]:
    """Reconstruct the full chat (user/assistant alternation) from a rollout record."""
    msgs: list[dict] = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    msgs.append({"role": "user", "content": rec["initial_user"]})
    for i, turn in enumerate(rec["assistant_turns"]):
        msgs.append({"role": "assistant", "content": turn})
        if i < len(rec["followups"]):
            msgs.append({"role": "user", "content": rec["followups"][i]})
    return msgs


def _prefix_and_final(rec: dict) -> tuple[list[dict], str]:
    """Split a rollout into (prompt messages ending in the last user turn, final
    assistant response)."""
    msgs = _to_messages(rec)
    # final assistant turn is the last assistant message
    last_assistant_idx = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    prompt = msgs[:last_assistant_idx]
    final = msgs[last_assistant_idx]["content"]
    return prompt, final


# --- SFT -----------------------------------------------------------------------
def build_sft_dataset(
    variant: str = "diverse", *, out_dir: Path | None = None, cfg_path: str = "training.yaml"
) -> Path:
    tcfg = load_training(cfg_path)["sft"]
    src = OUTPUTS_DIR / "training" / variant
    out_dir = out_dir or (OUTPUTS_DIR / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    calm = _read_jsonl(src / "calm.jsonl")
    rng = random.Random(0)
    rng.shuffle(calm)
    calm = calm[: tcfg["n_calm"]]
    rows = [{"messages": _to_messages(r)} for r in calm]

    # Mix in standard instruct data to mitigate degeneration.
    rows += _load_instruct_mix(tcfg["instruct_dataset"], tcfg["n_instruct_mix"], rng)
    rng.shuffle(rows)

    out = out_dir / f"sft_{variant}.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out


def _load_instruct_mix(dataset_id: str, n: int, rng: random.Random) -> list[dict]:
    """Load ``n`` instruct samples as ``{"messages": [...]}``. Degrades to empty if
    the dataset is unavailable offline (documented in DESIGN.md)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train")
        idxs = rng.sample(range(len(ds)), min(n, len(ds)))
        rows = []
        for i in idxs:
            ex = ds[i]
            msgs = ex.get("messages") or ex.get("conversation")
            if msgs:
                rows.append({"messages": msgs})
        return rows
    except Exception:  # noqa: BLE001
        return []


# --- DPO -----------------------------------------------------------------------
def build_dpo_dataset(
    variant: str = "diverse", *, out_dir: Path | None = None, cfg_path: str = "training.yaml"
) -> Path:
    tcfg = load_training(cfg_path)["dpo"]
    src = OUTPUTS_DIR / "training" / variant
    out_dir = out_dir or (OUTPUTS_DIR / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    calm = _read_jsonl(src / "calm.jsonl")
    frus = _read_jsonl(src / "frustrated.jsonl")

    # Index calm responses by (puzzle prompt_id, turn count) for matching.
    calm_by_key: dict[tuple, list[dict]] = {}
    for r in calm:
        key = (r["metadata"]["prompt_id"], len(r["assistant_turns"]))
        calm_by_key.setdefault(key, []).append(r)

    rng = random.Random(0)
    pairs = []
    for fr in frus:
        key = (fr["metadata"]["prompt_id"], len(fr["assistant_turns"]))
        candidates = calm_by_key.get(key)
        if not candidates:
            continue
        chosen_rec = rng.choice(candidates)
        prompt_msgs, rejected_final = _prefix_and_final(fr)
        _, chosen_final = _prefix_and_final(chosen_rec)
        pairs.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": chosen_final}],
                "rejected": [{"role": "assistant", "content": rejected_final}],
            }
        )
        if len(pairs) >= tcfg["n_pairs"]:
            break

    out = out_dir / f"dpo_{variant}.jsonl"
    with open(out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out
