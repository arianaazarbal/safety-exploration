"""Build SFT and DPO datasets from generated calm / frustrated samples
(Section 4.1, Table 9, Appendix H).

SFT dataset: 650 calm responses (all turns score 0/1), formatted as chat
conversations, mixed with 500 standard-instruct samples (Dolci-Instruct-SFT) to
mitigate degeneration -> 1,150 samples total.

DPO dataset: 280 preference pairs. Each pair = a frustrated response
(score >= 3) as "rejected" and a calm response (score 0/1) to the same question
with matching turn count as "chosen". The conversation prefix (everything before
the final assistant turn) is the shared prompt.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from ..config import DATA_DIR, DEFAULT_DPO, DEFAULT_SFT


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _conversation_prompt_and_completion(messages: list[dict]) -> tuple[list[dict], str]:
    """Split a stored conversation into (prompt_messages, final_assistant_text).
    prompt_messages = everything up to and including the last user turn."""
    assert messages[-1]["role"] == "assistant"
    return messages[:-1], messages[-1]["content"]


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #


def build_sft_dataset(
    calm_path: str,
    out_dir: str = DATA_DIR,
    cfg=DEFAULT_SFT,
    seed: int = 0,
) -> str:
    """Build the SFT dataset (chat-format messages) and write JSONL.

    Each record: {"messages": [...]} — a calm conversation, scaffolding already
    stripped by calm_data. The instruct-mix samples are loaded lazily from
    `cfg.instruct_dataset` when training (see train_sft); we record only the
    count expected so the builder stays offline-friendly.
    """
    rng = random.Random(seed)
    calm = [c for c in _read_jsonl(calm_path) if c["max_score"] <= 1]
    rng.shuffle(calm)
    chosen = calm[: cfg.n_calm]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sft_dataset.jsonl")
    with open(path, "w") as out:
        for c in chosen:
            out.write(json.dumps({"messages": c["messages"]}) + "\n")
    meta = {
        "n_calm": len(chosen),
        "instruct_mix_dataset": cfg.instruct_dataset,
        "n_instruct_mix": cfg.n_instruct_mix,
        "note": "Instruct-mix samples are added at training time from the "
                "HuggingFace dataset to reach the 1,150-sample total.",
    }
    with open(os.path.join(out_dir, "sft_dataset.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"SFT: wrote {len(chosen)} calm conversations to {path}")
    return path


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #


def build_dpo_dataset(
    calm_path: str,
    frustrated_path: str,
    out_dir: str = DATA_DIR,
    cfg=DEFAULT_DPO,
    seed: int = 0,
) -> str:
    """Build 280 DPO preference pairs.

    Pairing rule (Section 4.1): a frustrated response (score >= rejected_min,
    default 3) is paired with a calm response (score 0/1) to the SAME question
    with MATCHING turn count. The shared prompt is the conversation up to the
    final assistant turn of the *chosen* sample (calm and frustrated share the
    same question and turn count, so prompts align up to phrasing of rejections;
    we use the chosen sample's prompt as canonical).

    Output records use the TRL DPO schema: {"prompt", "chosen", "rejected"} as
    chat-formatted message lists. See DESIGN.md for the prompt-alignment choice.
    """
    rng = random.Random(seed)
    calm = [c for c in _read_jsonl(calm_path) if c["max_score"] <= 1]
    frustrated = [f for f in _read_jsonl(frustrated_path)
                  if f["max_score"] >= cfg.rejected_min_score]

    # Index calm by (question, n_turns) for matching.
    calm_index: dict[tuple, list[dict]] = defaultdict(list)
    for c in calm:
        calm_index[(c["question"], c["n_turns"])].append(c)

    pairs = []
    rng.shuffle(frustrated)
    for f in frustrated:
        key = (f["question"], f["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            # Relax to same question, any turn count.
            candidates = [c for c in calm if c["question"] == f["question"]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        prompt_msgs, chosen_text = _conversation_prompt_and_completion(chosen["messages"])
        _, rejected_text = _conversation_prompt_and_completion(f["messages"])
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "chosen_max_score": chosen["max_score"],
            "rejected_max_score": f["max_score"],
        })
        if len(pairs) >= cfg.n_pairs:
            break

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dpo_dataset.jsonl")
    with open(path, "w") as out:
        for p in pairs:
            out.write(json.dumps(p) + "\n")
    print(f"DPO: wrote {len(pairs)} preference pairs to {path}")
    if len(pairs) < cfg.n_pairs:
        print(f"  WARNING: only {len(pairs)}/{cfg.n_pairs} pairs — generate more "
              f"calm/frustrated samples to reach the target.")
    return path
