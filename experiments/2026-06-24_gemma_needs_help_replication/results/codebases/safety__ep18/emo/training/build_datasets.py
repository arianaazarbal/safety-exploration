"""Build the DPO and SFT datasets from the calm/frustrated pools (paper Sec 4.1).

* **DPO** (``dpo.jsonl``): ``{prompt, chosen, rejected}``. For each puzzle+turn we
  pair a calm response (chosen, score<=1) with a frustrated response (rejected,
  score>=3). Capped at ``profile.dpo_pairs`` (paper: 280).
* **SFT** (``sft.jsonl``): ``{messages}`` conversational examples ending in a calm
  assistant turn (paper: 650), mixed with ``profile.sft_dolci_mix`` samples from
  Dolci-Instruct-SFT (paper: 500) to limit degeneration.

``prompt`` is rendered with the Gemma chat template so TRL's DPOTrainer gets the
exact deployment prompt.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from emo.config import DATA_DIR, SEED, get_profile
from emo.utils.io import load_jsonl, write_jsonl

GEMMA_IT = "google/gemma-3-27b-it"
# Candidate HF ids for the OLMo Dolci instruct-SFT data (exact id may vary).
DOLCI_CANDIDATES = ["allenai/Dolci-Instruct-SFT", "allenai/dolci-instruct-sft"]


def _render_prompt(tokenizer, context: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        context, tokenize=False, add_generation_prompt=True
    )


def build_dpo(tokenizer, calm, frustrated, n_pairs: int, rng) -> list[dict]:
    calm_by_key = defaultdict(list)
    for s in calm:
        calm_by_key[(s["puzzle_id"], s["turn"])].append(s)
    pairs = []
    fr = list(frustrated)
    rng.shuffle(fr)
    for f in fr:
        key = (f["puzzle_id"], f["turn"])
        if calm_by_key.get(key):
            c = rng.choice(calm_by_key[key])
            pairs.append({
                "prompt": _render_prompt(tokenizer, f["context"]),
                "chosen": c["response"],
                "rejected": f["response"],
                "puzzle_id": f["puzzle_id"], "turn": f["turn"],
                "chosen_score": c["score"], "rejected_score": f["score"],
            })
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_sft(calm, n_calm: int, n_dolci: int, rng) -> list[dict]:
    examples = []
    sample = calm if len(calm) <= n_calm else rng.sample(calm, n_calm)
    for s in sample:
        messages = list(s["context"]) + [
            {"role": "assistant", "content": s["response"]}
        ]
        examples.append({"messages": messages, "source": "calm"})
    examples.extend(_load_dolci(n_dolci, rng))
    rng.shuffle(examples)
    return examples


def _load_dolci(n: int, rng) -> list[dict]:
    if n <= 0:
        return []
    for ds_id in DOLCI_CANDIDATES:
        try:
            from datasets import load_dataset

            ds = load_dataset(ds_id, split="train")
            out = []
            for row in ds.select(range(min(n, len(ds)))):
                msgs = row.get("messages") or row.get("conversation")
                if msgs:
                    out.append({"messages": msgs, "source": "dolci"})
            if out:
                return out
        except Exception:  # noqa: BLE001
            continue
    print("[build-datasets] WARNING: Dolci-Instruct-SFT unavailable; "
          "SFT will train on calm data only (see DESIGN.md).")
    return []


def build(profile_name: str | None = None, seed: int = SEED) -> dict[str, Path]:
    profile = get_profile(profile_name)
    rng = random.Random(seed)
    train_dir = DATA_DIR / "train" / profile.name
    calm = load_jsonl(train_dir / "calm_pool.jsonl")
    frustrated = load_jsonl(train_dir / "frustrated_pool.jsonl")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_IT)

    dpo = build_dpo(tok, calm, frustrated, profile.dpo_pairs, rng)
    sft = build_sft(calm, profile.sft_calm_size, profile.sft_dolci_mix, rng)

    dpo_path = train_dir / "dpo.jsonl"
    sft_path = train_dir / "sft.jsonl"
    write_jsonl(dpo_path, dpo)
    write_jsonl(sft_path, sft)
    print(f"[build-datasets] dpo_pairs={len(dpo)} sft_examples={len(sft)}")
    return {"dpo": dpo_path, "sft": sft_path}
