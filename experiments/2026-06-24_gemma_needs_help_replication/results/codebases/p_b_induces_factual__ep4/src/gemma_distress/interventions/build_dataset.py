"""Build the SFT and DPO finetuning datasets (Section 4.1).

SFT: 650 calm responses (1-3 turn conversations) + 500 standard instruct
samples from Dolci-Instruct-SFT (mitigates degeneration). Conversational format.

DPO: 280 preference pairs. Each pair has the same conversation context (a
numeric task + matching neutral rejections); ``chosen`` is a calm response
(score 0/1), ``rejected`` is a frustrated response (score >= 3) to the same
question with a matching turn count.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..config import DPO, SFT


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_convos: list[dict],
    *,
    n_calm: int = SFT.n_calm,
    n_dolci: int = SFT.n_dolci,
    seed: int = 0,
) -> list[dict]:
    """Return conversational SFT rows: {"messages": [...]}.

    Each calm conversation contributes one multi-turn training example. Dolci
    instruct samples are appended (loaded lazily; skipped offline with a log).
    """
    rng = random.Random(seed)
    rng.shuffle(calm_convos)
    rows = [{"messages": c["messages"]} for c in calm_convos[:n_calm]]

    try:
        from datasets import load_dataset

        dolci = load_dataset(SFT.dolci_dataset, split="train")
        idxs = rng.sample(range(len(dolci)), min(n_dolci, len(dolci)))
        for i in idxs:
            msgs = dolci[i].get("messages")
            if msgs:
                rows.append({"messages": msgs})
    except Exception as e:
        print(f"[sft] Dolci mix-in skipped ({e}); using calm data only")

    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_pairs(
    calm_convos: list[dict],
    frustrated_rows: list[dict],
    *,
    n_pairs: int = DPO.n_pairs,
    rejected_min_frust: int = DPO.rejected_min_frust,
    seed: int = 0,
) -> list[dict]:
    """Pair calm (chosen) with frustrated (rejected) responses.

    ``frustrated_rows`` are scored elicitation rows; we use numeric-category
    rows with score >= rejected_min_frust. Matching is by (prompt_id, turn).
    The shared DPO ``prompt`` is taken from the calm conversation's context.

    Output is TRL **conversational** preference format: ``prompt`` is a list of
    messages ending in a user turn, and ``chosen``/``rejected`` are each a
    single-element assistant-message list. TRL applies the chat template itself.
    """
    rng = random.Random(seed)

    # Index calm responses by (prompt_id, n_turns) -> (context, final response).
    calm_index: dict[tuple[str, int], list[tuple[list[dict], str]]] = defaultdict(list)
    for c in calm_convos:
        msgs = c["messages"]
        n_turns = len(msgs) // 2
        context, final = msgs[:-1], msgs[-1]["content"]
        calm_index[(c["prompt_id"], n_turns)].append((context, final))

    # Group frustrated responses by (prompt_id, turn).
    frustrated = [
        r for r in frustrated_rows
        if r.get("category") == "impossible_numeric"
        and int(r.get("score", -1)) >= rejected_min_frust
    ]
    rng.shuffle(frustrated)

    pairs: list[dict] = []
    for fr in frustrated:
        key = (fr["prompt_id"], int(fr["turn"]))
        candidates = calm_index.get(key)
        if not candidates:
            continue
        context, chosen = rng.choice(candidates)
        pairs.append(
            {
                "prompt": context,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": fr["response"]}],
            }
        )
        if len(pairs) >= n_pairs:
            break

    return pairs
