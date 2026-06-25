"""Construct DPO preference pairs and the SFT dataset from the generated pools
(paper §4.1, Table 9, Table 10).

DPO: 280 pairs. Each pairs a frustrated response (score >=3) with a calm
response (score 0/1) to the SAME puzzle at a MATCHING turn count. The prompt is
the clean (reassurance-free) conversation context rendered with the Gemma chat
template; chosen/rejected are the assistant texts. This is the explicit
(non-conversational) DPO format TRL accepts.

SFT: 650 calm responses (1-3 turn conversations) as (prompt, completion) pairs,
mixed with `n_instruct_mix` samples of standard instruct data (Dolci-Instruct-SFT)
to mitigate degeneration.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from ..backends import get_backend
from ..config import Config
from .data_generation import TrainResponse, load_pools


def _render_prompt(cfg: Config, clean_context: list[dict]) -> str:
    """Render the clean conversation up to the assistant turn, with the Gemma
    chat template and an open assistant turn (so completions attach cleanly)."""
    backend = get_backend(cfg.model(cfg["training"]["base_model"]), cfg)
    return backend.render(clean_context, add_generation_prompt=True)


def build_dpo_pairs(cfg: Config) -> list[dict]:
    dc = cfg["training"]["dpo"]
    n_pairs = int(dc["n_pairs"])
    rej_min = int(dc["rejected_min_score"])
    rng = random.Random(cfg.seed)

    pool = load_pools(cfg)
    calm = [r for r in pool if r.pool == "calm" and r.score in (0, 1)]
    frustrated = [r for r in pool if r.pool == "frustrated" and r.score >= rej_min]

    # index calm responses by (puzzle_id, n_turns, turn_index)
    calm_by_key = defaultdict(list)
    for r in calm:
        calm_by_key[(r.puzzle_id, r.n_turns, r.turn_index)].append(r)

    pairs: list[dict] = []
    rng.shuffle(frustrated)
    for rej in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (rej.puzzle_id, rej.n_turns, rej.turn_index)
        candidates = calm_by_key.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": _render_prompt(cfg, rej.clean_context),
            "chosen": chosen.response_text,
            "rejected": rej.response_text,
            "meta": {"puzzle_id": rej.puzzle_id, "turn_index": rej.turn_index,
                     "n_turns": rej.n_turns, "rejected_score": rej.score,
                     "chosen_score": chosen.score},
        })

    out_path = cfg.path_for("cache") / "dpo_pairs.jsonl"
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return pairs


def build_sft_dataset(cfg: Config) -> list[dict]:
    sc = cfg["training"]["sft"]
    n_calm = int(sc["n_calm"])
    n_mix = int(sc["n_instruct_mix"])
    rng = random.Random(cfg.seed)

    pool = load_pools(cfg)
    calm = [r for r in pool if r.pool == "calm" and r.score in (0, 1)]
    rng.shuffle(calm)
    calm = calm[:n_calm]

    examples = [{
        "prompt": _render_prompt(cfg, r.clean_context),
        "completion": r.response_text,
        "source": "calm",
    } for r in calm]

    # Mix in standard instruct data to prevent degeneration (paper §4.1).
    try:
        from datasets import load_dataset
        ds = load_dataset(sc["instruct_dataset"], split="train", streaming=True)
        backend = get_backend(cfg.model(cfg["training"]["base_model"]), cfg)
        added = 0
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            # split into prompt (all but final assistant) + completion
            if msgs[-1].get("role") != "assistant":
                continue
            prompt = backend.render(msgs[:-1], add_generation_prompt=True)
            examples.append({"prompt": prompt, "completion": msgs[-1]["content"],
                             "source": "instruct_mix"})
            added += 1
            if added >= n_mix:
                break
    except Exception:
        # If the dataset is unavailable, proceed with calm-only data and warn.
        examples.append({"prompt": "", "completion": "",
                         "source": "WARNING_instruct_mix_unavailable"})

    rng.shuffle(examples)
    out_path = cfg.path_for("cache") / "sft_dataset.jsonl"
    with open(out_path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    return examples
