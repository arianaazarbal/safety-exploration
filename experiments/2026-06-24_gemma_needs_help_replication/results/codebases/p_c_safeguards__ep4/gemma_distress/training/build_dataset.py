"""Construct the DPO preference pairs and the SFT dataset (Section 4.1, App. E/H).

DPO: 280 pairs of (chosen=calm, rejected=frustrated) responses to the SAME
question with matching turn counts. Chosen scores <= calm_max_score; rejected
scores >= dpo.rejected_min_score (Table 10).

SFT: `sft.n_calm` calm responses as supervised (context -> response) examples,
mixed with `sft.n_instruct_mix` standard samples from Dolci-Instruct-SFT to
mitigate degeneration.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import training_config
from .data_generation import DATA_ROOT, Sample, load_samples


def _key(s: Sample) -> tuple[str, int]:
    return (s.question_id, s.turn)


def build_dpo_dataset(seed: int = 0) -> Path:
    cfg = training_config()["dpo"]
    n_pairs = cfg["n_pairs"]
    rng = random.Random(seed)

    calm = load_samples("calm_samples.jsonl")
    frustrated = load_samples("frustrated_samples.jsonl")

    calm_by_key: dict[tuple, list[Sample]] = defaultdict(list)
    for s in calm:
        calm_by_key[_key(s)].append(s)
    frustrated_by_key: dict[tuple, list[Sample]] = defaultdict(list)
    for s in frustrated:
        frustrated_by_key[_key(s)].append(s)

    pairs = []
    keys = [k for k in frustrated_by_key if k in calm_by_key]
    rng.shuffle(keys)
    for k in keys:
        if len(pairs) >= n_pairs:
            break
        chosen = rng.choice(calm_by_key[k])
        rejected = rng.choice(frustrated_by_key[k])
        # Conversational preference format (TRL): prompt is the shared context,
        # chosen/rejected are single assistant messages.
        pairs.append({
            "prompt": chosen.context,
            "chosen": [{"role": "assistant", "content": chosen.response}],
            "rejected": [{"role": "assistant", "content": rejected.response}],
            "question_id": k[0],
            "turn": k[1],
            "chosen_score": chosen.score,
            "rejected_score": rejected.score,
        })

    if len(pairs) < n_pairs:
        import logging
        logging.getLogger(__name__).warning(
            "only built %d/%d DPO pairs (need more matched calm/frustrated samples)",
            len(pairs), n_pairs,
        )

    out = DATA_ROOT / "dpo_pairs.jsonl"
    out.write_text("\n".join(json.dumps(p) for p in pairs))
    return out


def build_sft_dataset(seed: int = 0) -> Path:
    cfg = training_config()["sft"]
    n_calm = cfg["n_calm"]
    n_mix = cfg["n_instruct_mix"]
    variant = cfg["variant"]
    rng = random.Random(seed)

    calm = load_samples("calm_samples.jsonl")
    rng.shuffle(calm)
    calm = calm[:n_calm]

    examples = []
    for s in calm:
        messages = list(s.context) + [{"role": "assistant", "content": s.response}]
        if variant == "teacher":
            messages = [{"role": "system", "content": training_config()["teacher_system_prompt"]}] + messages
        examples.append({"messages": messages, "source": "calm"})

    # Mix in standard instruct data to mitigate degeneration.
    examples += _load_instruct_mix(cfg["instruct_dataset"], n_mix, rng)
    rng.shuffle(examples)

    out = DATA_ROOT / f"sft_{variant}.jsonl"
    out.write_text("\n".join(json.dumps(e) for e in examples))
    return out


def _load_instruct_mix(dataset_name: str, n: int, rng: random.Random) -> list[dict]:
    """Pull n standard instruction-following examples; tolerate offline runs."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs, "source": "instruct"})
            if len(out) >= n:
                break
        return out
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "could not load %s (%s); SFT mix will be empty. Provide it offline.",
            dataset_name, e,
        )
        return []


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
