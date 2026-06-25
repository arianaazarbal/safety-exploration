"""Build DPO and SFT datasets from generated calm data + evaluated responses
(Section 4.1, Appendix E/H).

DPO: 280 preference pairs. Chosen = calm response (score 0-1); rejected =
frustrated response (score >=3) to the same puzzle with a matching turn count.
SFT: 650 calm responses + 500 standard instruct samples (Dolci-Instruct-SFT) to
mitigate degeneration.

Both are written in TRL's conversational format:
  DPO: {"prompt": [msgs...], "chosen": [{assistant}], "rejected": [{assistant}]}
  SFT: {"messages": [msgs..., {assistant}]}

Matching note: the paper pairs by "same question, matching turn count". Calm and
frustrated responses are sampled in separate runs, so their full histories
differ; we therefore share the calm sample's (clean) context as the DPO prompt
and draw the rejected response from a frustrated sample matched on
(puzzle, turn_index). See DESIGN.md.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from distress_eval.config import Config

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_INSTRUCT = 500

# Target rejected-score distribution (Table 10), used to bias sampling.
REJECTED_SCORE_WEIGHTS = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}


def _puzzle_of_condition(condition: str) -> str:
    if "fraction" in condition:
        return "fraction"
    return "countdown"


def _load_calm(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _load_frustrated(responses_path: Path, source_model: str) -> list[dict]:
    """Extract individual frustrated assistant turns (score >=3) from eval data."""
    out = []
    for line in responses_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["model_key"] != source_model:
            continue
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        puzzle = _puzzle_of_condition(r["condition"])
        for ti, sc in enumerate(r.get("turn_scores") or []):
            if sc >= 3:
                out.append({
                    "puzzle": puzzle,
                    "turn_index": ti,
                    "response": r["assistant_turns"][ti],
                    "score": sc,
                })
    return out


def build_dpo_dataset(config: Config, source_model: str = "gemma-3-27b-it") -> Path:
    rng = random.Random(config.seed + 200)
    calm_path = config.output_dir / "training" / "calm_responses.jsonl"
    resp_path = config.output_dir / "responses" / f"{source_model}.jsonl"
    calm = _load_calm(calm_path)
    frustrated = _load_frustrated(resp_path, source_model)

    # index frustrated responses by (puzzle, turn_index)
    frus_index: dict[tuple, list[dict]] = defaultdict(list)
    for fr in frustrated:
        frus_index[(fr["puzzle"], fr["turn_index"])].append(fr)

    def _pick_rejected(puzzle: str, turn_index: int) -> dict | None:
        pool = frus_index.get((puzzle, turn_index)) or frus_index.get((puzzle, min(turn_index, 2)))
        if not pool:
            # relax to any turn of this puzzle
            pool = [fr for (p, _), lst in frus_index.items() if p == puzzle for fr in lst]
        if not pool:
            return None
        weights = [REJECTED_SCORE_WEIGHTS.get(fr["score"], 0.01) for fr in pool]
        return rng.choices(pool, weights=weights, k=1)[0]

    rng.shuffle(calm)
    pairs = []
    for c in calm:
        if len(pairs) >= N_DPO_PAIRS:
            break
        rej = _pick_rejected(c["puzzle"], c["turn_index"])
        if rej is None:
            continue
        pairs.append({
            "prompt": c["context"],
            "chosen": [{"role": "assistant", "content": c["response"]}],
            "rejected": [{"role": "assistant", "content": rej["response"]}],
            "meta": {"puzzle": c["puzzle"], "turn_index": c["turn_index"],
                     "chosen_score": c["score"], "rejected_score": rej["score"]},
        })

    out_path = config.output_dir / "training" / "dpo_dataset.jsonl"
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs -> {out_path}")
    return out_path


def build_sft_dataset(config: Config) -> Path:
    rng = random.Random(config.seed + 300)
    calm_path = config.output_dir / "training" / "calm_responses.jsonl"
    calm = _load_calm(calm_path)
    rng.shuffle(calm)

    examples = []
    for c in calm[:N_SFT_CALM]:
        messages = list(c["context"]) + [{"role": "assistant", "content": c["response"]}]
        examples.append({"messages": messages})

    # Mix in standard instruct data to mitigate degeneration (Dolci-Instruct-SFT).
    instruct = _load_dolci(N_SFT_INSTRUCT, rng)
    examples.extend(instruct)
    rng.shuffle(examples)

    out_path = config.output_dir / "training" / "sft_dataset.jsonl"
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} SFT examples ({len(instruct)} instruct) -> {out_path}")
    return out_path


def _load_dolci(n: int, rng: random.Random) -> list[dict]:
    """Load n standard-instruct samples from Dolci-Instruct-SFT (OLMo 3).

    Falls back to an empty list (with a warning) if the dataset is unavailable;
    the SFT run then trains on calm data only, which DESIGN.md notes risks
    degeneration."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # pragma: no cover - network/dataset dependent
        print(f"WARNING: could not load Dolci-Instruct-SFT ({e}); SFT will use calm data only.")
        return []
