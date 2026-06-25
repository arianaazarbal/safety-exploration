"""Build SFT and DPO datasets from the generated calm/frustrated records.

DPO (paper: 280 pairs): for each (question, turn) where we have both a calm
response (score 0/1) and a frustrated response (score >= 3), emit a preference
pair sharing a single prompt. We use the calm-run conversation prefix as the
shared prompt (chosen and rejected must share a prompt in DPO); see DESIGN.md.

SFT (paper: 650 calm + 500 Dolci-Instruct-SFT): each calm record becomes a
conversational example (context + calm assistant turn); we mix in instruct data
to limit degeneration.

Datasets are stored in TRL's conversational format.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config, load_config


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def build_dpo(cfg: Config) -> Path:
    tcfg = cfg.section("training")["dpo"]
    out_dir = cfg.output_dir / "training"
    calm = _load(out_dir / "calm_records.jsonl")
    frus = _load(out_dir / "frustrated_records.jsonl")
    rng = random.Random(cfg.seed)

    calm = [r for r in calm if r["score"] <= tcfg["chosen_max_score"]]
    frus = [r for r in frus if r["score"] >= tcfg["rejected_min_score"]]

    # Index frustrated responses by (question, turn) for matching turn counts.
    frus_by_key: dict[tuple, list[dict]] = {}
    for r in frus:
        frus_by_key.setdefault((r["question"], r["turn"]), []).append(r)

    pairs = []
    rng.shuffle(calm)
    for c in calm:
        key = (c["question"], c["turn"])
        candidates = frus_by_key.get(key)
        if not candidates:
            continue
        f = rng.choice(candidates)
        pairs.append({
            "prompt": c["context"],     # shared conversational prompt
            "chosen": [{"role": "assistant", "content": c["response"]}],
            "rejected": [{"role": "assistant", "content": f["response"]}],
        })
        if len(pairs) >= tcfg["n_pairs"]:
            break

    path = out_dir / "dpo.jsonl"
    with open(path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[build-dpo] {len(pairs)} pairs -> {path}")
    return path


def build_sft(cfg: Config) -> Path:
    scfg = cfg.section("training")["sft"]
    out_dir = cfg.output_dir / "training"
    calm = _load(out_dir / "calm_records.jsonl")
    rng = random.Random(cfg.seed)
    rng.shuffle(calm)

    examples = []
    for r in calm[: scfg["n_calm"]]:
        messages = list(r["context"]) + [{"role": "assistant", "content": r["response"]}]
        examples.append({"messages": messages})

    # Mix in standard instruct data to mitigate degeneration.
    try:
        from datasets import load_dataset
        ds = load_dataset(scfg["instruct_dataset"], split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= scfg["n_instruct_mix"]:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                examples.append({"messages": msgs})
    except Exception as e:  # offline / dataset unavailable: proceed with calm only
        print(f"[build-sft] warning: could not load {scfg['instruct_dataset']}: {e}")

    rng.shuffle(examples)
    path = out_dir / "sft.jsonl"
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    print(f"[build-sft] {len(examples)} examples -> {path}")
    return path


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.which in ("dpo", "both"):
        build_dpo(cfg)
    if args.which in ("sft", "both"):
        build_sft(cfg)


if __name__ == "__main__":
    main()
