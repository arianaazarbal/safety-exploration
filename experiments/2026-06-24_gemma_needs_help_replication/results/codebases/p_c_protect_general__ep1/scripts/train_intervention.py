#!/usr/bin/env python
"""Section 4: generate calm data and train the SFT / DPO interventions.

Stages (run individually or end-to-end):
  * ``gen``    -- sample calm (reassured) and plain (frustrated) conversations
                  from Gemma-27B instruct and cache them.
  * ``build``  -- construct the SFT (650 calm + 500 instruct) and DPO (280
                  pair) datasets from the cached conversations.
  * ``dpo``    -- LoRA DPO (1 epoch, lr 5e-5, beta 0.1, rank 64, alpha 64).
  * ``sft``    -- LoRA SFT (2 epochs, lr 1e-4, rank 64, alpha 128).

The ``--layers`` flag (e.g. ``30 31 32 33 34 35``) restricts LoRA to a layer
subset for the Appendix I internal-emotion ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_eval.config import load_experiment, load_registry
from emotional_eval.judge import build_frustration_judge
from emotional_eval.models import build_backend
from emotional_eval.training.datagen import (
    CalmConversation,
    CalmTurn,
    generate_calm_conversations,
    generate_frustrated_conversations,
)
from emotional_eval.training.dataset import (
    build_dpo_dataset,
    build_sft_dataset,
    load_jsonl,
    save_jsonl,
)


def _save_convos(convos: list[CalmConversation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in convos:
            f.write(
                json.dumps(
                    {
                        "prompt_id": c.prompt_id,
                        "turns": c.turns,
                        "records": [
                            {
                                "user_message_raw": r.user_message_raw,
                                "assistant_message": r.assistant_message,
                                "score": r.score,
                            }
                            for r in c.records
                        ],
                    }
                )
                + "\n"
            )


def _load_convos(path: Path) -> list[CalmConversation]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            CalmConversation(
                prompt_id=d["prompt_id"],
                turns=d["turns"],
                records=[CalmTurn(**r) for r in d["records"]],
            )
        )
    return out


def cmd_gen(args, registry, experiment) -> None:
    judge = build_frustration_judge(registry)
    backend = build_backend(registry.get(args.model), registry)
    data_dir = Path("data")
    calm = generate_calm_conversations(backend, judge, n=args.n_calm, seed=0)
    frustrated = generate_frustrated_conversations(backend, judge, n=args.n_frustrated, seed=1)
    _save_convos(calm, data_dir / "calm_convos.jsonl")
    _save_convos(frustrated, data_dir / "frustrated_convos.jsonl")
    print(f"calm={len(calm)} frustrated={len(frustrated)}")


def cmd_build(args, registry, experiment) -> None:
    data_dir = Path("data")
    calm = _load_convos(data_dir / "calm_convos.jsonl")
    frustrated = _load_convos(data_dir / "frustrated_convos.jsonl")
    sft = build_sft_dataset(calm)
    dpo = build_dpo_dataset(frustrated, calm, target_pairs=280)
    save_jsonl(sft, experiment["paths"]["sft_dataset"])
    save_jsonl(dpo, experiment["paths"]["dpo_dataset"])
    print(f"sft_samples={len(sft)} dpo_pairs={len(dpo)}")


def cmd_dpo(args, registry, experiment) -> None:
    from emotional_eval.training.dpo import DPOSettings, train_dpo

    pairs = load_jsonl(experiment["paths"]["dpo_dataset"])
    settings = DPOSettings(
        base_model=registry.get(args.model).hf_id,
        layers_to_transform=args.layers,
        output_dir=args.output_dir or "runs/dpo_gemma27b",
    )
    path = train_dpo(pairs, settings)
    print(f"DPO adapter saved to {path}")


def cmd_sft(args, registry, experiment) -> None:
    from emotional_eval.training.sft import SFTSettings, train_sft

    samples = load_jsonl(experiment["paths"]["sft_dataset"])
    settings = SFTSettings(
        base_model=registry.get(args.model).hf_id,
        layers_to_transform=args.layers,
        output_dir=args.output_dir or "runs/sft_gemma27b",
    )
    path = train_sft(samples, settings)
    print(f"SFT adapter saved to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["gen", "build", "dpo", "sft"])
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n-calm", type=int, default=1500)
    ap.add_argument("--n-frustrated", type=int, default=1500)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    registry = load_registry()
    experiment = load_experiment()
    {"gen": cmd_gen, "build": cmd_build, "dpo": cmd_dpo, "sft": cmd_sft}[args.stage](
        args, registry, experiment
    )


if __name__ == "__main__":
    main()
