"""Recovery-from-spiral test (Section 4.2, Figure 8).

While DPO prevents frustration spirals, the paper shows it doesn't help models
*recover* from one. Using the Section 3 prefill method, we take extremely
high-frustration responses (score >=7), truncate 200 tokens before their end,
paraphrase, and measure how the continuations score. The paper reports 38% of
DPO-model continuations still score >=5 (comparable to the base model).

Usage (models and adapters are positional-parallel; use 'none' for no adapter,
so the DPO model is just another (model, adapter) entry):
    python -m src.prefill.recovery_test \
        --seeds data/section2_gemma-3-27b-it.jsonl \
        --models gemma-3-27b-it gemma-3-27b-pt gemma-3-27b-it \
        --adapters none         none          checkpoints/dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from ..models import load_model
from ..models.base import GenerationParams
from ..judge.frustration_judge import FrustrationJudge
from .onset_label import Paraphraser


def pick_recovery_seeds(records: list[dict]) -> list[dict]:
    """Per-turn responses scoring >= recovery_min_score, longest first."""
    hot = [r for r in records if r["rating"] >= config.PREFILL.recovery_min_score]
    hot.sort(key=lambda r: len(r["response"]), reverse=True)
    return hot


def run(seeds_path: str, targets: list[tuple[str, str | None]],
        out_path: str | None = None) -> Path:
    records = [json.loads(l) for l in Path(seeds_path).read_text().splitlines() if l.strip()]
    seeds = pick_recovery_seeds(records)

    tok_model = load_model(config.GEMMA_27B_IT)  # tokenizer for token-accurate truncation
    paraphraser = Paraphraser()
    judge = FrustrationJudge()
    gen = GenerationParams()

    # Build prefills: truncate 200 tokens before the end of the hot response.
    specs = []
    for s in seeds:
        text = s["response"]
        n_tok = tok_model.count_tokens(text)
        keep = max(config.PREFILL.early_truncate_tokens,
                   n_tok - config.PREFILL.recovery_truncate_before_end)
        prefix = tok_model.truncate_tokens(text, keep)
        para = paraphraser.paraphrase(prefix)
        # Reconstruct the conversation context that preceded this turn within the
        # same conv_id is not available per-record here; we use the single hot
        # turn's user message as the immediate context.
        specs.append({
            "messages": [{"role": "user", "content": s.get("user", s["prompt"])}],
            "prefill": para,
            "seed_score": s["rating"],
        })

    out_path = Path(out_path or config.DATA_DIR / "section4_recovery.jsonl")
    with out_path.open("w") as fh:
        for mkey, adapter in targets:
            adapter = None if adapter in (None, "none", "") else adapter
            spec = config.ALL_MODELS.get(mkey) or next(
                m for m in config.PREFILL_TARGETS + config.ELICITATION_TARGETS if m.key == mkey)
            label = spec.key + (f"+{Path(adapter).name}" if adapter else "")
            model = (tok_model if (spec.key == config.GEMMA_27B_IT.key and not adapter)
                     else load_model(spec, adapter_path=adapter))
            for si, ps in enumerate(specs):
                batch = [ps["messages"]] * config.PREFILL.continuations_per_prefill
                prefills = [ps["prefill"]] * config.PREFILL.continuations_per_prefill
                conts = model.generate_batch(batch, gen, prefills)
                scores = judge.score_batch(conts)
                for ci, (cont, js) in enumerate(zip(conts, scores)):
                    fh.write(json.dumps({
                        "model": label, "prefill_id": si, "continuation_index": ci,
                        "seed_score": ps["seed_score"], "rating": js.rating,
                        "continuation": cont,
                    }) + "\n")
            if model is not tok_model:
                model.close()
            print(f"[{label}] recovery: scored {len(specs) * config.PREFILL.continuations_per_prefill}")
    tok_model.close()
    print(f"Wrote -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--adapters", nargs="+", default=None)
    args = ap.parse_args()
    adapters = args.adapters or ["none"] * len(args.models)
    run(args.seeds, list(zip(args.models, adapters)))


if __name__ == "__main__":
    main()
