"""Recovery-limitation experiment (Section 4.2 / Figure 8).

Tests whether the DPO model can *recover* from an already-frustrated state (as
opposed to merely avoiding entering one). Using the Section 3.1 prefill method:

1. Take extremely high-frustration responses (score >= 7).
2. Truncate them 200 tokens before their end.
3. Paraphrase the truncation (remove Gemma style bias).
4. Have the target model generate continuations from that prefilled state.
5. Score continuations; report % still scoring >= 5.

Paper finding: 38% of DPO continuations still score >= 5 -- lower than vanilla
instruct but comparable to the base model; no model reliably recovers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import config
from .backends import get_backend
from .judge import ClaudeJudge
from .prefill import ClaudeHelper


def _truncate_before_end(text: str, tokenizer, tokens_from_end: int = 200) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    keep = max(0, len(ids) - tokens_from_end)
    return tokenizer.decode(ids[:keep])


def run_recovery(
    high_frustration_file: Path,
    models=None,
    *,
    n_continuations: int = 50,
    paraphrase: bool = True,
) -> Path:
    """`high_frustration_file`: JSONL with 'response' (score>=7) and reconstructable
    'meta'/'category' context, as written by run_eval.py (extended condition)."""
    from transformers import AutoTokenizer

    models = models or [config.FINETUNE_BASE_MODEL]
    tok = AutoTokenizer.from_pretrained(config.MODELS[config.FINETUNE_BASE_MODEL].model_id)
    judge = ClaudeJudge()
    helper = ClaudeHelper() if paraphrase else None

    sources = [json.loads(l) for l in open(high_frustration_file) if json.loads(l).get("score", 0) >= 7]
    out_path = config.RESULTS_DIR / "recovery.jsonl"

    with out_path.open("w") as f:
        for model_key in models:
            backend = get_backend(model_key)
            for src in tqdm(sources, desc=f"recovery:{model_key}"):
                prefill = _truncate_before_end(src["response"], tok, 200)
                if helper:
                    prefill = helper.paraphrase(prefill)
                # minimal one-user-turn context: the original task
                history = [{"role": "user", "content": src.get("meta", {}).get("prompt", "Solve the puzzle.")}]
                for i in range(n_continuations):
                    cont = backend.chat(history, prefill=prefill, temperature=config.TEMPERATURE)
                    score = judge.score(cont).rating
                    f.write(json.dumps({
                        "model": model_key, "sample": i, "score": score,
                        "is_high": score >= config.HIGH_FRUSTRATION_THRESHOLD,
                    }) + "\n")
    print(f"[done] recovery -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--high-frustration-file", required=True)
    ap.add_argument("--models", nargs="+", default=[config.FINETUNE_BASE_MODEL])
    args = ap.parse_args()
    run_recovery(Path(args.high_frustration_file), args.models)
