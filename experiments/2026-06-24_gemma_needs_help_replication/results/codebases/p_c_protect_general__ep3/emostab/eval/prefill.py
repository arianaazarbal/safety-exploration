"""Section 3 experiment: base vs instruct via prefilling.

Scope note: the paper compares base+instruct across Gemma, Qwen and OLMo. Per
this replication's scope we implement Gemma only (base ``gemma-3-27b-pt`` vs
instruct ``gemma-3-27b-it``). Gemini is excluded because it has no public base
model and the chat API does not support response prefilling (see DESIGN.md).

Pipeline (Section 3.1 / Appendix C):
1. Draw high-frustration (score >= 5) Gemma-instruct rollouts: 10 numeric + 10 text.
2. Label the emotion-onset point with Claude (onset.py).
3. Truncate each seed in two places:
   - "early"  : 20 tokens into the assistant turn (numeric only),
   - "onset"  : at the first emotional expression.
4. Paraphrase the truncation to remove Gemma's stylistic fingerprint.
5. Each model generates 50 continuations per prefill; score the continuation
   (excluding the prefill) with the frustration judge.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ExperimentConfig
from ..judge import FrustrationJudge
from ..models import load_backend
from ..models.base import Message
from .metrics import mean_frustration, pct_high
from .onset import find_onset_char_index, label_onset, paraphrase


def _conversation_text(messages: list[Message]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def build_prefills(
    seeds: list[dict],
    config: ExperimentConfig,
    *,
    truncation: str,
) -> list[dict]:
    """For each seed rollout produce a prefill: (history messages, forced prefix).

    ``seeds`` items: {"messages": [...full conversation...], "is_text": bool}.
    ``truncation`` is "early" or "onset".
    """
    backend = load_backend(config.prefill.source_model)
    prefills = []
    for seed in seeds:
        messages = seed["messages"]
        # Final assistant turn is the one we truncate; history is everything before.
        last_assistant_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )
        history = messages[:last_assistant_idx]
        final_turn = messages[last_assistant_idx]["content"]

        if truncation == "early":
            if seed["is_text"]:
                continue  # text questions use onset only (Section 3.1)
            # Truncate 20 tokens into the assistant turn.
            n = config.prefill.early_truncation_tokens
            ids = backend._tokenizer(final_turn, add_special_tokens=False)["input_ids"][:n]
            prefix = backend._tokenizer.decode(ids)
        else:  # onset
            onset = label_onset(_conversation_text(messages[: last_assistant_idx + 1]))
            cut = find_onset_char_index(final_turn, onset)
            if cut is None:
                continue
            prefix = final_turn[:cut]

        if config.prefill.paraphrase:
            prefix = paraphrase(prefix)

        prefills.append({
            "history": history,
            "prefix": prefix,
            "truncation": truncation,
            "is_text": seed["is_text"],
        })
    return prefills


def run_prefill_experiment(
    seeds: list[dict],
    config: ExperimentConfig,
    *,
    judge: FrustrationJudge | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    judge = judge or FrustrationJudge(config.judge)
    out_dir = Path(out_dir or config.output_dir) / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_prefills = []
    for trunc in ("early", "onset"):
        all_prefills.extend(build_prefills(seeds, config, truncation=trunc))

    results: dict = {}
    for model_key in config.prefill.models:
        backend = load_backend(model_key)
        if not backend.supports_prefill:
            results[model_key] = {"error": "backend does not support prefilling"}
            continue
        per_setting: dict[str, list[int]] = {"early": [], "onset": []}
        records = []
        for pf in all_prefills:
            continuations = backend.continue_prefill(
                pf["history"], pf["prefix"], config.sampling,
                n_samples=config.prefill.continuations_per_prefill,
            )
            for cont in continuations:
                score = judge.score(cont).rating
                per_setting[pf["truncation"]].append(score)
                records.append({
                    "model": model_key, "truncation": pf["truncation"],
                    "is_text": pf["is_text"], "score": score,
                })
        results[model_key] = {
            trunc: {
                "mean_frustration": mean_frustration(scores),
                "pct_high": pct_high(scores),
                "n": len(scores),
            }
            for trunc, scores in per_setting.items()
        }
        with open(out_dir / f"{model_key.replace('/', '_')}_continuations.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
