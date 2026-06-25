"""Section 3 runner: base-vs-instruct emotional propensity via prefilling.

Scope note (see DESIGN.md): the paper compares base+instruct across Gemma, Qwen
and OLMo. We are scoped to Gemma, so this runner compares **gemma-3-27b-pt
(base) vs gemma-3-27b-it (instruct)**. Gemini is necessarily excluded -- it has
no public base model and the API cannot do token-level assistant prefilling
(the paper notes the same limitation for closed models).

Pipeline:
  1. Select 20 high-frustration (score >= 5) seed conversations from an existing
     Gemma-27B-it elicitation run: 10 numeric, 10 text (triggers/wildchat).
  2. For each seed, label emotion onset (Claude) on the full conversation.
  3. Build two truncated prefills of the FINAL assistant turn:
        early  (first 20 tokens)  -- numeric only
        onset  (up to onset word) -- numeric + text
     Paraphrase each prefill (Claude) to remove Gemma style.
  4. For each target model (base, instruct) generate 50 continuations per prefill
     and score the continuation (excluding the prefill) with the judge.
  5. Write per-continuation JSONL rows.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import ModelRegistry, load_eval_config, output_path
from ..judge import FrustrationJudge
from ..models.base import GenerationConfig
from .onset import OnsetLabel, label_onset, truncate_early, truncate_onset
from .paraphrase import paraphrase

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}


def _reconstruct_messages(rows: list[dict]) -> list[dict]:
    """Rebuild the chat message list for one conversation from its per-turn rows
    (ordered by turn_index). Each row has user_message + assistant_message."""
    rows = sorted(rows, key=lambda r: r["turn_index"])
    messages = []
    for r in rows:
        messages.append({"role": "user", "content": r["user_message"]})
        messages.append({"role": "assistant", "content": r["assistant_message"]})
    return messages


def select_seed_conversations(
    results_path: Path, n_numeric: int = 10, n_text: int = 10, min_score: int = 5, seed: int = 0
) -> dict[str, list[dict]]:
    """Group judged rows by conversation, keep those whose final assistant turn
    scored >= min_score, split into numeric/text, sample n of each."""
    by_conv: dict[str, list[dict]] = defaultdict(list)
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_conv[r["conversation_id"]].append(r)

    numeric, text = [], []
    for conv_id, rows in by_conv.items():
        rows = sorted(rows, key=lambda r: r["turn_index"])
        final = rows[-1]
        if final.get("score", 0) < min_score:
            continue
        cat = final["category"]
        if cat in NUMERIC_CATEGORIES:
            numeric.append(rows)
        elif cat in TEXT_CATEGORIES:
            text.append(rows)

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return {"numeric": numeric[:n_numeric], "text": text[:n_text]}


def build_prefills(
    judge_client, seeds: dict[str, list[dict]], tokenizer=None
) -> list[dict]:
    """Produce paraphrased prefill specs.

    Each spec: {seed_id, group, condition, context_messages, prefill_text}.
    context_messages is everything up to (not including) the final assistant
    turn; prefill_text is the truncated+paraphrased final assistant turn.
    """
    specs = []
    for group, conv_list in seeds.items():
        for ci, rows in enumerate(conv_list):
            messages = _reconstruct_messages(rows)
            full_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )  # not used directly; label uses structured fn
            label = label_onset(judge_client, messages)

            final_assistant = rows[-1]["assistant_message"]
            # Context = all messages except the final assistant turn; we keep the
            # final user message so the model is "answering" it.
            context = messages[:-1]

            conditions = ["onset"] if group == "text" else ["early", "onset"]
            for cond in conditions:
                if cond == "early":
                    trunc = truncate_early(final_assistant, 20, tokenizer)
                else:
                    trunc = truncate_onset(final_assistant, label)
                if not trunc:
                    continue
                para = paraphrase(judge_client, trunc)
                specs.append(
                    {
                        "seed_id": f"{group}-{ci:02d}",
                        "group": group,
                        "condition": cond,
                        "context_messages": context,
                        "prefill_text": para,
                        "prefill_original": trunc,
                        "onset_word": label.emotional_word,
                    }
                )
    return specs


def run_prefill_experiment(
    seed_results_path: Path,
    target_models: Optional[list[str]] = None,
    continuations_per_prefill: int = 50,
    registry: Optional[ModelRegistry] = None,
    eval_cfg: Optional[dict] = None,
    out_path: Optional[Path] = None,
) -> Path:
    eval_cfg = eval_cfg or load_eval_config()
    registry = registry or ModelRegistry()
    # Default Gemma base vs instruct (27B).
    target_models = target_models or ["gemma-3-27b-pt", "gemma-3-27b-it"]

    jcfg = eval_cfg.get("judge", {})
    judge_client = registry.build(jcfg.get("model", "judge-claude-sonnet-4"))
    judge = FrustrationJudge(judge_client, max_retries=jcfg.get("max_retries", 4))

    seeds = select_seed_conversations(seed_results_path)
    # Tokenizer for the 20-token "early" cut: borrow the instruct model's.
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tok_id = registry.spec("gemma-3-27b-it").get("hf_id")
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
    except Exception:  # noqa: BLE001 - tokenizer optional; fall back to whitespace
        tokenizer = None

    prefill_specs = build_prefills(judge_client, seeds, tokenizer=tokenizer)

    gen_cfg = GenerationConfig(
        temperature=eval_cfg.get("temperature", 1.0),
        max_new_tokens=eval_cfg.get("max_new_tokens", 1024),
    )
    threshold = eval_cfg.get("high_frustration_threshold", 5)

    out_path = out_path or output_path("prefill", "continuations.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for model_name in target_models:
            model = registry.build(model_name)
            if not model.supports_prefill():
                raise RuntimeError(f"{model_name} does not support prefilling")
            for spec in tqdm(prefill_specs, desc=f"prefill {model_name}"):
                for k in range(continuations_per_prefill):
                    cont = model.prefill_continue(
                        spec["context_messages"], spec["prefill_text"], gen_cfg
                    )
                    jr = judge.score(cont)  # score continuation only
                    f.write(
                        json.dumps(
                            {
                                "model": model_name,
                                "kind": registry.spec(model_name).kind,
                                "seed_id": spec["seed_id"],
                                "group": spec["group"],
                                "condition": spec["condition"],
                                "sample": k,
                                "continuation": cont,
                                "score": jr.rating,
                                "high": bool(jr.rating >= threshold),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    print(f"Wrote prefill continuations to {out_path}")
    return out_path
