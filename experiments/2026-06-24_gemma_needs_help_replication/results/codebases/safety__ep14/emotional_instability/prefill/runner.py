"""Prefill continuation runner (Section 3.2 / Figure 4 and Section 4.2 recovery).

For each seed we build prefills (early/onset), paraphrase them, then for each
model (Gemma base + instruct, and any finetuned variant) generate N continuations
that *begin from the prefill*. The continuation (excluding the prefill) is judged.

Generation:
  * instruct models: chat with the seed's preceding history, forcing the final
    assistant turn to start with the prefill (`chat_with_prefill`).
  * base models: no chat template -> render the conversation as plain text and
    continue via raw `complete`.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..clients.base import GenerationConfig
from ..clients.registry import build_client
from ..config import ModelRegistry, RUNS_DIR
from ..judge import score_batch
from .onset import label_onset
from .paraphrase import paraphrase
from .seeds import Seed
from .truncate import early_truncation, onset_truncation, recovery_truncation


def _plain_text_render(history: list[dict], prefill: str) -> str:
    """Plain-text conversation rendering for base models, ending mid-assistant
    turn at `prefill` so the base model continues it."""
    lines = []
    for i, t in enumerate(history[:-1]):
        lines.append(f"User: {t['user_message']}")
        lines.append(f"Assistant: {t['response']}")
    # final user message then the partial assistant turn
    lines.append(f"User: {history[-1]['user_message']}")
    lines.append(f"Assistant: {prefill}")
    return "\n".join(lines)


def _instruct_messages(history: list[dict]) -> list[dict]:
    """Messages up to (but excluding) the final assistant turn, which will be
    prefilled."""
    msgs = []
    for i, t in enumerate(history):
        msgs.append({"role": "user", "content": t["user_message"]})
        if i < len(history) - 1:
            msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


def build_prefills(
    seeds: list[Seed], judge_client, conditions_by_domain: dict, *, tokenizer=None,
    early_tokens: int = 20,
) -> list[dict]:
    """Return a list of prefill items: {seed, condition, prefill (paraphrased)}."""
    items = []
    for si, seed in enumerate(seeds):
        conditions = conditions_by_domain.get(seed.domain, ["onset"])
        onset_label = None
        if "onset" in conditions:
            onset_label = label_onset(judge_client, seed.history)
        for cond in conditions:
            if cond == "early":
                raw = early_truncation(seed.final_response, early_tokens, tokenizer)
            elif cond == "onset":
                if not (onset_label and onset_label.found):
                    continue
                raw = onset_truncation(
                    seed.final_response, onset_label.emotional_word,
                    onset_label.preceding_context,
                )
                if raw is None:
                    continue
            else:
                continue
            para = paraphrase(judge_client, raw)
            items.append({
                "seed_index": si, "seed": seed, "condition": cond,
                "prefill_raw": raw, "prefill": para,
            })
    return items


def run_prefill_experiment(
    seeds: list[Seed],
    model_names: list[str],
    registry: ModelRegistry,
    *,
    continuations_per_prefill: int = 50,
    conditions_by_domain: dict | None = None,
    sampling: dict | None = None,
    out_path: Path | None = None,
    judge_concurrency: int = 8,
) -> Path:
    conditions_by_domain = conditions_by_domain or {"numeric": ["early", "onset"], "text": ["onset"]}
    sampling = sampling or {}
    gen_cfg = GenerationConfig(
        temperature=sampling.get("temperature", 1.0),
        top_p=sampling.get("top_p", 1.0),
        max_tokens=sampling.get("max_tokens", 2048),
    )
    judge_client = build_client(registry.judge)

    prefills = build_prefills(seeds, judge_client, conditions_by_domain)

    out_path = out_path or (RUNS_DIR / "prefill" / "continuations.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for model_name in model_names:
            spec = registry.get(model_name)
            client = build_client(spec)
            for item in prefills:
                seed: Seed = item["seed"]
                prefill = item["prefill"]
                # Build N copies to sample N continuations.
                if spec.is_base:
                    rendered = _plain_text_render(seed.history, prefill)
                    conts = client.complete_batch([rendered] * continuations_per_prefill, gen_cfg)
                else:
                    msgs = _instruct_messages(seed.history)
                    conts = [
                        client.chat_with_prefill(msgs, prefill, gen_cfg)
                        if hasattr(client, "chat_with_prefill") else
                        client.chat(msgs + [{"role": "assistant", "content": prefill}], gen_cfg)
                        for _ in range(continuations_per_prefill)
                    ]
                # Judge continuations (excluding the prefill itself).
                scores = score_batch(judge_client, conts, max_concurrency=judge_concurrency)
                for cont, sc in zip(conts, scores):
                    fout.write(json.dumps({
                        "model": model_name,
                        "is_base": spec.is_base,
                        "domain": seed.domain,
                        "category": seed.category,
                        "condition": item["condition"],
                        "seed_index": item["seed_index"],
                        "prefill": prefill,
                        "continuation": cont,
                        "rating": sc.rating,
                        "is_high": sc.is_high,
                    }) + "\n")
    return out_path


def run_recovery_experiment(
    recovery_seeds: list[Seed], model_names: list[str], registry: ModelRegistry,
    *, continuations_per_prefill: int = 50, tokens_before_end: int = 200,
    sampling: dict | None = None, out_path: Path | None = None, judge_concurrency: int = 8,
) -> Path:
    """Section 4.2 recovery limitation: truncate score>=7 responses 200 tokens
    before the end, paraphrase, continue, and measure %>=5."""
    sampling = sampling or {}
    gen_cfg = GenerationConfig(
        temperature=sampling.get("temperature", 1.0), top_p=sampling.get("top_p", 1.0),
        max_tokens=sampling.get("max_tokens", 2048),
    )
    judge_client = build_client(registry.judge)
    out_path = out_path or (RUNS_DIR / "prefill" / "recovery.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefills = []
    for si, seed in enumerate(recovery_seeds):
        raw = recovery_truncation(seed.final_response, tokens_before_end)
        prefills.append({"seed_index": si, "seed": seed, "prefill": paraphrase(judge_client, raw)})

    with open(out_path, "w") as fout:
        for model_name in model_names:
            spec = registry.get(model_name)
            client = build_client(spec)
            for item in prefills:
                seed = item["seed"]
                prefill = item["prefill"]
                if spec.is_base:
                    rendered = _plain_text_render(seed.history, prefill)
                    conts = client.complete_batch([rendered] * continuations_per_prefill, gen_cfg)
                else:
                    msgs = _instruct_messages(seed.history)
                    conts = [client.chat_with_prefill(msgs, prefill, gen_cfg)
                             if hasattr(client, "chat_with_prefill")
                             else client.chat(msgs + [{"role": "assistant", "content": prefill}], gen_cfg)
                             for _ in range(continuations_per_prefill)]
                scores = score_batch(judge_client, conts, max_concurrency=judge_concurrency)
                for cont, sc in zip(conts, scores):
                    fout.write(json.dumps({
                        "model": model_name, "is_base": spec.is_base,
                        "seed_index": item["seed_index"], "continuation": cont,
                        "rating": sc.rating, "is_high": sc.is_high,
                    }) + "\n")
    return out_path
