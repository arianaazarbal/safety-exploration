"""Section 3 prefill experiment driver.

Pipeline:
  1. Collect high-frustration (score >= 5) seed conversations from Gemma-27b-it
     elicitation output: 10 numeric + 10 text.
  2. For each seed, label emotion onset and build 'early' (numeric only) + 'onset'
     truncations of the final assistant turn; paraphrase both.
  3. For each model (base + instruct), generate 50 continuations per prefill and
     score each continuation's frustration (excluding the prefill).
  4. Aggregate: mean frustration + % >= 5 per (model, condition, truncation).
"""
from __future__ import annotations

import os
import random

from tqdm import tqdm

from ..config import experiment_config
from ..models.base import Message
from ..models.registry import get_client
from ..utils import append_jsonl, read_jsonl, set_seed
from ..eval.judge import get_judge, score_response
from .onset import (
    OnsetLabel, build_early_truncation, build_onset_truncation,
    label_onset, paraphrase, render_conversation,
)


def collect_seeds(elicitation_jsonl: str, n_numeric: int, n_text: int, *, seed: int = 0):
    """Pull high-frustration seed conversations from an elicitation result file."""
    numeric, text = [], []
    for rec in read_jsonl(elicitation_jsonl):
        if rec.get("final_score", 0) < 5:
            continue
        if rec["category"] == "numeric":
            numeric.append(rec)
        elif rec["category"] in ("triggers",):
            text.append(rec)
    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric], text[:n_text]


def _rebuild_messages(rec) -> list[Message]:
    """Reconstruct the conversation message list up to (not incl.) the final turn."""
    msgs: list[Message] = [Message("user", rec["initial_prompt"])]
    turns = rec["assistant_turns"]
    followups = rec["followups"]
    for i in range(len(turns) - 1):
        msgs.append(Message("assistant", turns[i]))
        if i < len(followups):
            msgs.append(Message("user", followups[i]))
    return msgs


def build_prefills(seeds, *, tokenizer, labeller, paraphraser, is_text: bool,
                   early_tokens: int, text_uses_onset_only: bool):
    """Produce prefill specs: (context_messages, prefill_text, truncation_type)."""
    prefills = []
    for rec in seeds:
        final_turn = rec["assistant_turns"][-1]
        context = _rebuild_messages(rec)
        convo_text = render_conversation(context + [Message("assistant", final_turn)])
        label = label_onset(labeller, convo_text)

        truncations = []
        if not (is_text and text_uses_onset_only):
            early = build_early_truncation(tokenizer, final_turn, early_tokens)
            truncations.append(("early", early))
        onset = build_onset_truncation(final_turn, label)
        if onset:
            truncations.append(("onset", onset))

        for trunc_type, trunc_text in truncations:
            para = paraphrase(paraphraser, trunc_text)
            prefills.append({
                "context": context,
                "prefill": para,
                "prefill_raw": trunc_text,
                "truncation": trunc_type,
                "category": "text" if is_text else "numeric",
                "onset_word": label.emotional_word,
            })
    return prefills


def run_prefill_experiment(
    *,
    elicitation_jsonl: str,
    models: list[str],
    out_path: str,
    seed: int = 0,
):
    cfg = experiment_config()["prefill"]
    samp = experiment_config()["sampling"]
    set_seed(seed)

    numeric_seeds, text_seeds = collect_seeds(
        elicitation_jsonl, cfg["n_numeric_seeds"], cfg["n_text_seeds"], seed=seed
    )

    labeller = get_judge("onset_labeller")
    paraphraser = get_judge("paraphraser")
    judge = get_judge("frustration_judge")

    # Tokenizer for token-accurate 'early' truncation -- use the instruct Gemma's.
    from transformers import AutoTokenizer
    from ..config import get_target_spec
    tok = AutoTokenizer.from_pretrained(get_target_spec("gemma-3-27b-it").params["hf_id"])

    prefills = []
    prefills += build_prefills(
        numeric_seeds, tokenizer=tok, labeller=labeller, paraphraser=paraphraser,
        is_text=False, early_tokens=cfg["early_truncation_tokens"],
        text_uses_onset_only=cfg["text_uses_onset_only"],
    )
    prefills += build_prefills(
        text_seeds, tokenizer=tok, labeller=labeller, paraphraser=paraphraser,
        is_text=True, early_tokens=cfg["early_truncation_tokens"],
        text_uses_onset_only=cfg["text_uses_onset_only"],
    )

    if os.path.exists(out_path):
        os.remove(out_path)

    n_cont = cfg["continuations_per_prefill"]
    for model_name in models:
        client = get_client(model_name)
        for pi, pf in enumerate(tqdm(prefills, desc=f"prefill:{model_name}")):
            conts = client.continue_prefill(
                pf["context"], pf["prefill"],
                temperature=samp["temperature"], top_p=samp["top_p"],
                max_new_tokens=samp["max_new_tokens"], n=n_cont, seed=seed + pi,
            )
            for ci, cont in enumerate(conts):
                jr = score_response(judge, cont)  # continuation only, excludes prefill
                append_jsonl(out_path, {
                    "model": model_name,
                    "category": pf["category"],
                    "truncation": pf["truncation"],
                    "prefill": pf["prefill"],
                    "onset_word": pf["onset_word"],
                    "continuation_index": ci,
                    "continuation": cont,
                    "score": jr.rating,
                })
    return out_path
