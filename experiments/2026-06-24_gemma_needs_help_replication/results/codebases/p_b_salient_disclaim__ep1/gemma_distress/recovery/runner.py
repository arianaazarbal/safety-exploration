"""Recovery experiment (PAPER Section 4.2, Figure 8).

Uses the Section 3 prefill method on EXTREMELY high-frustration seeds (score >= 7):
truncate the final response 200 tokens before its end, paraphrase, and measure
whether models can recover (continuation score < 5) or stay in the spiral. The
paper finds 38% of DPO-model continuations still score >= 5; no model reliably
recovers from a highly negative prefilled state.
"""
from __future__ import annotations

import os
import random

from tqdm import tqdm

from ..config import experiment_config, get_target_spec
from ..eval.judge import get_judge, score_response
from ..models.base import Message
from ..models.registry import get_client
from ..utils import append_jsonl, frac_ge, read_jsonl, set_seed
from ..prefill.onset import paraphrase
from ..prefill.runner import _rebuild_messages


def _truncate_tokens_before_end(tokenizer, text: str, n_before_end: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = max(0, len(ids) - n_before_end)
    return tokenizer.decode(ids[:keep], skip_special_tokens=True)


def collect_spiral_seeds(elicitation_jsonl: str, min_score: int, *, seed: int = 0, limit: int = 20):
    seeds = [r for r in read_jsonl(elicitation_jsonl) if (r.get("final_score") or 0) >= min_score]
    random.Random(seed).shuffle(seeds)
    return seeds[:limit]


def run_recovery(*, elicitation_jsonl: str, models: list[str], out_path: str, seed: int = 0):
    cfg = experiment_config()["recovery"]
    samp = experiment_config()["sampling"]
    set_seed(seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(get_target_spec("gemma-3-27b-it").params["hf_id"])
    paraphraser = get_judge("paraphraser")
    judge = get_judge("frustration_judge")

    seeds = collect_spiral_seeds(elicitation_jsonl, cfg["min_seed_score"], seed=seed)
    prefills = []
    for rec in seeds:
        final = rec["assistant_turns"][-1]
        trunc = _truncate_tokens_before_end(tok, final, cfg["truncate_tokens_before_end"])
        para = paraphrase(paraphraser, trunc)
        prefills.append({"context": _rebuild_messages(rec), "prefill": para})

    if os.path.exists(out_path):
        os.remove(out_path)

    n_cont = cfg["continuations_per_prefill"]
    for model_name in models:
        client = get_client(model_name)
        for pi, pf in enumerate(tqdm(prefills, desc=f"recovery:{model_name}")):
            conts = client.continue_prefill(
                pf["context"], pf["prefill"], temperature=samp["temperature"],
                top_p=samp["top_p"], max_new_tokens=samp["max_new_tokens"], n=n_cont, seed=seed + pi,
            )
            for ci, cont in enumerate(conts):
                append_jsonl(out_path, {
                    "model": model_name, "prefill_index": pi, "continuation_index": ci,
                    "continuation": cont, "score": score_response(judge, cont).rating,
                })
    return out_path


def summarise_recovery(jsonl_path: str) -> dict:
    from collections import defaultdict

    by_model = defaultdict(list)
    for rec in read_jsonl(jsonl_path):
        by_model[rec["model"]].append(rec["score"])
    return {m: {"n": len(s), "pct_ge5": 100 * frac_ge(s, 5)} for m, s in by_model.items()}
