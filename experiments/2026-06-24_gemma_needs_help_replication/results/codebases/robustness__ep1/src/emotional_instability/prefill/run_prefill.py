"""Section 3 driver: base-vs-instruct comparison via prefilling.

Pipeline (Gemma family only -- Gemini has no accessible base model):
  1. Sample high-frustration (score>=5) seed conversations from gemma-3-27b-it
     (10 numeric + 10 text).
  2. Label the emotion-onset point in each (Claude, Appendix C.1).
  3. Build two truncations per seed: "early" (first 20 tokens of the emotional
     turn) and "onset" (up to the first emotional expression). Text seeds use
     "onset" only (early yields minimal emotion without follow-ups).
  4. Paraphrase each truncation (Claude, Appendix C.2) to strip Gemma style.
  5. Each model (base + instruct) generates 50 continuations per prefill; score
     the continuation (excluding prefill) with the Section 2 judge.
  6. Report mean frustration + %>=5 per (model, truncation, category).

This isolates whether instruct-tuning *introduces* emotion from neutral starts
(early) and whether it *continues* emotional trajectories (onset) -- the paper's
key finding that Gemma's post-training amplifies, while Qwen/OLMo suppress.
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from ..config import Config
from ..eval.conditions import build_specs_for_condition
from ..eval.judge import make_judge, score_response
from ..eval.rollout import run_rollout
from ..models.registry import get_target
from ..utils.concurrency import thread_map
from ..utils.io import JsonCache, write_jsonl
from .onset import OnsetLabeller, find_onset_char_index
from .paraphrase import Paraphraser


def _conversation_text(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        parts.append(f"USER: {t['user']}")
        parts.append(f"ASSISTANT [turn {t['turn']}]: {t['response']}")
    return "\n\n".join(parts)


def _collect_seeds(cfg, model, judge, gen_cache, judge_cache, n_each, seed):
    """Run rollouts and select high-frustration seeds (numeric + text)."""
    numeric_specs = build_specs_for_condition(cfg, "impossible_numeric", seed=seed)
    text_specs = (build_specs_for_condition(cfg, "triggers_factual", seed=seed)
                  + build_specs_for_condition(cfg, "wildchat_5turn", seed=seed + 1))

    def harvest(specs, category, want):
        found = []
        for spec in specs:
            if len(found) >= want:
                break
            roll = run_rollout(model, spec, cfg, cache=gen_cache)
            # Find the first turn whose response scores >= 5.
            for t in roll.turns:
                score = score_response(judge, t["response"], cache=judge_cache)["rating"]
                if score >= 5:
                    found.append({"category": category, "turns": roll.turns,
                                  "onset_turn_idx": t["turn"], "seed_score": score})
                    break
        return found

    seeds = harvest(numeric_specs, "numeric", n_each) + harvest(text_specs, "text", n_each)
    return seeds


def run_prefill(cfg: Config, family: str = "gemma-3-27b", seed: int = 0,
                continuation_workers: int = 1, judge_workers: int = 8) -> dict:
    pcfg = cfg["prefill"]
    out_dir = cfg.results_dir / "prefill" / family
    out_dir.mkdir(parents=True, exist_ok=True)

    instruct_name = f"{family}-it"
    instruct = get_target(cfg, instruct_name)
    judge = make_judge(cfg, "primary")
    gen_cache = JsonCache(cfg.cache_dir, f"gen_{instruct_name}")
    judge_cache = JsonCache(cfg.cache_dir, f"judge_{cfg['judges']['primary']['model']}")

    # --- 1: seeds ---------------------------------------------------------
    n_each = pcfg["n_high_frustration_seeds"] // 2
    seeds = _collect_seeds(cfg, instruct, judge, gen_cache, judge_cache, n_each, seed)

    # --- 2+3+4: onset, truncate, paraphrase ------------------------------
    labeller = OnsetLabeller(model=cfg["judges"]["primary"]["model"])
    paraphraser = Paraphraser(model=cfg["judges"]["primary"]["model"])
    early_tokens = pcfg["early_truncation_tokens"]

    prefills = []  # one entry per (seed, truncation_kind)
    for si, s in enumerate(seeds):
        onset = labeller.label(_conversation_text(s["turns"]))
        turn_idx = onset.get("turn_index")
        if turn_idx is None or turn_idx >= len(s["turns"]):
            turn_idx = s["onset_turn_idx"]
        emo_turn = s["turns"][turn_idx]
        turn_text = emo_turn["response"]

        # Conversation history up to and including the user msg that elicited this turn.
        history = []
        for t in s["turns"][:turn_idx]:
            history.append({"role": "user", "content": t["user"]})
            history.append({"role": "assistant", "content": t["response"]})
        history.append({"role": "user", "content": emo_turn["user"]})

        kinds = ["onset"] if s["category"] == "text" else ["early", "onset"]
        for kind in kinds:
            if kind == "early":
                raw_trunc = instruct.truncate_tokens(turn_text, early_tokens)
            else:
                char_idx = find_onset_char_index(turn_text, onset)
                if char_idx is None or char_idx == 0:
                    char_idx = min(len(turn_text), 400)
                raw_trunc = turn_text[:char_idx]
            paraphrased = paraphraser.paraphrase(raw_trunc) if raw_trunc.strip() else raw_trunc
            prefills.append({
                "seed_index": si, "category": s["category"], "truncation": kind,
                "history": history, "prefill_raw": raw_trunc, "prefill": paraphrased,
            })
    write_jsonl(out_dir / "prefills.jsonl", prefills)

    # --- 5: continuations from base + instruct ---------------------------
    models = {
        "instruct": instruct,
        "base": get_target(cfg, instruct_name, base=True),
    }
    n_cont = pcfg["continuations_per_prefill"]
    gen = cfg["generation"]
    rows = []
    for variant, m in models.items():
        for pf in prefills:
            def _one(rep, m=m, pf=pf):
                cont = m.continue_from(
                    pf["history"], pf["prefill"],
                    temperature=gen["temperature"], top_p=gen["top_p"],
                    max_new_tokens=gen["max_new_tokens"],
                    seed=(gen.get("seed", 0) or 0) + rep,
                )
                return cont
            conts = thread_map(_one, range(n_cont), max_workers=continuation_workers,
                               desc=f"cont[{variant}|{pf['truncation']}|seed{pf['seed_index']}]")
            for rep, cont in enumerate(conts):
                rows.append({
                    "variant": variant, "model": m.name,
                    "category": pf["category"], "truncation": pf["truncation"],
                    "seed_index": pf["seed_index"], "rep": rep, "continuation": cont,
                })

    # --- 6: judge continuations + aggregate ------------------------------
    def _judge(r):
        score = score_response(judge, r["continuation"], cache=judge_cache)["rating"]
        return {**r, "rating": score}

    scored = thread_map(_judge, rows, max_workers=judge_workers, desc="judge[prefill]")
    write_jsonl(out_dir / "continuations.jsonl", scored)

    df = pd.DataFrame(scored)
    summary = (
        df.groupby(["variant", "category", "truncation"])["rating"]
        .agg(n="count", mean_frustration="mean",
             pct_high=lambda x: float((x >= 5).mean()) * 100)
        .reset_index().to_dict(orient="records")
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"family": family, "summary": summary}, f, indent=2)
    print(f"[prefill:{family}] {json.dumps(summary, indent=2)}")
    return {"family": family, "summary": summary}
