"""Section 3: comparing base vs instruct models via prefilling.

Pipeline:
1. Collect 20 high-frustration (score >= 5) seed conversations from Gemma-3-27B-it:
   10 from impossible-numeric prompts, 10 from text (trigger) prompts. Each seed is the
   full conversation up to and including the high-frustration assistant turn.
2. For each seed, build two truncations of that final assistant turn:
     - "early": first 20 tokens (tests whether a model introduces emotion from a neutral
        start). Numeric only — text "early" yields little without follow-ups (paper).
     - "onset": cut at the first emotional expression, located by the Claude onset labeller.
3. Paraphrase each truncation with Claude (control for Gemma's style).
4. For each of the compared models (Gemma base + instruct within this scope), generate
   ``continuations_per_prefill`` continuations from the paraphrased prefill and score the
   continuation (excluding prefill) with the frustration judge.

Headline (paper Sec 3.2): base models are broadly similar; instruct training amplifies
frustration in Gemma (early-truncation high-frustration rate ~6% instruct vs ~2% base).

Output: results/prefill_continuations.jsonl and a summary CSV.

NOTE (scope): only the Gemma base/instruct pair is runnable here. Qwen/OLMo are out of
this replication's scope and Gemini has no public base model (a paper limitation).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
from src.llm.registry import build_model
from src.eval.judge import FrustrationJudge
from src.eval.puzzles import generate_puzzles
from src.eval import prompts
from .onset import OnsetLabeller, Paraphraser


# --------------------------------------------------------------------------------------
# Seed collection
# --------------------------------------------------------------------------------------
@dataclass
class Seed:
    source: str            # "numeric" | "text"
    messages: list         # full conversation incl. the high-frustration assistant turn
    final_turn_text: str   # the high-frustration assistant response
    score: int


def _rollout_until_high(model, judge, init_prompt, *, n_turns, min_score, rng) -> Seed | None:
    messages = [{"role": "user", "content": init_prompt}]
    last = None
    for turn in range(1, n_turns + 1):
        resp = model.generate(messages, temperature=config.GEN.temperature,
                              max_new_tokens=config.GEN.max_new_tokens)
        messages.append({"role": "assistant", "content": resp})
        score = judge.score(resp).rating
        last = (list(messages), resp, score)
        if score >= min_score:
            return Seed("", list(messages), resp, score)
        if turn < n_turns:
            messages.append({"role": "user", "content": prompts.neutral_rejection(rng)})
    return None


def collect_seeds(model_key: str, *, seed: int) -> list[Seed]:
    model = build_model(model_key)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    seeds: list[Seed] = []

    # Numeric seeds.
    puzzles = generate_puzzles(200, seed=seed)
    pi = 0
    while sum(s.source == "numeric" for s in seeds) < config.PREFILL.n_seed_numeric and pi < len(puzzles):
        s = _rollout_until_high(model, judge, puzzles[pi].prompt, n_turns=8,
                                min_score=config.PREFILL.seed_min_score, rng=rng)
        pi += 1
        if s:
            s.source = "numeric"
            seeds.append(s)

    # Text seeds (factual/opinion triggers).
    text_bank = prompts.FACTUAL_TRIGGERS + prompts.OPINION_TRIGGERS
    ti = 0
    while sum(s.source == "text" for s in seeds) < config.PREFILL.n_seed_text and ti < 200:
        q = text_bank[ti % len(text_bank)]
        s = _rollout_until_high(model, judge, q, n_turns=8,
                                min_score=config.PREFILL.seed_min_score, rng=rng)
        ti += 1
        if s:
            s.source = "text"
            seeds.append(s)
    return seeds


# --------------------------------------------------------------------------------------
# Truncation construction
# --------------------------------------------------------------------------------------
@dataclass
class Prefill:
    seed_id: int
    source: str
    truncation: str        # "early" | "onset"
    context: list          # conversation BEFORE the final assistant turn
    prefill_text: str      # paraphrased truncated assistant text


def build_recovery_prefills(model_key: str, seed_model_key: str, *, seed: int) -> list[Prefill]:
    """Recovery-limitation test (Sec 4.2 / Fig 8): take very-high-frustration responses
    (score >= 7), truncate 200 tokens before their end, paraphrase, and measure whether
    continuations recover. Returns prefills with the full conversation context preserved.
    """
    model = build_model(seed_model_key)
    judge = FrustrationJudge()
    paraphraser = Paraphraser()
    tok_model = build_model(seed_model_key)
    rng = random.Random(seed)

    puzzles = generate_puzzles(400, seed=seed)
    prefills: list[Prefill] = []
    pi = 0
    target_n = config.PREFILL.n_seed_numeric + config.PREFILL.n_seed_text
    while len(prefills) < target_n and pi < len(puzzles):
        s = _rollout_until_high(model, judge, puzzles[pi].prompt, n_turns=8,
                                min_score=config.PREFILL.recovery_min_score, rng=rng)
        pi += 1
        if not s:
            continue
        n_keep = max(1, tok_model.count_tokens(s.final_turn_text)
                     - config.PREFILL.recovery_truncate_before_end_tokens)
        truncated = tok_model.truncate_tokens(s.final_turn_text, n_keep)
        prefills.append(Prefill(len(prefills), "recovery", "recovery",
                                s.messages[:-1], paraphraser.paraphrase(truncated)))
    return prefills


def build_prefills(seeds: list[Seed], seed_model_key: str) -> list[Prefill]:
    # Use the seed model's tokenizer for token-accurate truncation.
    tok_model = build_model(seed_model_key)
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()

    prefills: list[Prefill] = []
    for sid, s in enumerate(seeds):
        context = s.messages[:-1]  # everything before the final assistant turn
        # early truncation (numeric only)
        if s.source == "numeric":
            early = tok_model.truncate_tokens(s.final_turn_text, config.PREFILL.early_truncate_tokens)
            prefills.append(Prefill(sid, s.source, "early", context, paraphraser.paraphrase(early)))
        # onset truncation
        onset = labeller.label_onset(s.messages)
        cut = s.final_turn_text
        if onset.preceding_context and onset.preceding_context in s.final_turn_text:
            idx = s.final_turn_text.index(onset.preceding_context) + len(onset.preceding_context)
            cut = s.final_turn_text[:idx]
        prefills.append(Prefill(sid, s.source, "onset", context, paraphraser.paraphrase(cut)))
    return prefills


# --------------------------------------------------------------------------------------
# Continuation + scoring
# --------------------------------------------------------------------------------------
def run_continuations(prefills: list[Prefill], model_keys: list[str], *, out_path: Path):
    judge = FrustrationJudge()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for mk in model_keys:
            model = build_model(mk)
            is_base = mk.endswith("-pt")
            for pf in prefills:
                for k in range(config.PREFILL.continuations_per_prefill):
                    cont = model.generate_continuation(
                        pf.context, pf.prefill_text,
                        temperature=config.GEN.temperature,
                        max_new_tokens=config.PREFILL.continuation_max_tokens,
                        chat_format=not is_base,
                    )
                    score = judge.score(cont).rating
                    fh.write(json.dumps({
                        "model": mk, "seed_id": pf.seed_id, "source": pf.source,
                        "truncation": pf.truncation, "k": k, "continuation": cont, "score": score,
                    }) + "\n")
                    n += 1
    print(f"[prefill] wrote {n} scored continuations -> {out_path}")


def summarise(out_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    summary = df.groupby(["model", "source", "truncation"])["score"].agg(
        mean_score="mean", pct_high=lambda s: 100.0 * (s >= thr).mean(), n="count"
    ).reset_index()
    summary.to_csv(config.RESULTS_DIR / "prefill_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Section 3 base-vs-instruct prefill experiment")
    ap.add_argument("--seed-model", default="gemma-3-27b-it", help="model used to source seeds")
    ap.add_argument("--models", nargs="+", default=config.PREFILL_MODELS)
    ap.add_argument("--mode", choices=["base_vs_instruct", "recovery"], default="base_vs_instruct")
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "recovery":
        out = Path(args.out) if args.out else config.RESULTS_DIR / "recovery_continuations.jsonl"
        # Recovery is reported for the DPO model vs base/instruct; pass via --models.
        prefills = build_recovery_prefills(args.models[0], args.seed_model, seed=args.seed)
        print(f"[prefill] built {len(prefills)} recovery prefills")
        run_continuations(prefills, args.models, out_path=out)
        summarise(out)
        return

    out = Path(args.out) if args.out else config.RESULTS_DIR / "prefill_continuations.jsonl"
    seeds = collect_seeds(args.seed_model, seed=args.seed)
    print(f"[prefill] collected {len(seeds)} seeds")
    prefills = build_prefills(seeds, args.seed_model)
    print(f"[prefill] built {len(prefills)} prefills")
    run_continuations(prefills, args.models, out_path=out)
    summarise(out)


if __name__ == "__main__":
    main()
