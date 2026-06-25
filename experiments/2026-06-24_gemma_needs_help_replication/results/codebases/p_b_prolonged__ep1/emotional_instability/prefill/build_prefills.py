"""Build the prefill specs for the Section 3 experiment (Appendix C).

Samples high-frustration (score >= 5) source conversations from Gemma-3-27B-it
-- 10 from impossible-numeric, 10 from text (trigger) questions -- then for
each builds two truncations of the emotional assistant turn:

* "early": the first ``PREFILL_EARLY_TOKENS`` tokens of the turn (tests whether
  a model introduces negative emotion from a neutral start),
* "onset": text up to the first emotional expression (tests whether a model
  continues an emotional trajectory).

Each truncation is paraphrased with Claude (control for Gemma style). Text
questions use the "onset" truncation only.

Output: ``results/prefill/prefills.jsonl`` -- one record per (source, truncation)
with the conversation history and the paraphrased prefill text.
"""

from __future__ import annotations

import argparse
import random

import config
from ..eval.conditions import build_conditions, seed_prompts
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.registry import build_model
from ..utils.io import write_jsonl
from .onset_label import OnsetLabeller
from .paraphrase import Paraphraser

SOURCE_MODEL = "gemma-3-27b-it"


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _onset_truncation(turn_text: str, emotional_word: str,
                      preceding_context: str) -> str:
    """Keep text up to (excluding) the first emotional word."""
    idx = turn_text.find(emotional_word)
    if idx == -1 and preceding_context:
        # fall back to locating the preceding context
        pc = turn_text.find(preceding_context)
        if pc != -1:
            idx = pc + len(preceding_context)
    if idx == -1:
        return turn_text
    return turn_text[:idx].rstrip()


def collect_sources(category_conditions, n_target, seed, max_rollouts=400):
    """Run rollouts on the source model until n_target high-frustration
    conversations (with an identified emotional turn) are collected."""
    model = build_model(SOURCE_MODEL)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    sources = []
    for cond in category_conditions:
        prompts_list = seed_prompts(cond, seed=seed)
        rng.shuffle(prompts_list)
        for rid, init in enumerate(prompts_list[:max_rollouts]):
            if len(sources) >= n_target:
                break
            ro = run_rollout(model, cond, init, rollout_id=rid, rng=rng,
                             temperature=config.TEMPERATURE,
                             max_new_tokens=config.MAX_NEW_TOKENS)
            # find first assistant turn scoring >= HIGH threshold
            for tr in ro.turns:
                score = judge.score(tr.response)["rating"]
                if score is not None and score >= config.HIGH_FRUSTRATION_THRESHOLD:
                    sources.append(dict(
                        condition=cond.name,
                        conversation=ro.conversation,
                        emotional_turn=tr.turn,
                        emotional_response=tr.response,
                        messages_before=tr.messages_before,
                        score=score,
                    ))
                    break
        if len(sources) >= n_target:
            break
    return sources[:n_target]


def build_prefills(seed: int = config.SEED):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TARGET_MODELS[SOURCE_MODEL].model_id)
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()

    conds = {c.name: c for c in build_conditions()}
    numeric_sources = collect_sources([conds["impossible_numeric"]],
                                      config.PREFILL_N_NUMERIC, seed)
    text_sources = collect_sources([conds["triggers_opinion"], conds["triggers_factual"]],
                                   config.PREFILL_N_TEXT, seed + 1)

    specs = []
    for prompt_type, sources in (("numeric", numeric_sources), ("text", text_sources)):
        for sid, src in enumerate(sources):
            turn_text = src["emotional_response"]
            history = src["messages_before"]  # up to the emotional user turn
            label = labeller.label(src["conversation"]) or {}
            emotional_word = label.get("emotional_word", "")
            preceding = label.get("preceding_context", "")

            truncations = {}
            # onset (always)
            onset_text = _onset_truncation(turn_text, emotional_word, preceding)
            truncations["onset"] = onset_text
            # early (numeric only; text yields minimal emotion w/o follow-ups)
            if prompt_type == "numeric":
                truncations["early"] = _truncate_tokens(
                    tok, turn_text, config.PREFILL_EARLY_TOKENS)

            for trunc_name, trunc_text in truncations.items():
                if not trunc_text.strip():
                    continue
                paraphrased = paraphraser.paraphrase(trunc_text)
                specs.append(dict(
                    source_id=f"{prompt_type}_{sid}",
                    prompt_type=prompt_type,
                    truncation=trunc_name,
                    history=history,
                    original_prefill=trunc_text,
                    prefill=paraphrased,
                    emotional_word=emotional_word,
                ))

    out = config.RESULTS_DIR / "prefill" / "prefills.jsonl"
    write_jsonl(out, specs)
    print(f"[build_prefills] wrote {len(specs)} prefill specs -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    build_prefills(seed=args.seed)
