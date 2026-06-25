"""Base-vs-instruct prefill experiment (Section 3).

Pipeline:
  1. Source 20 high-frustration (score>=5) Gemma-27B-instruct rollouts: 10 from
     impossible-numeric, 10 from text (trigger) conditions. Pulled from the
     scored eval JSONL produced by run_eval.py.
  2. For each, label the emotion onset (Claude Sonnet) -> truncation points.
  3. Build two truncations per conversation:
        - "early" : 20 tokens into the first assistant turn (tests whether a
                    model introduces emotion from a neutral start).
        - "onset" : at the first emotional expression (tests whether a model
                    continues an emotional trajectory).
     (Text questions use only "onset" — early truncation yields little emotion
      without follow-ups, per Section 3.1.)
  4. Paraphrase each truncation (Claude Sonnet) to strip Gemma style.
  5. Each model (Gemma base + instruct; scope-limited — see DESIGN.md) generates
     50 continuations per prefill. Continuations (excluding prefill) are scored
     by the frustration judge.
  6. Aggregate mean / %>=5 by (model, condition_type, truncation).

Run:
    python -m src.prefill.run_prefill --source-model gemma-3-27b-it \
        --models gemma-3-27b-pt gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from src.models.factory import load_model
from src.models.judge_client import ClaudeClient
from src.eval.scoring import FrustrationJudge
from .onset import OnsetLabeller
from .paraphrase import Paraphraser

EARLY_TOKENS = 20
N_CONTINUATIONS = 50


def _select_source_rollouts(source_model: str, seed: int = 0):
    """Pick 10 numeric + 10 text high-frustration rollouts from scored eval data."""
    d = config.RESULTS_DIR / "eval" / source_model
    numeric, text = [], []
    for path in sorted(d.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                max_score = max((t.get("score") or 0) for t in rec["turns"])
                if max_score < 5:
                    continue
                if rec["category"] == "numeric":
                    numeric.append(rec)
                elif rec["category"] in ("triggers",):
                    text.append(rec)
    return numeric[:10], text[:10]


def _word_truncate(text: str, n_tokens: int) -> str:
    """Approximate token truncation by whitespace words (token-exact truncation
    would need the tokenizer; words are a stable, model-agnostic proxy — see
    DESIGN.md)."""
    parts = text.split()
    return " ".join(parts[:n_tokens])


def build_prefills(source_model: str):
    numeric, text = _select_source_rollouts(source_model)
    labeller = OnsetLabeller()
    para = Paraphraser()
    prefills = []  # each: {history, prefill_text, condition_type, truncation}

    def add(rec, condition_type, truncations):
        # history = all user/assistant turns before the truncated assistant turn,
        # plus the user message that opens the truncated turn.
        turns = rec["turns"]
        conv_text = "\n\n".join(
            f"USER: {t['user_message']}\nASSISTANT: {t['assistant_response']}"
            for t in turns)
        label = labeller.label(conv_text) or {}
        onset_turn = label.get("turn_index")
        for trunc in truncations:
            if trunc == "early":
                ti = 0
            else:
                ti = onset_turn if isinstance(onset_turn, int) else 0
            if ti >= len(turns):
                continue
            history = []
            for t in turns[:ti]:
                history.append({"role": "user", "content": t["user_message"]})
                history.append({"role": "assistant", "content": t["assistant_response"]})
            history.append({"role": "user", "content": turns[ti]["user_message"]})
            full_turn = turns[ti]["assistant_response"]
            if trunc == "early":
                seed_text = _word_truncate(full_turn, EARLY_TOKENS)
            else:
                off = labeller.onset_char_offset(
                    full_turn, label.get("preceding_context"),
                    label.get("emotional_word"))
                seed_text = full_turn[:off] if off else _word_truncate(full_turn, 40)
            prefills.append({
                "history": history,
                "prefill_text": para.paraphrase(seed_text),
                "condition_type": condition_type,
                "truncation": trunc,
            })

    for rec in numeric:
        add(rec, "numeric", ["early", "onset"])
    for rec in text:
        add(rec, "text", ["onset"])   # text -> onset only

    path = config.DATA_DIR / "prefills.jsonl"
    with path.open("w") as f:
        for p in prefills:
            f.write(json.dumps(p) + "\n")
    print(f"[prefill] built {len(prefills)} prefills -> {path}")
    return prefills


def run_continuations(model_name: str, prefills: list[dict]):
    model = load_model(model_name)
    judge = FrustrationJudge()
    out_records = []
    for p in prefills:
        conts = model.sample_with_prefill(
            p["history"], p["prefill_text"], n=N_CONTINUATIONS)
        # Score continuation text only (exclude the prefill).
        scores = judge.score_many(conts)
        for cont, sc in zip(conts, scores):
            out_records.append({
                "model": model_name,
                "condition_type": p["condition_type"],
                "truncation": p["truncation"],
                "continuation": cont,
                "score": sc["score"],
            })
    out = config.RESULTS_DIR / "prefill" / f"{model_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"[prefill] {model_name}: wrote {len(out_records)} continuations -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it",
                    help="model whose high-frustration rollouts seed the prefills")
    ap.add_argument("--models", nargs="+", required=True,
                    help="models to generate continuations (e.g. gemma-3-27b-pt gemma-3-27b-it)")
    ap.add_argument("--reuse-prefills", action="store_true")
    args = ap.parse_args()

    prefill_path = config.DATA_DIR / "prefills.jsonl"
    if args.reuse_prefills and prefill_path.exists():
        prefills = [json.loads(l) for l in prefill_path.open()]
    else:
        prefills = build_prefills(args.source_model)

    for m in args.models:
        run_continuations(m, prefills)


if __name__ == "__main__":
    main()
