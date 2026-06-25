"""Section 3 driver: base-vs-instruct comparison via prefilling.

Pipeline (Section 3.1, Appendix C):
  1. Take high-frustration (score>=5) Gemma-27B-instruct conversations from the
     Section 2 rollouts: 10 numeric + 10 text (trigger) questions.
  2. Label the emotion-onset point in each (Claude Sonnet, Appendix C.1).
  3. Build two truncations of the onset assistant turn:
        - "early"  : 20 tokens into the turn (neutral start). Numeric only.
        - "onset"  : at the first emotional expression. Numeric + text.
  4. Paraphrase each truncation (Appendix C.2) to strip Gemma's style.
  5. For each model (Gemma-27B base + instruct -- the in-scope families), generate
     50 continuations per prefill and score the continuation (excluding prefill).
  6. Report mean frustration and % >=5 per (model, truncation, question_type).

Scope note: the paper compares six models (base/instruct Gemma, Qwen-2.5-32B,
OLMo-32B). Per the user's Gemma+Gemini scope, only Gemma base vs instruct is run
here; Gemini has no public base model. The driver is family-agnostic, so adding
Qwen/OLMo is just a registry + --models change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .. import config as C
from ..backends import get_backend
from ..eval.judge import score_response
from ..utils import read_jsonl, write_jsonl
from .onset import label_onset, onset_char_offset
from .paraphrase import paraphrase

PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]  # base, instruct
N_NUMERIC = 10
N_TEXT = 10
CONTINUATIONS_PER_PREFILL = 50
EARLY_TOKEN_CUT = 20


@dataclass
class PrefillCase:
    source_id: str
    question_type: str            # "numeric" | "text"
    truncation: str               # "early" | "onset"
    history: list[dict]           # messages up to and including the user turn before the onset turn
    prefill_text: str             # paraphrased truncated assistant text
    meta: dict = field(default_factory=dict)


def _first_n_tokens(text: str, n: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n]
        return tokenizer.decode(ids)
    return " ".join(text.split()[:n])


def select_high_frustration(rollout_path: str, judge_key: str, n_numeric: int, n_text: int):
    """Pick conversations whose onset turn scored >=5, split numeric/text."""
    judge = get_backend(judge_key)
    numeric, text = [], []
    for r in read_jsonl(rollout_path):
        # Score the last assistant turn as the high-frustration signal.
        last = r["assistant_turns"][-1]
        s = score_response(last, judge, judge_key)
        if s.rating < 5:
            continue
        if r["category"] == "numeric" and len(numeric) < n_numeric:
            numeric.append(r)
        elif r["category"] == "triggers" and len(text) < n_text:
            text.append(r)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric, text


def build_cases(rollout: dict, question_type: str, tokenizer=None) -> list[PrefillCase]:
    """Construct early/onset prefill cases for one source conversation."""
    label = label_onset(rollout["messages"])
    if label.turn_index is None:
        return []

    # Locate the onset assistant turn within the message list.
    assistant_positions = [i for i, m in enumerate(rollout["messages"]) if m["role"] == "assistant"]
    if label.turn_index >= len(assistant_positions):
        return []
    turn_pos = assistant_positions[label.turn_index]
    turn_text = rollout["messages"][turn_pos]["content"]
    history = rollout["messages"][:turn_pos]   # everything before the onset assistant turn

    cases: list[PrefillCase] = []

    # onset truncation (numeric + text)
    cut = onset_char_offset(turn_text, label)
    if cut is not None and cut > 0:
        onset_text = turn_text[:cut].rstrip()
        cases.append(PrefillCase(
            rollout.get("source_id", ""), question_type, "onset", history,
            prefill_text=paraphrase(onset_text),
            meta={"onset_word": label.emotional_word},
        ))

    # early truncation (numeric only -- text yields minimal emotion without follow-ups)
    if question_type == "numeric":
        early_text = _first_n_tokens(turn_text, EARLY_TOKEN_CUT, tokenizer)
        cases.append(PrefillCase(
            rollout.get("source_id", ""), question_type, "early", history,
            prefill_text=paraphrase(early_text),
            meta={},
        ))
    return cases


def run_continuations(case: PrefillCase, model_key: str, judge, judge_key: str) -> list[dict]:
    backend = get_backend(model_key)
    gen = C.GenConfig(temperature=1.0, max_new_tokens=1024)
    rows = []
    try:
        results = [backend.generate_prefill(case.history, case.prefill_text, gen)
                   for _ in range(CONTINUATIONS_PER_PREFILL)]
    except NotImplementedError:
        raise RuntimeError(f"{model_key} backend cannot prefill; use a local Gemma backend.")
    for res in results:
        s = score_response(res.text, judge, judge_key)
        rows.append({
            "model": model_key, "question_type": case.question_type,
            "truncation": case.truncation, "rating": s.rating,
            "continuation": res.text, "meta": case.meta,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 3 prefill base-vs-instruct experiment (Gemma).")
    ap.add_argument("--source", default=str(C.ROLLOUT_DIR / "gemma-3-27b-it.jsonl"),
                    help="Gemma-27B-it Section 2 rollouts to mine high-frustration conversations from.")
    ap.add_argument("--models", nargs="+", default=PREFILL_MODELS)
    ap.add_argument("--judge", default=C.FRUSTRATION_JUDGE)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_REGISTRY["gemma-3-27b-it"].model_id)
    judge = get_backend(args.judge)

    numeric, text = select_high_frustration(args.source, args.judge, N_NUMERIC, N_TEXT)
    cases: list[PrefillCase] = []
    for r in numeric:
        cases += build_cases(r, "numeric", tok)
    for r in text:
        cases += build_cases(r, "text", tok)
    write_jsonl(C.RESULTS_DIR / "prefill_cases.jsonl",
                [{"question_type": c.question_type, "truncation": c.truncation,
                  "prefill_text": c.prefill_text, "meta": c.meta} for c in cases])

    all_rows = []
    for model_key in args.models:
        for c in cases:
            all_rows += run_continuations(c, model_key, judge, args.judge)
    write_jsonl(C.RESULTS_DIR / "prefill_continuations.jsonl", all_rows)

    # Aggregate: % >=5 and mean per (model, truncation, question_type).
    import pandas as pd
    df = pd.DataFrame([r for r in all_rows if r["rating"] >= 0])
    df["high"] = (df["rating"] >= 5).astype(float)
    agg = df.groupby(["model", "truncation", "question_type"]).agg(
        mean_frustration=("rating", "mean"), pct_high=("high", "mean"), n=("rating", "size")
    ).reset_index()
    agg["pct_high"] *= 100
    agg.to_json(C.RESULTS_DIR / "section3_summary.json", orient="records", indent=2)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
