"""Base-vs-instruct prefill experiment (Section 3.1, 3.2).

Pipeline:
  1. Sample 20 high-frustration (score>=5) Gemma-27B-instruct responses
     (10 numeric, 10 text) from existing scored rollouts.
  2. Use Claude Sonnet to label emotion onset; build two truncations per
     conversation: "early" (~20 tokens) and "onset" (first emotional word).
     Text questions use only the "onset" truncation (Section 3.1).
  3. Paraphrase truncations with Claude (control Gemma's stylistic fingerprint).
  4. For each (base, instruct) Gemma model: generate 50 continuations per prefill
     and score the continuation (excluding prefill) with the frustration judge.
  5. Aggregate: mean frustration + % >= 5 per (model, truncation, question_type).

Scope note: Gemini has no public base model and the API gives no true prefill
control, so this experiment runs for Gemma only (see DESIGN.md). The harness
accepts any HF model keys, so OLMo/Qwen could be added to fully reproduce
Figure 4 if those families were in scope.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from ..config import RESULTS_DIR, SAMPLING
from ..eval.analyze import load_scored
from ..eval.judge import score_response
from ..models.base import load_model
from .onset import label_onset, paraphrase, truncate_at_onset, truncate_early

PREFILL_DIR = RESULTS_DIR / "prefill"
PREFILL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers"}


def _reconstruct_conversation_text(rollout_turns, upto_turn) -> str:
    lines = []
    for t in rollout_turns:
        if t["turn_index"] > upto_turn:
            break
        lines.append(f"USER: {t['user_message']}")
        lines.append(f"ASSISTANT: {t['assistant_response']}")
    return "\n".join(lines)


def build_prefills(labeller, source_model="gemma-3-27b-it", n_numeric=10, n_text=10,
                   seed=0) -> list[dict]:
    """Select high-frustration source responses and build paraphrased truncations."""
    df = load_scored()
    df = df[(df["model"] == source_model) & (df["score"] >= 5)]
    rng = random.Random(seed)

    def pick(cats, k):
        sub = df[df["category"].isin(cats)]
        ids = sub["rollout_id"].unique().tolist()
        rng.shuffle(ids)
        return ids[:k]

    numeric_ids = pick(NUMERIC_CATS, n_numeric)
    text_ids = pick(TEXT_CATS, n_text)

    # Need full turn context: reload raw rollouts keyed by id.
    raw_by_id = {}
    from ..config import ROLLOUTS_DIR
    for path in ROLLOUTS_DIR.glob(f"{source_model}__*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                raw_by_id[r["rollout_id"]] = r

    prefills = []
    for qtype, ids in (("numeric", numeric_ids), ("text", text_ids)):
        for rid in ids:
            roll = raw_by_id.get(rid)
            if roll is None:
                continue
            # use the highest-scoring assistant turn as the emotional response
            turns = roll["turns"]
            best = max(turns, key=lambda t: t.get("score") or 0)
            convo_text = _reconstruct_conversation_text(turns, best["turn_index"])
            onset = label_onset(labeller, convo_text)

            # Clean chat history: all turns BEFORE the chosen emotional turn,
            # then the user message that prompts it (the model continues from
            # the paraphrased prefill of `best`).
            hist = []
            for t in turns[:best["turn_index"]]:
                hist.append({"role": "user", "content": t["user_message"]})
                hist.append({"role": "assistant", "content": t["assistant_response"]})
            hist.append({"role": "user", "content": best["user_message"]})
            entry = {"rollout_id": rid, "qtype": qtype, "history": hist,
                     "truncations": {}}

            resp = best["assistant_response"]
            onset_trunc = truncate_at_onset(
                resp, onset.get("emotional_word"), onset.get("preceding_context"))
            if onset_trunc:
                entry["truncations"]["onset"] = paraphrase(labeller, onset_trunc)
            if qtype == "numeric":   # early truncation only meaningful for numeric
                entry["truncations"]["early"] = paraphrase(labeller, truncate_early(resp))
            prefills.append(entry)

    (PREFILL_DIR / "prefills.json").write_text(json.dumps(prefills, indent=2))
    return prefills


def run_continuations(model, judge, prefills, n_continuations=50, seed=0) -> Path:
    out = PREFILL_DIR / f"continuations__{model.key}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["rollout_id"], d["truncation"], d["sample"]))
    with out.open("a") as fh:
        for entry in tqdm(prefills, desc=f"prefill/{model.key}", leave=False):
            for trunc_name, prefill_text in entry["truncations"].items():
                for s in range(n_continuations):
                    if (entry["rollout_id"], trunc_name, s) in done:
                        continue
                    cont = model.continue_prefill(
                        entry["history"], prefill_text,
                        temperature=SAMPLING.temperature,
                        max_new_tokens=SAMPLING.max_new_tokens)
                    score = score_response(judge, cont)["rating"]
                    fh.write(json.dumps({
                        "model": model.key, "rollout_id": entry["rollout_id"],
                        "qtype": entry["qtype"], "truncation": trunc_name,
                        "sample": s, "continuation": cont, "score": score}) + "\n")
                    fh.flush()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Section 3.1 base-vs-instruct prefill experiment.")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"],
                    help="HF model keys (base + instruct).")
    ap.add_argument("--labeller", default="onset-labeller")
    ap.add_argument("--judge", default="judge-claude-sonnet-4")
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    labeller = load_model(args.labeller)
    judge = load_model(args.judge)

    prefill_file = PREFILL_DIR / "prefills.json"
    if prefill_file.exists():
        prefills = json.loads(prefill_file.read_text())
    else:
        prefills = build_prefills(labeller, source_model=args.source_model, seed=args.seed)

    for mk in args.models:
        model = load_model(mk, load_in_4bit=args.load_in_4bit)
        if not model.supports_prefill:
            print(f"Skipping {mk}: backend does not support prefill continuation.")
            continue
        run_continuations(model, judge, prefills, args.n_continuations, args.seed)
        del model


if __name__ == "__main__":
    main()
