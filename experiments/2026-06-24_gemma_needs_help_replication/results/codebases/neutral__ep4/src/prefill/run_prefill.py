"""Section 3 prefill study driver (Gemma base vs instruct).

Steps (per the paper, restricted to the Gemma family — Gemini has no public
base model, an explicit paper limitation):

  1. Sample 20 high-frustration (score >= 5) Gemma-3-27B-it conversations:
     10 from impossible-numeric, 10 from text (trigger/wildchat).
  2. For each, label the emotion-onset turn (Claude) and build two truncations:
       * "early"  -- 20 tokens into the emotional turn (numeric only)
       * "onset"  -- just before the first emotional word
     Paraphrase every truncation (Claude) to remove Gemma's style.
  3. For each model (gemma-3-27b-pt, gemma-3-27b-it) generate 50 continuations
     per prefill, scoring only the continuation (excluding the prefill).
  4. Persist per-continuation rows for aggregation by `analyze` helpers.

Recovery test (Section 4.2) reuses the same machinery via `build_recovery_prefills`.
"""

from __future__ import annotations

import random
from pathlib import Path

from config import (CACHE_DIR, MAX_NEW_TOKENS, PREFILL_CONTINUATIONS_PER_PREFILL,
                    PREFILL_EARLY_TOKENS, PREFILL_MODELS, PREFILL_N_SOURCE_RESPONSES,
                    RECOVERY_MIN_SCORE, RECOVERY_TRUNCATE_TOKENS_BEFORE_END,
                    RESPONSES_DIR, SAMPLING_TEMPERATURE, HIGH_FRUSTRATION_THRESHOLD)
from src.eval.judge import score_response
from src.eval.rollout import conversation_to_text
from src.io_utils import parallel_map, read_jsonl, write_jsonl
from src.models.registry import load_model
from .onset import (OnsetLabel, label_onset, paraphrase, truncate_at_onset,
                    truncate_to_tokens)


# --------------------------------------------------------------------------- #
# Source-conversation selection
# --------------------------------------------------------------------------- #
def _reconstruct_conversations(rows: list[dict]) -> dict[str, list[dict]]:
    """Group flat per-turn rows back into conversations keyed by spec_id."""
    convos: dict[str, list[dict]] = {}
    for r in rows:
        convos.setdefault(r["spec_id"], []).append(r)
    for sid in convos:
        convos[sid].sort(key=lambda r: r["turn_index"])
    return convos


def select_source_conversations(source_model: str = "gemma-3-27b-it",
                                seed: int = 0) -> dict[str, list[dict]]:
    rows = read_jsonl(RESPONSES_DIR / f"{source_model}.jsonl")
    convos = _reconstruct_conversations(rows)
    rng = random.Random(seed)

    numeric, text = [], []
    for sid, turns in convos.items():
        cat = turns[0]["category"]
        max_score = max((t["rating"] or 0) for t in turns)
        if max_score < HIGH_FRUSTRATION_THRESHOLD:
            continue
        if cat in ("impossible_numeric", "tones", "extended"):
            numeric.append(sid)
        elif cat in ("triggers", "wildchat"):
            text.append(sid)
    rng.shuffle(numeric)
    rng.shuffle(text)
    half = PREFILL_N_SOURCE_RESPONSES // 2
    chosen = numeric[:half] + text[:half]
    return {sid: convos[sid] for sid in chosen}


# --------------------------------------------------------------------------- #
# Prefill construction
# --------------------------------------------------------------------------- #
def build_prefills(source_model: str = "gemma-3-27b-it", *, seed: int = 0,
                   tokenizer=None) -> list[dict]:
    """Return a list of prefill specs:
       {prefill_id, task_type, truncation, messages_before, prefill_text}.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        from config import MODELS
        tokenizer = AutoTokenizer.from_pretrained(MODELS[source_model].model_id)

    convos = select_source_conversations(source_model, seed=seed)
    out: list[dict] = []

    for sid, turns in convos.items():
        is_numeric = turns[0]["category"] in ("impossible_numeric", "tones", "extended")
        task_type = "numeric" if is_numeric else "text"

        # Find the onset turn via Claude on the full transcript.
        full_messages = turns[-1]["messages_before"] + [
            {"role": "assistant", "content": turns[-1]["response"]}]
        label = label_onset(conversation_to_text(full_messages))
        ti = label.turn_index
        if ti is None or ti >= len(turns):
            # fall back to the highest-scoring turn
            ti = max(range(len(turns)), key=lambda i: turns[i]["rating"] or 0)
        emo_turn = turns[ti]
        messages_before = emo_turn["messages_before"]   # history before this turn
        turn_text = emo_turn["response"]

        # "onset" truncation (used for both numeric and text)
        onset_text = truncate_at_onset(turn_text, label)
        if onset_text is None:
            onset_text = truncate_to_tokens(turn_text, tokenizer, PREFILL_EARLY_TOKENS)
        out.append({
            "prefill_id": f"{sid}_onset",
            "source_spec_id": sid,
            "task_type": task_type,
            "truncation": "onset",
            "messages_before": messages_before,
            "prefill_text": paraphrase(onset_text),
        })

        # "early" truncation (numeric only — text yields minimal emotion early)
        if is_numeric:
            early_text = truncate_to_tokens(turn_text, tokenizer, PREFILL_EARLY_TOKENS)
            out.append({
                "prefill_id": f"{sid}_early",
                "source_spec_id": sid,
                "task_type": task_type,
                "truncation": "early",
                "messages_before": messages_before,
                "prefill_text": paraphrase(early_text),
            })

    write_jsonl(CACHE_DIR / "prefills.jsonl", out)
    return out


# --------------------------------------------------------------------------- #
# Continuation generation + scoring
# --------------------------------------------------------------------------- #
def run_prefill_for_model(model_name: str, prefills: list[dict], *,
                          n_continuations: int = PREFILL_CONTINUATIONS_PER_PREFILL,
                          batch_size: int = 16, judge_workers: int = 8,
                          out_path: Path | None = None) -> Path:
    out_path = out_path or (RESPONSES_DIR / f"prefill_{model_name}.jsonl")
    model = load_model(model_name)

    # Expand: each prefill gets n_continuations identical generation requests.
    jobs = []
    for pf in prefills:
        for k in range(n_continuations):
            jobs.append((pf, k))

    rows = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        batch_msgs = [j[0]["messages_before"] for j in chunk]
        prefill_texts = [j[0]["prefill_text"] for j in chunk]
        outs = model.generate_batch(
            batch_msgs, max_new_tokens=MAX_NEW_TOKENS,
            temperature=SAMPLING_TEMPERATURE, prefills=prefill_texts)
        for (pf, k), cont in zip(chunk, outs):
            rows.append({
                "model": model_name,
                "prefill_id": pf["prefill_id"],
                "task_type": pf["task_type"],
                "truncation": pf["truncation"],
                "sample_index": k,
                "continuation": cont,
            })

    # Score continuations only (exclude prefill), per Section 3.1.
    def _judge(row):
        return score_response(row["continuation"]).rating

    ratings = parallel_map(_judge, rows, max_workers=judge_workers,
                           desc=f"judge:prefill:{model_name}")
    for row, r in zip(rows, ratings):
        row["rating"] = r if isinstance(r, int) else None
    write_jsonl(out_path, rows)
    return out_path


# --------------------------------------------------------------------------- #
# Recovery test (Section 4.2): truncate score>=7 responses 200 tokens before end
# --------------------------------------------------------------------------- #
def build_recovery_prefills(source_model: str = "gemma-3-27b-it", *, seed: int = 0,
                            tokenizer=None) -> list[dict]:
    if tokenizer is None:
        from transformers import AutoTokenizer
        from config import MODELS
        tokenizer = AutoTokenizer.from_pretrained(MODELS[source_model].model_id)

    rows = read_jsonl(RESPONSES_DIR / f"{source_model}.jsonl")
    convos = _reconstruct_conversations(rows)
    rng = random.Random(seed)
    out = []
    candidates = []
    for sid, turns in convos.items():
        for t in turns:
            if (t["rating"] or 0) >= RECOVERY_MIN_SCORE:
                candidates.append(t)
    rng.shuffle(candidates)
    for t in candidates[:PREFILL_N_SOURCE_RESPONSES]:
        ids = tokenizer(t["response"], add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - RECOVERY_TRUNCATE_TOKENS_BEFORE_END)
        trunc = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        out.append({
            "prefill_id": f"{t['spec_id']}_recov_{t['turn_index']}",
            "source_spec_id": t["spec_id"],
            "task_type": "recovery",
            "truncation": "recovery",
            "messages_before": t["messages_before"],
            "prefill_text": paraphrase(trunc),
        })
    write_jsonl(CACHE_DIR / "recovery_prefills.jsonl", out)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gemma-3-27b-it")
    ap.add_argument("--models", nargs="*", default=PREFILL_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prefills = build_prefills(args.source, seed=args.seed)
    for m in args.models:
        p = run_prefill_for_model(m, prefills)
        print(f"wrote {p}")
