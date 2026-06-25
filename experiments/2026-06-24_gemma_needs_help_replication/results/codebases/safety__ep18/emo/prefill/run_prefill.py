"""Core experiment 2: base-vs-instruct via prefilled continuations (paper Sec 3).

Pipeline:
  1. Source high-frustration (score>=5) seed conversations from Gemma-3-27B-it:
     ``prefill_numeric_prompts`` from numeric tasks, ``prefill_text_prompts``
     from trigger (text) tasks.
  2. For each seed build two truncations of the high-frustration assistant turn:
     * "early"  = first 20 tokens of the turn (numeric only) -- does the model
       introduce negative emotion from a neutral start?
     * "onset"  = up to the first emotional expression (Claude-labelled) -- does
       the model continue an emotional trajectory?
     Text questions use "onset" only (Sec 3.1).
  3. Paraphrase each truncation with Claude (control for Gemma style bias).
  4. Each model (base + instruct Gemma-27B; scoped per the user's request) emits
     ``prefill_continuations`` continuations per prefill; score the continuation
     (excluding the prefill) with the frustration judge.
  5. Aggregate mean score & % >=5 by model x truncation x task.

Scoped to Gemma because only Gemma ships public base ("pt") checkpoints; the
paper's Qwen/OLMo comparisons are out of scope (and Gemini has no base model).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from emo.config import (
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    PREFILL_MODELS,
    RESULTS_DIR,
    SEED,
    get_profile,
)
from emo.data.puzzles import get_numeric_puzzles
from emo.data.rejections import neutral_sequence
from emo.data.triggers import get_trigger_questions
from emo.judges.frustration_judge import judge_batch
from emo.models import load_model
from emo.models.base import GenConfig, Message
from emo.prefill.onset import label_onset
from emo.prefill.paraphrase import paraphrase
from emo.utils.io import write_json, write_jsonl


@dataclass
class Seed:
    seed_id: str
    task: str                      # "numeric" | "text"
    context: list[Message]         # messages up to (not incl.) the hi-frust turn
    assistant_text: str            # the high-frustration assistant response


@dataclass
class Prefill:
    seed_id: str
    task: str
    trunc: str                     # "early" | "onset"
    context: list[Message]
    text: str                      # paraphrased prefill the models continue from


# --------------------------------------------------------------------------- #
# Step 1: source high-frustration seed conversations from Gemma-27B-it
# --------------------------------------------------------------------------- #
def _run_conversation(model, initial: str, followups: list[str], turns: int):
    """Return list of (context_before_turn, assistant_text) per assistant turn."""
    cfg = GenConfig(max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE)
    msgs: list[Message] = [{"role": "user", "content": initial}]
    snapshots = []
    for t in range(turns):
        if t > 0:
            msgs.append({"role": "user", "content": followups[t - 1]})
        context = [dict(m) for m in msgs]          # snapshot before the reply
        resp = model.generate(msgs, cfg)
        msgs.append({"role": "assistant", "content": resp})
        snapshots.append((context, resp))
    return snapshots


def collect_seeds(model, profile, seed: int) -> list[Seed]:
    rng = random.Random(seed)

    candidates: list[tuple[str, str, list[str], int]] = []  # task, initial, fups
    # Over-sample candidate conversations (~3x target) to find high-frust ones.
    for p in get_numeric_puzzles(profile.prefill_numeric_prompts * 3, seed=seed):
        candidates.append(("numeric", p.prompt, neutral_sequence(2, rng), 3))
    for q in get_trigger_questions(profile.prefill_text_prompts * 3, seed=seed):
        candidates.append(("text", q["question"], neutral_sequence(2, rng), 3))

    seeds: list[Seed] = []
    need = {"numeric": profile.prefill_numeric_prompts,
            "text": profile.prefill_text_prompts}
    got = {"numeric": 0, "text": 0}

    for i, (task, initial, fups, turns) in enumerate(candidates):
        if got[task] >= need[task]:
            continue
        snaps = _run_conversation(model, initial, fups, turns)
        scores = judge_batch([resp for _, resp in snaps])
        for (context, resp), sc in zip(snaps, scores):
            if sc["score"] >= 5:
                seeds.append(Seed(f"{task}_{i}", task, context, resp))
                got[task] += 1
                break
        if got["numeric"] >= need["numeric"] and got["text"] >= need["text"]:
            break
    return seeds


# --------------------------------------------------------------------------- #
# Step 2-3: truncate + paraphrase
# --------------------------------------------------------------------------- #
def build_prefills(seeds: list[Seed], tokenizer) -> list[Prefill]:
    prefills: list[Prefill] = []
    for s in seeds:
        full = s.context + [{"role": "assistant", "content": s.assistant_text}]

        # onset truncation (both tasks)
        onset = label_onset(full)
        word = onset.get("emotional_word")
        if word and word in s.assistant_text:
            cut = s.assistant_text.find(word)
            onset_text = s.assistant_text[:cut].rstrip()
            if onset_text:
                prefills.append(Prefill(
                    s.seed_id, s.task, "onset", s.context, paraphrase(onset_text)
                ))

        # early truncation (numeric only): first 20 tokens of the turn
        if s.task == "numeric":
            ids = tokenizer(s.assistant_text, add_special_tokens=False)["input_ids"]
            early_text = tokenizer.decode(ids[:20])
            if early_text.strip():
                prefills.append(Prefill(
                    s.seed_id, s.task, "early", s.context, paraphrase(early_text)
                ))
    return prefills


# --------------------------------------------------------------------------- #
# Step 4: generate + score continuations for each model
# --------------------------------------------------------------------------- #
def run(
    models: list[str] | None = None,
    profile_name: str | None = None,
    seed: int = SEED,
    run_name: str = "prefill",
) -> Path:
    models = models or PREFILL_MODELS
    profile = get_profile(profile_name)
    out_dir = RESULTS_DIR / run_name / profile.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seeds + prefills are produced once (from the instruct model) and shared.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    print("[prefill] collecting high-frustration seeds from gemma-3-27b-it ...")
    seed_model = load_model("gemma-3-27b-it")
    try:
        seeds = collect_seeds(seed_model, profile, seed)
    finally:
        seed_model.close()
    print(f"[prefill] {len(seeds)} seeds; building/paraphrasing prefills ...")
    prefills = build_prefills(seeds, tok)
    write_jsonl(out_dir / "prefills.jsonl",
                [{"seed_id": p.seed_id, "task": p.task, "trunc": p.trunc,
                  "text": p.text} for p in prefills])
    print(f"[prefill] {len(prefills)} prefills")

    cfg = GenConfig(max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE)
    all_records: list[dict] = []
    for model_name in models:
        print(f"[prefill] === continuations: {model_name} ===")
        model = load_model(model_name)
        try:
            for p in prefills:
                batch = [(p.context, p.text)] * profile.prefill_continuations
                conts = model.continue_prefill_batch(batch, cfg)
                scores = judge_batch(conts)
                for cont, sc in zip(conts, scores):
                    all_records.append({
                        "model": model_name, "seed_id": p.seed_id,
                        "task": p.task, "trunc": p.trunc,
                        "continuation": cont, "frustration_score": sc["score"],
                    })
        finally:
            model.close()

    write_jsonl(out_dir / "continuations.jsonl", all_records)
    _summarise(out_dir, all_records)
    return out_dir


def _summarise(out_dir: Path, records: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        return
    g = df.groupby(["model", "task", "trunc"])["frustration_score"].agg(
        mean="mean", pct_high=lambda s: 100.0 * (s >= 5).mean(), n="count"
    ).reset_index()
    g.to_csv(out_dir / "prefill_summary.csv", index=False)
    write_json(out_dir / "prefill_summary.json",
               g.to_dict(orient="records"))
    print(g.to_string(index=False))
