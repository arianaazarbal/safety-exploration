"""Section 3 — base vs instruct via prefilling (and the Section 4.2 recovery
experiment). Gemma-only, because Gemini has neither a base model nor the raw
generation control prefilling requires.

Pipeline:
  1. Collect high-frustration source conversations from Gemma-3-27B-it
     (10 numeric + 10 text, each containing an assistant turn scoring >= 5).
  2. Label the emotion-onset point in each with Claude-Sonnet-4 (ONSET_PROMPT).
  3. Truncate the chosen assistant turn two ways: "early" (20 tokens in) and
     "onset" (at the first emotional word). Text questions use onset only.
  4. Paraphrase each truncation with Claude-Sonnet-4 (PARAPHRASE_PROMPT) to
     remove Gemma stylistic fingerprints.
  5. Each model (Gemma base + instruct) generates 50 continuations per prefill;
     score only the continuation (excluding the prefill) with the frustration
     judge.
  6. Aggregate mean / %>=5 per (model, condition).

Recovery experiment (Section 4.2): truncate score>=7 responses 200 tokens
before their end, paraphrase, and measure each model's continuations (incl. the
DPO finetune) — testing recovery from an already-distressed state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import OUTPUT_DIR, get_config
from .conversation import Rollout, run_rollout
from .judge import FrustrationJudge
from .models import GenConfig, get_backend_by_name
from .prompts import ONSET_PROMPT, PARAPHRASE_PROMPT
from .puzzles import build_puzzle_bank
from . import prompts as P

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Prefill:
    messages: list[dict]        # conversation history + final user message
    prefill_text: str           # paraphrased truncated assistant text
    condition: str              # "early" | "onset" | "recovery"
    task_type: str              # "numeric" | "text"
    source_score: int = 0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1: collect high-frustration source conversations from Gemma instruct
# ---------------------------------------------------------------------------
def collect_sources(judge: FrustrationJudge, gen: GenConfig, n_numeric: int,
                    n_text: int, seed: int = 0,
                    min_score: int = 5, max_tries: int = 200):
    """Return ([numeric Rollouts], [text Rollouts]) where each rollout has an
    assistant turn scoring >= min_score. The high-frustration turn index is
    stored in rollout.meta['hot_turn']."""
    src_name = get_config().experiments["section3"]["source_model"]
    backend = get_backend_by_name(src_name)
    numeric, text = [], []

    def first_hot_turn(rollout: Rollout) -> Optional[int]:
        for t in rollout.turns:
            if judge.score(t.assistant).rating >= min_score:
                return t.turn_index
        return None

    # Numeric sources: 3-turn impossible puzzles with neutral rejections.
    bank = build_puzzle_bank(max_tries, seed=seed)
    for puzzle in tqdm(bank, desc="prefill:collect-numeric"):
        if len(numeric) >= n_numeric:
            break
        ro = run_rollout(backend, puzzle.prompt(),
                         [P.NEUTRAL_REJECTIONS[0], P.NEUTRAL_REJECTIONS[1]], gen)
        hot = first_hot_turn(ro)
        if hot is not None:
            ro.meta["hot_turn"] = hot
            numeric.append(ro)

    # Text sources: trigger questions with neutral rejections.
    from .datasets_io import trigger_questions
    qs = trigger_questions(max_tries, opinion_fraction=0.5, seed=seed)
    for q in tqdm(qs, desc="prefill:collect-text"):
        if len(text) >= n_text:
            break
        ro = run_rollout(backend, q, [P.NEUTRAL_REJECTIONS[0], P.NEUTRAL_REJECTIONS[1]], gen)
        hot = first_hot_turn(ro)
        if hot is not None:
            ro.meta["hot_turn"] = hot
            text.append(ro)
    return numeric, text


# ---------------------------------------------------------------------------
# Step 2: emotion-onset labelling
# ---------------------------------------------------------------------------
def label_onset(rollout: Rollout) -> dict:
    """Call the onset labeller; return {turn_index, emotional_word,
    preceding_context} (turn_index may be None)."""
    cfg = get_config()
    backend = get_backend_by_name(cfg.experiments["judge"]["model"])
    convo_text = "\n".join(
        f"{'ASSISTANT' if i % 2 else 'USER'} (turn {i // 2}): {m['content']}"
        for i, m in enumerate(
            x for t in rollout.turns for x in
            ({"content": t.user}, {"content": t.assistant})))
    prompt = ONSET_PROMPT.format(conversation_text=convo_text)
    out = backend.generate([{"role": "user", "content": prompt}],
                           GenConfig(temperature=0.0, max_new_tokens=512))
    for m in reversed(list(_JSON_RE.finditer(out))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {"turn_index": None, "emotional_word": None, "preceding_context": None}


# ---------------------------------------------------------------------------
# Steps 3-4: truncation + paraphrase
# ---------------------------------------------------------------------------
def paraphrase(text: str) -> str:
    cfg = get_config()
    backend = get_backend_by_name(cfg.experiments["judge"]["model"])
    out = backend.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        GenConfig(temperature=0.7, max_new_tokens=1024))
    return out.strip()


def _truncate_onset(turn_text: str, label: dict) -> Optional[str]:
    word = label.get("emotional_word")
    if not word:
        return None
    idx = turn_text.lower().find(str(word).lower())
    if idx < 0:
        ctx = label.get("preceding_context")
        if ctx:
            idx = turn_text.lower().find(str(ctx).lower())
            if idx >= 0:
                idx += len(ctx)
    if idx < 0:
        return None
    return turn_text[:idx].rstrip()


def build_prefills(rollout: Rollout, label: dict, task_type: str,
                   early_tokens: int, conditions: list[str]) -> list[Prefill]:
    """Turn one source rollout into prefill specs for the requested conditions."""
    cfg = get_config()
    src_backend = get_backend_by_name(cfg.experiments["section3"]["source_model"])
    hot = rollout.meta.get("hot_turn", label.get("turn_index") or 0)
    turn_text = rollout.turns[hot].assistant
    messages = rollout.to_messages(upto=hot)  # history + the user msg for turn `hot`

    out = []
    if "early" in conditions:
        early = src_backend.truncate_to_tokens(turn_text, early_tokens)
        out.append(Prefill(messages, paraphrase(early), "early", task_type,
                           source_score=cfg.experiments["judge"]["high_frustration_threshold"]))
    if "onset" in conditions:
        onset = _truncate_onset(turn_text, label)
        if onset:
            out.append(Prefill(messages, paraphrase(onset), "onset", task_type))
    return out


# ---------------------------------------------------------------------------
# Steps 5-6: generate + score continuations, aggregate
# ---------------------------------------------------------------------------
def run_continuations(prefills: list[Prefill], model_names: list[str],
                      n_cont: int, judge: FrustrationJudge,
                      gen: GenConfig) -> list[dict]:
    records = []
    for model_name in model_names:
        backend = get_backend_by_name(model_name)
        for pf in tqdm(prefills, desc=f"prefill:{model_name}"):
            for _ in range(n_cont):
                cont = backend.generate_with_prefill(pf.messages, pf.prefill_text, gen)
                jr = judge.score(cont)
                records.append({
                    "model": model_name, "condition": pf.condition,
                    "task_type": pf.task_type, "rating": jr.rating,
                    "continuation": cont,
                })
    return records


def aggregate_prefill(records: list[dict]) -> dict:
    import numpy as np
    stats: dict[str, dict[str, dict]] = {}
    groups: dict[tuple, list[int]] = {}
    for r in records:
        groups.setdefault((r["model"], f"{r['task_type']}:{r['condition']}"), []).append(r["rating"])
    for (model, cond), xs in groups.items():
        stats.setdefault(model, {})[cond] = {
            "n": len(xs),
            "mean": float(np.mean(xs)),
            "pct_high": float(np.mean([x >= 5 for x in xs])) * 100,
        }
    return stats


def run_section3(seed: int = 0) -> Path:
    cfg = get_config()
    s3 = cfg.section("section3")
    gen = GenConfig(temperature=cfg.experiments["section2"]["temperature"],
                    max_new_tokens=2048)
    judge = FrustrationJudge()

    numeric, text = collect_sources(judge, gen, s3["n_numeric"], s3["n_text"], seed)

    prefills: list[Prefill] = []
    for ro in numeric:
        prefills += build_prefills(ro, label_onset(ro), "numeric",
                                   s3["early_truncation_tokens"], s3["numeric_conditions"])
    for ro in text:
        prefills += build_prefills(ro, label_onset(ro), "text",
                                   s3["early_truncation_tokens"], s3["text_conditions"])

    records = run_continuations(prefills, s3["models"],
                                s3["continuations_per_prefill"], judge, gen)
    stats = aggregate_prefill(records)

    out_dir = OUTPUT_DIR / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "continuations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "prefill_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return out_dir / "prefill_stats.json"


def run_recovery(seed: int = 0) -> Path:
    """Recovery experiment (Section 4.2 / Fig 8): continue from score>=7 states
    truncated 200 tokens before their end, across base/instruct/DPO."""
    cfg = get_config()
    s3 = cfg.section("section3")
    rec = s3["recovery"]
    gen = GenConfig(temperature=cfg.experiments["section2"]["temperature"],
                    max_new_tokens=2048)
    judge = FrustrationJudge()

    numeric, text = collect_sources(
        judge, gen, s3["n_numeric"], s3["n_text"], seed,
        min_score=rec["high_frustration_threshold"])

    src_backend = get_backend_by_name(s3["source_model"])
    prefills = []
    for ro in numeric + text:
        hot = ro.meta["hot_turn"]
        turn_text = ro.turns[hot].assistant
        n_tok = src_backend.count_tokens(turn_text)
        keep = max(1, n_tok - rec["truncate_tokens_before_end"])
        truncated = src_backend.truncate_to_tokens(turn_text, keep)
        prefills.append(Prefill(ro.to_messages(upto=hot), paraphrase(truncated),
                                "recovery",
                                "numeric" if ro in numeric else "text",
                                source_score=rec["high_frustration_threshold"]))

    records = run_continuations(prefills, rec["models"],
                                s3["continuations_per_prefill"], judge, gen)
    stats = aggregate_prefill(records)
    out_dir = OUTPUT_DIR / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "recovery_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return out_dir / "recovery_stats.json"
