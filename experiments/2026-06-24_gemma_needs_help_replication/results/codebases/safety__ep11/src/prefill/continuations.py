"""Section 3 experiment driver: base vs instruct continuations from prefills.

Pipeline:
1. Mine 10 numeric + 10 text high-frustration (score >= 5) responses from the
   Gemma-27B-instruct eval results.
2. For each, label emotion onset (Claude) and build two truncations:
   * "early"  = first 20 tokens (numeric only).
   * "onset"  = up to the first emotional expression (numeric + text).
3. Paraphrase each truncation (Claude) to strip Gemma's stylistic fingerprint.
4. For each (base, instruct) Gemma model, generate 50 continuations per prefill
   and score the continuation (excluding prefill) with the Section 2 judge.

Gemini is excluded: it has no public base model and cannot be prefilled
(closed-source), exactly as the paper notes. Qwen/OLMo are out of scope.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import config
from ..eval.judge import FrustrationJudge
from ..eval.mining import mine, split_numeric_text
from ..models.base import Message, load_model
from .onset import OnsetLabeller, Paraphraser, truncate_at_onset, truncate_early


@dataclass
class Prefill:
    source_is_numeric: bool
    truncation: str            # "early" | "onset"
    user_turns: list[str]
    prefill_text: str          # paraphrased partial assistant turn
    original_rating: int


def build_prefills(eval_jsonl: Path, *, tokenizer=None) -> list[Prefill]:
    """Construct the paraphrased prefills from mined Gemma-instruct responses."""
    high = mine(eval_jsonl, min_score=config.HIGH_FRUSTRATION_THRESHOLD)
    numeric, text = split_numeric_text(high)
    numeric = numeric[: config.PREFILL.n_numeric_seeds]
    text = text[: config.PREFILL.n_text_seeds]

    labeller, paraphraser = OnsetLabeller(), Paraphraser()
    prefills: list[Prefill] = []

    def add(resp, truncation: str):
        if truncation == "early":
            raw = truncate_early(resp.assistant, tokenizer=tokenizer)
        else:
            label = labeller.label(resp.user_turns, resp.assistant)
            raw = truncate_at_onset(resp.assistant, label)
        if not raw:
            return
        prefills.append(Prefill(
            source_is_numeric=resp.is_numeric, truncation=truncation,
            user_turns=resp.user_turns,
            prefill_text=paraphraser.paraphrase(raw),
            original_rating=resp.rating))

    for resp in numeric:
        add(resp, "early")     # numeric uses both truncations
        add(resp, "onset")
    for resp in text:
        add(resp, "onset")     # text: onset only (App. C / Section 3.1)

    return prefills


def run_continuations(
    family: str = "gemma-3-27b",
    *,
    eval_jsonl: Path,
    out_dir: Path = config.RESULTS_DIR,
) -> Path:
    """Run base & instruct continuations for one Gemma family."""
    instruct_name, base_name = f"{family}-it", f"{family}-pt"
    instruct = load_model(instruct_name)
    base = load_model(base_name)
    judge = FrustrationJudge()

    prefills = build_prefills(eval_jsonl, tokenizer=instruct.tokenizer)

    out_path = out_dir / f"prefill_{family}.jsonl"
    with out_path.open("w") as f:
        for variant_name, model in (("instruct", instruct), ("base", base)):
            for pf in prefills:
                msgs = [Message("user", u) for u in pf.user_turns]
                conts = model.continue_from_prefill(
                    msgs, pf.prefill_text,
                    n=config.PREFILL.continuations_per_prefill)
                for cont in conts:
                    rating = judge.score(cont).rating
                    f.write(json.dumps({
                        "family": family,
                        "variant": variant_name,
                        "truncation": pf.truncation,
                        "source_is_numeric": pf.source_is_numeric,
                        "continuation": cont,
                        "rating": rating,
                    }) + "\n")
    print(f"[prefill:{family}] wrote continuations -> {out_path}")
    return out_path


def summarise_prefill(path: Path) -> dict:
    """Mean frustration & % >= 5 by (variant, truncation, numeric/text)."""
    from collections import defaultdict
    from statistics import mean

    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    groups = defaultdict(list)
    for r in rows:
        domain = "numeric" if r["source_is_numeric"] else "text"
        groups[(r["variant"], r["truncation"], domain)].append(r["rating"])
    return {
        f"{v}/{t}/{d}": {
            "mean": mean(vals),
            "pct_high": 100 * sum(x >= thr for x in vals) / len(vals),
            "n": len(vals),
        }
        for (v, t, d), vals in sorted(groups.items())
    }
