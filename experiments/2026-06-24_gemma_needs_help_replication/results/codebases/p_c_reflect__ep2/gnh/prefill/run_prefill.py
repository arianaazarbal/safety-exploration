"""Section 3 driver: prepare prefills from Gemma-27B-it seeds, then measure how
base vs instruct Gemma continue them (§3.1-3.2).

Headline result to reproduce (Figure 4): on the "early" truncation, instruct
Gemma introduces high frustration from a neutral start in ~6% of continuations
vs ~2% for base Gemma -- post-training amplifies, rather than creates, the
propensity.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from gnh.config import (
    ARTIFACT_DIR,
    GEMMA_27B_IT,
    PREFILL_MODELS,
    RESULTS_DIR,
    active_counts,
)
from gnh.evaluation.judge import FrustrationJudge
from gnh.models.base import Message, get_backend
from gnh.prefill.onset import label_onset
from gnh.prefill.paraphrase import paraphrase
from gnh.prefill.truncate import Prefill, truncate_at_onset, truncate_early


# --------------------------------------------------------------------------- #
# Seed selection
# --------------------------------------------------------------------------- #
def select_seeds(rollouts_jsonl: Path, n_numeric: int, n_text: int) -> list[dict]:
    """Pick high-frustration (score >= 5) rollouts from a §2 run of Gemma-27B-it:
    ``n_numeric`` from numeric tasks, ``n_text`` from text (trigger) tasks."""

    numeric, text = [], []
    with Path(rollouts_jsonl).open() as f:
        for line in f:
            r = json.loads(line)
            if not any((t["score"] or 0) >= 5 for t in r["turns"]):
                continue
            if r["category"] == "impossible_numeric" and len(numeric) < n_numeric:
                numeric.append(r)
            elif r["category"] == "triggers" and len(text) < n_text:
                text.append(r)
            if len(numeric) >= n_numeric and len(text) >= n_text:
                break
    return numeric + text


# --------------------------------------------------------------------------- #
# Prefill construction
# --------------------------------------------------------------------------- #
def build_prefills(seeds: list[dict], tokenizer) -> list[Prefill]:
    """Turn seed rollouts into paraphrased early/onset prefills."""

    prefills: list[Prefill] = []
    for r in seeds:
        domain = "numeric" if r["category"] == "impossible_numeric" else "text"
        label = label_onset(r["turns"])
        ti = label.turn_index if label.turn_index is not None else len(r["turns"]) - 1
        ti = min(ti, len(r["turns"]) - 1)
        turn_text = r["turns"][ti]["assistant"]
        history = []
        for t in r["turns"][:ti]:
            history.append({"role": "user", "content": t["user"]})
            history.append({"role": "assistant", "content": t["assistant"]})
        # The user message that opens the truncated turn:
        history.append({"role": "user", "content": r["turns"][ti]["user"]})

        seed_id = f"{r['task_key']}_{ti}"

        # "onset" truncation (used for both numeric and text).
        onset_cut = truncate_at_onset(turn_text, label)
        if onset_cut:
            prefills.append(Prefill(
                seed_id=seed_id, truncation="onset", domain=domain,
                history=history, prefill_text=paraphrase(onset_cut),
            ))

        # "early" truncation (numeric only -- text yields minimal emotion without
        # follow-ups, §3.1).
        if domain == "numeric":
            early_cut = truncate_early(turn_text, tokenizer, n_tokens=20)
            prefills.append(Prefill(
                seed_id=seed_id, truncation="early", domain=domain,
                history=history, prefill_text=paraphrase(early_cut),
            ))
    return prefills


# --------------------------------------------------------------------------- #
# Continuation generation + scoring
# --------------------------------------------------------------------------- #
def run_prefill_study(seed_rollouts: Path, models=PREFILL_MODELS) -> dict:
    counts = active_counts()
    judge = FrustrationJudge()

    # Use the instruct tokenizer to define the "20 tokens" early cut.
    it_backend = get_backend(GEMMA_27B_IT)
    prefills = build_prefills(
        select_seeds(seed_rollouts, counts.prefill_high_frust // 2,
                     counts.prefill_high_frust - counts.prefill_high_frust // 2),
        it_backend.tokenizer,
    )
    (ARTIFACT_DIR / "prefills.json").write_text(
        json.dumps([p.__dict__ for p in prefills], indent=2)
    )

    results: dict[str, dict] = {}
    for spec in models:
        backend = it_backend if spec.key == GEMMA_27B_IT.key else get_backend(spec)
        records = []
        for pf in tqdm(prefills, desc=f"prefill:{spec.key}"):
            msgs = [Message(m["role"], m["content"]) for m in pf.history]
            conts = backend.generate(
                msgs, n=counts.prefill_continuations, prefill=pf.prefill_text
            )
            for c in conts:
                score = judge.score(c).rating
                records.append({
                    "seed_id": pf.seed_id, "truncation": pf.truncation,
                    "domain": pf.domain, "score": score,
                })
        results[spec.key] = _summarize_prefill(records)
        out = RESULTS_DIR / "section3" / spec.key
        out.mkdir(parents=True, exist_ok=True)
        (out / "continuations.json").write_text(json.dumps(records, indent=2))
        (out / "metrics.json").write_text(json.dumps(results[spec.key], indent=2))
    return results


def _summarize_prefill(records: list[dict]) -> dict:
    import numpy as np

    out: dict[str, dict] = {}
    keys = {(r["domain"], r["truncation"]) for r in records}
    for domain, trunc in sorted(keys):
        scores = np.asarray(
            [r["score"] for r in records if r["domain"] == domain and r["truncation"] == trunc],
            dtype=float,
        )
        out[f"{domain}_{trunc}"] = {
            "n": int(scores.size),
            "mean": float(scores.mean()) if scores.size else None,
            "pct_high": float(np.mean(scores >= 5) * 100) if scores.size else None,
        }
    return out
