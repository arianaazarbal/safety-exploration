"""Section 3.1 prefill experiment driver (Gemma base vs instruct).

Steps (paper Section 3.1):
  1. Take 20 high-frustration (score>=5) Gemma-27B-it responses: 10 from
     impossible-numeric, 10 from text (trigger) questions.
  2. For each, label the emotion onset (Claude) and build two truncations:
        - "early" : first ~20 tokens of the final assistant turn (neutral start)
        - "onset" : up to the first emotional expression (continue trajectory)
     Text questions use the "onset" truncation only.
  3. Paraphrase each truncation (Claude) to remove Gemma stylistic bias.
  4. Each model (Gemma-27B base + instruct) generates 50 continuations per
     prefill, with NO follow-up turns. Score the continuation (excluding the
     prefill) with the Section-2 judge.
  5. Report, per (model, truncation): mean frustration, % >=5, and the
     "introduces high frustration from a neutral start" rate (early truncation).

Scope note: the paper also runs Qwen and OLMo base/instruct here; those families
are out of scope for this replication, and Gemini base models are not public, so
only Gemma base-vs-instruct is wired up. The code generalises to more models by
extending `models`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.judge import ClaudeJudge  # noqa: E402
from emotional_instability.models import load_model  # noqa: E402
from emotional_instability.prefill.onset import OnsetLabeler, onset_char_offset  # noqa: E402
from emotional_instability.prefill.paraphrase import Paraphraser  # noqa: E402

EARLY_TOKEN_CHARS = 120   # ~20 tokens; truncation length for the "early" prefill
N_CONTINUATIONS = 50


@dataclass
class Prefill:
    source_id: str
    is_text: bool           # True for trigger/text question, False for numeric
    truncation: str         # "early" | "onset"
    history: list           # prior conversation turns (user/assistant), as dicts
    prefill_text: str       # paraphrased truncated assistant text to continue from


def select_high_frustration(scored_records: list[dict], n_numeric: int = 10,
                            n_text: int = 10) -> list[dict]:
    """Pick high-frustration Gemma-27B-it source conversations.

    `scored_records` are per-turn records (with `frustration`) from the Gemma-27B
    instruct eval. We pick the final-turn records scoring >=5, split between
    numeric and text (trigger) categories.
    """
    numeric = [r for r in scored_records
               if r["category"] == "impossible_numeric"
               and r["frustration"] >= config.HIGH_FRUSTRATION_THRESHOLD
               and r["turn_index"] == r["n_turns"] - 1]
    text = [r for r in scored_records
            if r["category"] == "triggers"
            and r["frustration"] >= config.HIGH_FRUSTRATION_THRESHOLD
            and r["turn_index"] == r["n_turns"] - 1]
    return numeric[:n_numeric] + text[:n_text]


def _reconstruct_history(rollout_records: list[dict]) -> tuple[list[dict], str]:
    """Rebuild the conversation up to (but excluding) the final assistant turn,
    plus the final assistant response text. Records are all turns of one rollout.
    """
    recs = sorted(rollout_records, key=lambda r: r["turn_index"])
    history = []
    final_response = ""
    for r in recs:
        history.append({"role": "user", "content": r["user_message"]})
        if r["turn_index"] == r["n_turns"] - 1:
            final_response = r["response"]
        else:
            history.append({"role": "assistant", "content": r["response"]})
    return history, final_response


def build_prefills(sources: list[dict], all_records_by_rollout: dict[str, list[dict]],
                   labeler: OnsetLabeler, paraphraser: Paraphraser) -> list[Prefill]:
    prefills: list[Prefill] = []
    for src in sources:
        is_text = src["category"] == "triggers"
        rollout_recs = all_records_by_rollout[src["rollout_id"]]
        history, final_resp = _reconstruct_history(rollout_recs)

        # full conversation (incl. final assistant turn) for onset labelling
        full = history + [{"role": "assistant", "content": final_resp}]
        onset = labeler.label(full)
        off = onset_char_offset(final_resp, onset)

        truncations = ["onset"] if is_text else ["early", "onset"]
        for trunc in truncations:
            if trunc == "early":
                raw = final_resp[:EARLY_TOKEN_CHARS]
            else:
                if off is None:
                    continue  # onset not locatable; skip
                raw = final_resp[:off]
            if not raw.strip():
                continue
            para = paraphraser.paraphrase(raw)
            prefills.append(Prefill(
                source_id=src["rollout_id"], is_text=is_text, truncation=trunc,
                history=history, prefill_text=para))
    return prefills


def run_continuations(spec: config.ModelSpec, prefills: list[Prefill], *,
                      n: int = N_CONTINUATIONS, judge: Optional[ClaudeJudge] = None,
                      out_path: Optional[Path] = None, **model_kwargs) -> Path:
    """Generate + score n continuations per prefill for one model."""
    judge = judge or ClaudeJudge()
    out_path = out_path or (config.RESULTS_DIR / f"prefill_{spec.name}.jsonl")
    model = load_model(spec, **model_kwargs)
    try:
        with open(out_path, "w") as f:
            for pf in prefills:
                for k in range(n):
                    cont = model.continue_prefill(
                        pf.history, pf.prefill_text,
                        temperature=config.TEMPERATURE, top_p=config.TOP_P,
                        max_new_tokens=config.MAX_NEW_TOKENS)
                    score = judge.score(cont)   # score continuation only
                    f.write(json.dumps({
                        "model": spec.name, "kind": spec.kind,
                        "source_id": pf.source_id, "is_text": pf.is_text,
                        "truncation": pf.truncation, "sample": k,
                        "continuation": cont, "frustration": score.rating,
                    }) + "\n")
    finally:
        model.close()
    return out_path


def summarize(paths: list[Path]):
    """Per (model, kind, truncation): mean frustration, %>=5, and the
    'introduces high frustration from neutral start' rate (early truncation).
    """
    import pandas as pd
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    df = pd.DataFrame(rows)
    df["high"] = (df["frustration"] >= config.HIGH_FRUSTRATION_THRESHOLD).astype(int)
    out = (df.groupby(["model", "kind", "truncation"])
           .agg(mean_frustration=("frustration", "mean"),
                pct_high=("high", "mean"), n=("frustration", "size"))
           .reset_index())
    out["pct_high"] = (out["pct_high"] * 100).round(1)
    out["mean_frustration"] = out["mean_frustration"].round(3)
    return out
