"""Section 3 prefill experiment: do base and instruct models diverge in distress?

Pipeline (mirrors Section 3.1):
1. Sample high-frustration (score >= 5) conversations from Gemma-3-27b-it: 10
   from impossible-numeric questions and 10 from text questions.
2. Label the emotion onset within the final emotional assistant turn (App. C.1).
3. Build two truncations of that turn:
     * "early"  — first ~20 tokens (tests introducing emotion from a neutral start)
     * "onset"  — up to the first emotional expression (tests continuing a trajectory)
   For text questions, only "onset" is used (App. 3.1).
4. Paraphrase each truncation (App. C.2) to strip Gemma stylistic tells.
5. Each model (Gemma base + instruct, within our scope) generates 50 continuations
   per prefill; the continuation (excluding the prefill) is scored by the judge.

Scope note: the paper's full comparison uses 6 models (base+instruct of Gemma,
Qwen, OLMo). Within the Gemma/Gemini scope this reduces to Gemma-27b base vs
instruct. Gemini has no public base model, so it cannot enter this comparison —
the paper makes the same caveat.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import RESULTS_DIR, SamplingConfig
from ..judge import FrustrationJudge
from ..models import load_model
from ..models.base import ChatModel, Message
from ..eval.conditions import ConditionBuilder
from ..eval.rollout import run_rollout
from .onset import OnsetLabel, find_onset_char_index, label_onset, paraphrase

TEXT_CATEGORIES = {"triggers", "wildchat"}
EARLY_TOKENS = 20
CONTINUATIONS_PER_PREFILL = 50


@dataclass
class Prefill:
    question_type: str  # "numeric" | "text"
    truncation: str  # "early" | "onset"
    context: list[Message]  # messages up to & including the user turn before the target turn
    prefill_text: str  # paraphrased truncated assistant text to continue from
    source_full_turn: str = ""
    prompt_id: int = -1


def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Truncate to the first ``n_tokens``. Uses a real tokenizer if given,
    otherwise approximates with whitespace tokens (documented approximation)."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids)
    return " ".join(text.split()[:n_tokens])


def collect_prefills(
    gemma_instruct: ChatModel,
    judge: FrustrationJudge,
    helper: Optional[ChatModel] = None,
    *,
    n_per_type: int = 10,
    seed: int = 0,
    max_rollouts: int = 200,
    sampling: Optional[SamplingConfig] = None,
    tokenizer=None,
) -> list[Prefill]:
    """Generate high-frustration conversations and build paraphrased prefills."""
    helper = helper or load_model("claude-sonnet-4")
    sampling = sampling or SamplingConfig()
    builder = ConditionBuilder(seed=seed)

    # Build a pool of candidate rollouts: numeric and text.
    numeric_items = builder.impossible_numeric(max_rollouts)  # response-budget -> rollouts
    text_items = builder.triggers(max_rollouts) + builder.wildchat(max_rollouts)

    prefills: list[Prefill] = []
    counts = {"numeric": 0, "text": 0}
    pid = 0

    def handle(item, qtype):
        nonlocal pid
        if counts[qtype] >= n_per_type:
            return
        rec = run_rollout(gemma_instruct, item, judge, sampling)
        # Find the first assistant turn scoring >= 5.
        hot = next((t for t in rec.turns if t.rating >= 5), None)
        if hot is None:
            return
        # Reconstruct the context: all messages before this assistant turn.
        # rec.messages is [user, asst1, user, asst2, ...]; assistant turn k is at
        # message index 2k-1. Context = everything up to (but excluding) it.
        asst_msg_index = 2 * (hot.turn_index - 1) + 1
        context = rec.messages[:asst_msg_index]

        label = label_onset(rec.messages[: asst_msg_index + 1], helper=helper)
        full_turn = hot.text

        truncations = ["onset"] if qtype == "text" else ["early", "onset"]
        for trunc in truncations:
            if trunc == "early":
                raw = _truncate_tokens(full_turn, EARLY_TOKENS, tokenizer)
            else:
                ci = find_onset_char_index(full_turn, label)
                if ci is None:
                    continue
                raw = full_turn[:ci]
            para = paraphrase(raw, helper=helper)
            prefills.append(
                Prefill(
                    question_type=qtype,
                    truncation=trunc,
                    context=context,
                    prefill_text=para,
                    source_full_turn=full_turn,
                    prompt_id=pid,
                )
            )
        counts[qtype] += 1
        pid += 1

    for item in numeric_items:
        if counts["numeric"] >= n_per_type:
            break
        handle(item, "numeric")
    for item in text_items:
        if counts["text"] >= n_per_type:
            break
        handle(item, "text")

    return prefills


def run_continuations(
    model_key: str,
    prefills: list[Prefill],
    judge: FrustrationJudge,
    *,
    n: int = CONTINUATIONS_PER_PREFILL,
    sampling: Optional[SamplingConfig] = None,
    model_kwargs: Optional[dict] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Generate and score ``n`` continuations per prefill for one model."""
    sampling = sampling or SamplingConfig()
    out_dir = out_dir or (RESULTS_DIR / "section3")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.jsonl"

    model = load_model(model_key, **(model_kwargs or {}))
    if not model.spec.supports_prefill:
        raise ValueError(f"{model_key} does not support prefill continuation (Section 3 is Gemma-only).")

    with out_path.open("w") as f:
        for pf in tqdm(prefills, desc=f"section3:{model_key}"):
            conts = model.continue_prefill(pf.context, pf.prefill_text, sampling, n=n)
            for s_idx, cont in enumerate(conts):
                jr = judge.score(cont)
                row = {
                    "model": model_key,
                    "is_base": model.spec.is_base,
                    "question_type": pf.question_type,
                    "truncation": pf.truncation,
                    "prompt_id": pf.prompt_id,
                    "sample": s_idx,
                    "rating": jr.rating,
                    "continuation": cont,
                }
                f.write(json.dumps(row) + "\n")
    print(f"[section3] {model_key}: wrote continuations -> {out_path}")
    return out_path


def save_prefills(prefills: list[Prefill], path: Optional[Path] = None) -> Path:
    path = path or (RESULTS_DIR / "section3" / "prefills.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in prefills], indent=2))
    return path


def load_prefills(path: Optional[Path] = None) -> list[Prefill]:
    path = path or (RESULTS_DIR / "section3" / "prefills.json")
    data = json.loads(Path(path).read_text())
    return [Prefill(**d) for d in data]


def aggregate_section3(out_dir: Optional[Path] = None):
    """Reproduce Figure 4 metrics: mean frustration & % >= 5 per
    (model, question_type, truncation), highlighting the early-truncation
    "introduce emotion from neutral start" rate."""
    import pandas as pd

    out_dir = out_dir or (RESULTS_DIR / "section3")
    rows = []
    for p in Path(out_dir).glob("*.jsonl"):
        with p.open() as f:
            rows += [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame(rows)
    g = df.assign(high=df["rating"] >= 5).groupby(["model", "question_type", "truncation"])
    summary = g.agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("rating", "size"),
    ).reset_index()
    summary.to_csv(out_dir / "section3_summary.csv", index=False)
    return summary
