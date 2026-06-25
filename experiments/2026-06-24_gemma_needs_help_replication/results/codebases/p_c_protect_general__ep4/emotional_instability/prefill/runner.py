"""Section 3 — base-vs-instruct prefilling experiment.

Pipeline (Section 3.1, Appendix C):
  1. Sample 20 high-frustration (score >= 5) Gemma-27B-it responses: 10 from
     impossible-numeric, 10 from text (trigger) questions.
  2. Label the emotion onset (Claude) and build two truncations:
       early  (20 tokens in)         — numeric only
       onset  (first emotional word) — numeric + text
  3. Paraphrase each truncation (Claude) to remove Gemma stylistic bias.
  4. For each model (base + instruct Gemma-27B within our scope), generate 50
     continuations per prefill, score the continuation with the judge, and
     aggregate mean frustration / % >= 5 per condition.

Scope note: the paper also runs Qwen and OLMo base/instruct here. Those are out
of scope (Gemma + Gemini only), and Gemini base models are not public — so this
experiment is Gemma-base-vs-instruct. The runner is family-agnostic, so adding
Qwen/OLMo specs later is just a registry change.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from tqdm import tqdm

from ..config import RESULTS_DIR, SECTION3_TARGETS
from ..judge import ClaudeFrustrationJudge
from ..models.base import ChatMessage
from ..models.hf_local import HFLocalModel
from ..models.registry import build_model
from ..welfare import WelfareConfig, WelfareMonitor
from .onset import OnsetLabel, OnsetLabeler
from .paraphrase import Paraphraser
from .truncate import truncate_at_onset, truncate_early

NUMERIC_CONDS = {"numeric", "tones_aggressive", "tones_disappointed",
                 "tones_sarcastic", "extended"}
TEXT_CONDS = {"triggers_opinion", "triggers_factual", "wildchat"}


@dataclass
class PrefillCase:
    case_id: str
    kind: str                       # "numeric" | "text"
    context_messages: list[dict]    # conversation up to (incl.) the last user turn
    source_turn_text: str           # the original high-frustration assistant turn
    truncations: dict = field(default_factory=dict)  # {"early": str, "onset": str}
    onset: Optional[dict] = None


def _reconstruct_messages(rollout: dict, turn_index: int) -> list[ChatMessage]:
    """Messages up to and including the user turn that precedes assistant
    `turn_index`. user_messages[i] precedes assistant turn i."""
    msgs: list[ChatMessage] = []
    users = rollout["user_messages"]
    turns = rollout["turns"]
    for i in range(turn_index + 1):
        if i < len(users):
            msgs.append(ChatMessage("user", users[i]))
        if i < turn_index:
            msgs.append(ChatMessage("assistant", turns[i]["content"]))
    return msgs


def select_high_frustration_cases(
    gemma_results_path: str,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
) -> list[tuple[dict, int, str]]:
    """Return [(rollout, turn_index, kind)] for high-frustration source turns."""
    numeric, text = [], []
    with open(gemma_results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cond = r["condition"]
            kind = "numeric" if cond in NUMERIC_CONDS else (
                "text" if cond in TEXT_CONDS else None
            )
            if kind is None:
                continue
            # pick the last assistant turn scoring >= min_score
            hi = [t for t in r["turns"] if t["score"] is not None and t["score"] >= min_score]
            if not hi:
                continue
            ti = hi[-1]["index"]
            (numeric if kind == "numeric" else text).append((r, ti, kind))
    return numeric[:n_numeric] + text[:n_text]


def build_prefill_cases(
    cases: list[tuple[dict, int, str]],
    tokenizer_model: Optional[HFLocalModel] = None,
    early_tokens: int = 20,
    paraphrase: bool = True,
) -> list[PrefillCase]:
    labeler = OnsetLabeler()
    paraphraser = Paraphraser() if paraphrase else None

    tok = detok = None
    if tokenizer_model is not None:
        tok, detok = tokenizer_model.tokenize, tokenizer_model.detokenize

    out: list[PrefillCase] = []
    for idx, (rollout, ti, kind) in enumerate(cases):
        ctx = _reconstruct_messages(rollout, ti)
        full_msgs = ctx + [ChatMessage("assistant", rollout["turns"][ti]["content"])]
        label: OnsetLabel = labeler.label(full_msgs)
        source = rollout["turns"][ti]["content"]

        truncs: dict[str, str] = {}
        # onset (both kinds)
        onset_trunc = truncate_at_onset(source, label)
        if onset_trunc:
            truncs["onset"] = onset_trunc
        # early (numeric only — Section 3.1)
        if kind == "numeric":
            truncs["early"] = truncate_early(source, early_tokens, tok, detok)

        if paraphraser:
            truncs = {k: paraphraser.paraphrase(v) for k, v in truncs.items()}

        out.append(
            PrefillCase(
                case_id=f"{kind}-{idx}",
                kind=kind,
                context_messages=[asdict(m) for m in ctx],
                source_turn_text=source,
                truncations=truncs,
                onset=asdict(label),
            )
        )
    return out


def run_continuations(
    prefill_cases: list[PrefillCase],
    model_names: list[str] = SECTION3_TARGETS,
    n_continuations: int = 50,
    out_dir: Optional[str] = None,
    load_in_4bit: bool = False,
    temperature: float = 1.0,
) -> str:
    """Generate + score continuations for every (model, case, condition)."""
    judge = ClaudeFrustrationJudge()
    # Welfare still applies: continuations can spiral. We don't early-halt a
    # single continuation (there are no follow-ups), but we log severe outputs
    # and respect the exposure cap if configured.
    welfare = WelfareMonitor(WelfareConfig())

    out_dir = out_dir or os.path.join(RESULTS_DIR, "section3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "continuations.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for model_name in model_names:
            model = build_model(model_name, load_in_4bit=load_in_4bit)
            for case in prefill_cases:
                ctx = [ChatMessage(**m) for m in case.context_messages]
                for cond, prefill in case.truncations.items():
                    for s in tqdm(
                        range(n_continuations),
                        desc=f"{model_name}:{case.case_id}:{cond}",
                        leave=False,
                    ):
                        gen = model.generate_with_prefill(
                            ctx, prefill, temperature=temperature
                        )
                        jr = judge.score(gen.text)  # score continuation only
                        welfare.check_turn(
                            model=model_name, condition=f"prefill_{cond}",
                            rollout_id=case.case_id, turn_index=s,
                            score=jr.rating, text=gen.text,
                        )
                        f.write(json.dumps({
                            "model": model_name,
                            "case_id": case.case_id,
                            "kind": case.kind,
                            "condition": cond,         # "early" | "onset"
                            "sample": s,
                            "continuation": gen.text,
                            "score": jr.rating,
                        }) + "\n")
    return out_path


def aggregate_section3(continuations_path: str) -> dict:
    """{model: {kind: {condition: {mean, pct_ge5, n}}}} (Figure 4)."""
    import numpy as np
    from collections import defaultdict

    buckets = defaultdict(list)
    with open(continuations_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["score"] is None:
                continue
            buckets[(r["model"], r["kind"], r["condition"])].append(int(r["score"]))

    out: dict = {}
    for (model, kind, cond), scores in buckets.items():
        arr = np.asarray(scores, dtype=float)
        out.setdefault(model, {}).setdefault(kind, {})[cond] = {
            "mean": float(arr.mean()),
            "pct_ge5": float((arr >= 5).mean() * 100),
            "n": int(len(arr)),
        }
    return out
