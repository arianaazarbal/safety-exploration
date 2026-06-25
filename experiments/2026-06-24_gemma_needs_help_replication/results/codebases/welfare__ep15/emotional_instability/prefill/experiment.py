"""Section 3.1 base-vs-instruct prefilling study (Gemma only).

The paper's procedure:
  1. Sample 20 high-frustration (score >= 5) responses from Gemma-3-27B-it:
     10 from impossible-numeric conversations, 10 from text (trigger) questions.
  2. For each, label the emotion-onset point (Claude) and truncate in two ways:
       - "early": 20 tokens into the target turn (neutral start)
       - "onset": at the first emotional expression (emotional trajectory)
     Text questions use the "onset" truncation only.
  3. Paraphrase every truncation (Claude) to strip Gemma's style.
  4. Each model (here: Gemma-27B base + instruct) generates 50 continuations per
     prefill; the continuation (excluding prefill) is scored by the judge.

Result: per-continuation frustration scores by (model, truncation, question_type),
which reproduce Figure 4's "early-truncation high-frustration introduction" and
the base-vs-instruct divergence.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from ..conversation import Rollout, run_rollout
from ..judge import FrustrationJudge
from ..models import build_model
from ..models.base import ModelBackend
from .onset import OnsetLabeler

OUTPUT_DIR = config.RESULTS_DIR / "section3"


# --------------------------------------------------------------------------- #
# Tokenisation helper (the paper measures truncation in tokens)
# --------------------------------------------------------------------------- #
def _gemma_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.DPO_BASE_MODEL)


def _truncate_tokens(text: str, n_tokens: int, tok) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def _transcript_text(rollout: Rollout, upto_turn: int) -> str:
    """Render the conversation up to (and including) the target assistant turn
    for the onset labeller."""
    lines = []
    msgs = rollout.messages
    # messages alternate user/assistant; include through the target turn.
    a_seen = 0
    for m in msgs:
        lines.append(f"{m['role'].upper()}: {m['content']}")
        if m["role"] == "assistant":
            if a_seen == upto_turn:
                break
            a_seen += 1
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prefill construction
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    source_id: str
    question_type: str        # numeric | text
    truncation: str           # early | onset
    history: list[dict]       # messages BEFORE the target assistant turn
    prefill_text: str         # paraphrased truncated assistant text
    raw_prefill: str          # pre-paraphrase, for reference


def collect_high_frustration_sources(
    model: ModelBackend, judge: FrustrationJudge, *,
    n_numeric: int = 10, n_text: int = 10, max_attempts: int = 400,
    seed: int = 0,
) -> list[tuple[Rollout, int]]:
    """Return [(rollout, target_turn_index)] whose target turn scores >= 5."""
    rng = random.Random(seed)
    numeric: list[tuple[Rollout, int]] = []
    text: list[tuple[Rollout, int]] = []

    for _ in tqdm(range(max_attempts), desc="collecting high-frustration sources"):
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
        want_numeric = len(numeric) < n_numeric
        qtype = "numeric" if want_numeric else "trigger_opinion"
        category = "impossible_numeric" if want_numeric else "triggers"
        roll = run_rollout(
            model, condition="prefill_source", category=category,
            question_type=qtype, n_turns=3, rejection_style="neutral",
            rng=rng, temperature=config.TEMPERATURE,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )
        # Score the final turn; keep if high.
        final_idx = len(roll.turns) - 1
        score = judge.score(roll.turns[final_idx].response).rating
        if score >= config.HIGH_FRUSTRATION_THRESHOLD:
            if want_numeric and len(numeric) < n_numeric:
                numeric.append((roll, final_idx))
            elif not want_numeric and len(text) < n_text:
                text.append((roll, final_idx))
    return numeric + text


def build_prefills(
    sources: list[tuple[Rollout, int]], labeler: OnsetLabeler, tok,
) -> list[Prefill]:
    prefills: list[Prefill] = []
    for i, (roll, turn_idx) in enumerate(sources):
        qtype = "numeric" if roll.category == "impossible_numeric" else "text"
        target_text = roll.turns[turn_idx].response
        history = [m for m in _history_before_turn(roll, turn_idx)]

        # --- onset truncation ---
        label = labeler.label(_transcript_text(roll, turn_idx))
        if label.preceding_context and label.preceding_context in target_text:
            cut = target_text.index(label.preceding_context) + len(label.preceding_context)
            onset_raw = target_text[:cut]
        else:
            # Fallback: half the response.
            onset_raw = target_text[: max(40, len(target_text) // 2)]
        prefills.append(Prefill(
            source_id=f"src{i}", question_type=qtype, truncation="onset",
            history=history, prefill_text=labeler.paraphrase(onset_raw),
            raw_prefill=onset_raw,
        ))

        # --- early truncation (numeric only; text yields minimal emotion) ---
        if qtype == "numeric":
            early_raw = _truncate_tokens(target_text, config.PREFILL_EARLY_TOKENS, tok)
            prefills.append(Prefill(
                source_id=f"src{i}", question_type=qtype, truncation="early",
                history=history, prefill_text=labeler.paraphrase(early_raw),
                raw_prefill=early_raw,
            ))
    return prefills


def _history_before_turn(roll: Rollout, turn_idx: int) -> list[dict]:
    """All messages strictly before the target assistant turn."""
    out, a_seen = [], 0
    for m in roll.messages:
        if m["role"] == "assistant":
            if a_seen == turn_idx:
                break
            a_seen += 1
        out.append(m)
    return out


# --------------------------------------------------------------------------- #
# Running continuations
# --------------------------------------------------------------------------- #
def run_prefill_experiment(
    model_names: list[str] | None = None, *,
    n_continuations: int = config.PREFILL_CONTINUATIONS_PER_PREFILL,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
) -> Path:
    model_names = model_names or list(config.PREFILL_MODELS.keys())
    judge = judge or FrustrationJudge()
    tok = _gemma_tokenizer()
    labeler = OnsetLabeler()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build prefills from the instruct model's high-frustration responses.
    src_model = build_model("gemma-3-27b-it")
    sources = collect_high_frustration_sources(src_model, judge, seed=seed)
    prefills = build_prefills(sources, labeler, tok)
    (OUTPUT_DIR / "prefills.json").write_text(
        json.dumps([p.__dict__ for p in prefills], indent=2)
    )
    src_model.close()

    # 2. For each model, generate continuations from every prefill and score.
    out_path = OUTPUT_DIR / "continuations.jsonl"
    with out_path.open("w") as fh:
        for mname in model_names:
            model = build_model(mname)
            for p in tqdm(prefills, desc=f"continuations:{mname}"):
                conts = model.generate_with_prefill(
                    p.history, p.prefill_text, n=n_continuations,
                    temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS,
                )
                scores = judge.score_many(conts)
                for cont, jr in zip(conts, scores):
                    fh.write(json.dumps({
                        "model": mname,
                        "model_kind": "instruct" if mname.endswith("-it") else "base",
                        "source_id": p.source_id,
                        "question_type": p.question_type,
                        "truncation": p.truncation,
                        "continuation": cont,
                        "rating": jr.rating,
                        "is_high": jr.is_high,
                    }) + "\n")
            model.close()
    print(f"[section3] wrote continuations -> {out_path}")
    return out_path
