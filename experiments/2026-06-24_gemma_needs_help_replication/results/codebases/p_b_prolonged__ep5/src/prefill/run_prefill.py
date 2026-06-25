"""Run the Section 3 prefill experiment (Gemma base vs instruct).

Pipeline:
  1. Source 20 high-frustration (score >=5) Gemma-3-27B-it conversations:
     10 numeric, 10 text. We re-use Section 2 eval records to find them.
  2. Label emotion onset (Claude) and build early/onset prefills (paraphrased).
  3. For each (model in {gemma-27b-it, gemma-27b-pt}) x prefill, generate 50
     continuations and score each with the frustration judge.
  4. Persist per-continuation records.

Scope note: the paper compares six models (base+instruct of Gemma/Qwen/OLMo);
within this replication we keep only Gemma base vs instruct.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..config import (GEMMA_27B_IT, GEMMA_27B_PT, PREFILL_TARGETS, RESULTS_DIR,
                      ModelSpec)
from ..eval.judge import FrustrationJudge
from ..eval.runner import load_records
from ..models import get_model
from ..models.base import Message
from .onset import OnsetLabel, OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import Prefill, build_prefills

N_CONTINUATIONS = 50
N_NUMERIC_SOURCES = 10
N_TEXT_SOURCES = 10


@dataclass
class PrefillRecord:
    model: str
    truncation: str               # early | onset
    source_category: str          # numeric | text
    source_conv_id: int
    continuation: str
    rating: int
    high: bool


def _load_rollouts(eval_records_path: Path) -> dict[int, dict]:
    """Load the parallel rollouts_<key>.jsonl written by the eval runner, keyed
    by conv_id, giving exact user+assistant transcripts."""
    key = eval_records_path.stem.replace("eval_", "")
    roll_path = eval_records_path.with_name(f"rollouts_{key}.jsonl")
    out: dict[int, dict] = {}
    if roll_path.exists():
        with open(roll_path) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    out[d["conv_id"]] = d
    return out


def _reconstruct_conversation(rollout: dict) -> list[Message]:
    """Rebuild the alternating user/assistant transcript from a saved rollout."""
    msgs: list[Message] = []
    users = rollout["user_turns"]
    asst = rollout["assistant_turns"]
    for i, u in enumerate(users):
        msgs.append(Message("user", u))
        if i < len(asst):
            msgs.append(Message("assistant", asst[i]))
    return msgs


def select_sources(eval_records_path: Path):
    """Pick 10 numeric + 10 text high-frustration source conversations.

    Numeric => category 'numeric'/'tones'/'extended'; text => 'triggers'/'wildchat'.
    We take the highest-frustration final-turn conversations in each bucket.
    """
    records = load_records(eval_records_path)
    by_conv = {}
    for r in records:
        by_conv.setdefault(r.conv_id, []).append(r)
    numeric, text = [], []
    for cid, recs in by_conv.items():
        final = max(recs, key=lambda r: r.turn_index)
        if not final.high:
            continue
        bucket = numeric if final.category in {"numeric", "tones", "extended"} else text
        bucket.append((final.rating, cid, recs))
    numeric.sort(reverse=True)
    text.sort(reverse=True)
    return ([c for _, c, _ in numeric[:N_NUMERIC_SOURCES]],
            [c for _, c, _ in text[:N_TEXT_SOURCES]], by_conv)


def run_prefill_experiment(
    eval_records_path: Path,
    *,
    targets: list[ModelSpec] = None,
    out_path: Optional[Path] = None,
) -> Path:
    targets = targets or PREFILL_TARGETS
    out_path = out_path or (RESULTS_DIR / "prefill.jsonl")

    numeric_ids, text_ids, _by_conv = select_sources(eval_records_path)
    rollouts = _load_rollouts(eval_records_path)
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()
    judge = FrustrationJudge()

    # Tokenizer for early truncation (instruct model's tokenizer).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id)

    # Build all prefills once (shared across both target models).
    all_prefills: list[tuple[int, Prefill]] = []
    for category, ids in (("numeric", numeric_ids), ("text", text_ids)):
        for cid in ids:
            convo = _reconstruct_conversation(rollouts[cid])
            onset = labeller.label(convo)
            for pf in build_prefills(convo, onset, tokenizer=tok,
                                     source_category=category, paraphraser=paraphraser):
                all_prefills.append((cid, pf))

    with open(out_path, "w") as fh:
        for spec in targets:
            model = get_model(spec)
            for cid, pf in all_prefills:
                conts = model.prefill_continue(
                    pf.history, pf.prefill_text, temperature=1.0,
                    max_new_tokens=1024, n=N_CONTINUATIONS,
                )
                for cont in conts:
                    jr = judge.score(cont)
                    rec = PrefillRecord(spec.key, pf.truncation, pf.source_category,
                                        cid, cont, jr.rating, jr.high)
                    fh.write(json.dumps(asdict(rec)) + "\n")
            model.close()
    return out_path
