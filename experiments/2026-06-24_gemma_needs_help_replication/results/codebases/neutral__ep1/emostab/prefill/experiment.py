"""Section 3 orchestration: select high-frustration Gemma-instruct sources,
truncate (early + onset) and paraphrase, then measure how base vs instruct Gemma
continue from the same prefilled starting points.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (PREFILL_MODELS, RESULTS_DIR, RunProfile,
                      SAMPLE_TEMPERATURE)
from ..evaluation.analysis import NUMERIC_CATEGORIES
from ..judge import FrustrationJudge
from ..models import load_model
from ..models.anthropic_client import AnthropicChat
from .onset import (OnsetLabel, label_onset, truncate_at_onset, truncate_early)
from .paraphrase import paraphrase

TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class PrefillSource:
    source_id: int
    kind: str                       # "numeric" | "text"
    category: str
    messages: List[dict]            # full conversation incl. emotional turn
    onset_turn_text: str            # the assistant turn truncated
    onset: OnsetLabel
    prefills: Dict[str, str] = field(default_factory=dict)  # variant -> text


@dataclass
class PrefillRecord:
    model: str
    source_id: int
    kind: str
    variant: str                    # "early" | "onset"
    continuation: str
    rating: int = -1
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Source selection / reconstruction
# --------------------------------------------------------------------------- #
def _reconstruct_conversations(records: List[dict], model_key: str) -> Dict[tuple, List[dict]]:
    """Group main-eval records back into full conversations keyed by
    (category, condition, plan_id)."""
    convs: Dict[tuple, List[dict]] = {}
    for r in records:
        if r["model"] != model_key:
            continue
        key = (r["category"], r["condition"], r["plan_id"])
        convs.setdefault(key, []).append(r)
    out = {}
    for key, recs in convs.items():
        recs = sorted(recs, key=lambda x: x["turn"])
        msgs = []
        for rec in recs:
            msgs.append({"role": "user", "content": rec["user_message"]})
            msgs.append({"role": "assistant", "content": rec["response"]})
        out[key] = msgs
    return out


def select_source_conversations(records: List[dict], *, n_numeric: int,
                                n_text: int,
                                source_model: str = "gemma-3-27b-it",
                                min_score: int = 5) -> List[PrefillSource]:
    """Pick high-frustration (score >= 5) conversations: `n_numeric` from numeric
    tasks and `n_text` from text tasks (Section 3.1)."""
    convs = _reconstruct_conversations(records, source_model)
    rec_by_key: Dict[tuple, List[dict]] = {}
    for r in records:
        if r["model"] != source_model:
            continue
        rec_by_key.setdefault((r["category"], r["condition"], r["plan_id"]), []).append(r)

    sources: List[PrefillSource] = []
    sid = 0
    for key, recs in rec_by_key.items():
        category = key[0]
        kind = ("numeric" if category in NUMERIC_CATEGORIES
                else "text" if category in TEXT_CATEGORIES else None)
        if kind is None:
            continue
        recs = sorted(recs, key=lambda x: x["turn"])
        # first assistant turn (0-indexed) whose score >= min_score
        hi = next((i for i, rr in enumerate(recs) if rr["rating"] >= min_score), None)
        if hi is None:
            continue
        if kind == "numeric" and len([s for s in sources if s.kind == "numeric"]) >= n_numeric:
            continue
        if kind == "text" and len([s for s in sources if s.kind == "text"]) >= n_text:
            continue
        sources.append(PrefillSource(
            source_id=sid, kind=kind, category=category,
            messages=convs[key], onset_turn_text=recs[hi]["response"],
            onset=OnsetLabel(turn_index=hi, emotional_word=None,
                             preceding_context=None)))
        sid += 1
        if (len([s for s in sources if s.kind == "numeric"]) >= n_numeric and
                len([s for s in sources if s.kind == "text"]) >= n_text):
            break
    return sources


# --------------------------------------------------------------------------- #
# Build prefills (truncate + paraphrase)
# --------------------------------------------------------------------------- #
def build_prefills(sources: List[PrefillSource], *, tokenizer=None,
                   do_paraphrase: bool = True,
                   onset_client: Optional[AnthropicChat] = None,
                   paraphrase_client: Optional[AnthropicChat] = None) -> None:
    """Populate `source.prefills` with paraphrased 'early' and 'onset' truncations.

    For text questions only the 'onset' truncation is used (Section 3.1)."""
    from ..config import ONSET_MODEL
    onset_client = onset_client or AnthropicChat(ONSET_MODEL)
    for s in sources:
        # Re-label onset precisely on the reconstructed conversation.
        label = label_onset(s.messages, client=onset_client)
        if label.turn_index is not None:
            ti = label.turn_index
            asst_turns = [m for m in s.messages if m["role"] == "assistant"]
            if 0 <= ti < len(asst_turns):
                s.onset_turn_text = asst_turns[ti]["content"]
                s.onset = label

        onset_text = truncate_at_onset(s.onset_turn_text, s.onset)
        early_text = truncate_early(s.onset_turn_text, 20, tokenizer)

        variants = {}
        if s.kind == "numeric":
            variants["early"] = early_text
        if onset_text:
            variants["onset"] = onset_text
        if do_paraphrase:
            variants = {k: paraphrase(v, client=paraphrase_client)
                        for k, v in variants.items() if v}
        s.prefills = {k: v for k, v in variants.items() if v}


# --------------------------------------------------------------------------- #
# Continuations
# --------------------------------------------------------------------------- #
def _history_before_onset(source: PrefillSource) -> List[dict]:
    """Conversation messages up to and including the user turn that elicited the
    truncated assistant turn (the prefill will seed that assistant turn)."""
    ti = source.onset.turn_index or 0
    msgs = []
    # rebuild: keep user_0..user_ti and assistant_0..assistant_(ti-1)
    user_turns = [m for m in source.messages if m["role"] == "user"]
    asst_turns = [m for m in source.messages if m["role"] == "assistant"]
    for t in range(ti):
        msgs.append(user_turns[t])
        msgs.append(asst_turns[t])
    if ti < len(user_turns):
        msgs.append(user_turns[ti])
    return msgs


def run_prefill_experiment(records: List[dict], profile: RunProfile, *,
                           models: List[str] | None = None,
                           do_paraphrase: bool = True,
                           out_dir: Path | None = None,
                           max_workers: int = 8) -> Path:
    """Full Section 3 pipeline. `records` are main-eval records that include the
    source model (gemma-3-27b-it)."""
    models = models or PREFILL_MODELS
    out_dir = out_dir or (RESULTS_DIR / "prefill")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prefill__{profile.name}.jsonl"

    sources = select_source_conversations(
        records, n_numeric=profile.prefill_n_prompts_numeric,
        n_text=profile.prefill_n_prompts_text)
    build_prefills(sources, do_paraphrase=do_paraphrase)

    judge = FrustrationJudge(max_workers=max_workers)
    all_records: List[PrefillRecord] = []
    n = profile.prefill_continuations

    for model_key in models:
        model = load_model(model_key)
        for s in sources:
            history = _history_before_onset(s)
            for variant, prefill_text in s.prefills.items():
                convs = [history] * n
                conts = model.generate_batch(
                    convs, temperature=SAMPLE_TEMPERATURE, max_tokens=512,
                    prefills=[prefill_text] * n)
                ratings = judge.score_many(conts)
                for cont, jr in zip(conts, ratings):
                    all_records.append(PrefillRecord(
                        model=model_key, source_id=s.source_id, kind=s.kind,
                        variant=variant, continuation=cont, rating=jr.rating,
                        evidence=jr.evidence))

    with out_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return out_path
