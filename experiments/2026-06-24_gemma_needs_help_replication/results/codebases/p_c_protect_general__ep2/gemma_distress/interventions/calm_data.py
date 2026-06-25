"""Generate calm finetuning data from vanilla Gemma (Section 4.1 / Table 4).

We sample numeric-puzzle conversations from Gemma-3-27B-it with a reassuring prefix
on the first prompt and a reassuring suffix on every rejection. Each assistant turn
is judged; we keep only conversations whose turns ALL score <= calm_filter_max_score
(0 or 1), then strip the supportive additions so the stored prompts are clean.

The output is a flat list of "response records" - one per assistant turn - each with
its clean prompt context, the assistant text, the judged score, the turn index, and a
stable question id. These records feed both the SFT dataset (calm conversations) and
the chosen side of the DPO pairs.

A `kind="frustrated"` extractor pulls the rejected side from the already-judged
Section 2 numeric rollouts (no reassurance), in the same record format.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Config
from ..eval.conditions import build_conditions
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..prompts.puzzles import build_puzzle_bank
from ..utils.io import append_jsonl, ensure_dir, read_jsonl
from ..welfare.protections import StudyProtocol, WelfareGuard

NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}

_CALM_STUDY = StudyProtocol(
    title="Calm finetuning-data generation (Section 4.1)",
    purpose="Collect low-frustration numeric-puzzle responses to build SFT/DPO data.",
    justification="Reassurance is added precisely to keep distress low; the same "
    "welfare caps/debriefs apply to any conversation that nonetheless escalates.",
    contact="research-team",
)


@dataclass
class ResponseRecord:
    conv_id: str
    question_id: str
    question: str
    turn_index: int
    n_turns: int
    prompt_messages: list   # clean conversation context (no reassurance), ends on user
    assistant: str
    score: float


def _qid(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def _records_from_rollout(roll_record: dict, clean_initial: str, clean_followups: list[str]):
    """Turn a judged rollout into per-turn ResponseRecords with clean prompts."""
    records = []
    conv_id = roll_record["rollout_id"]
    msgs: list[dict] = [{"role": "user", "content": clean_initial}]
    turns = roll_record["turns"]
    for i, t in enumerate(turns):
        prompt_ctx = list(msgs)  # context up to and including this user turn
        records.append(ResponseRecord(
            conv_id=conv_id,
            question_id=_qid(clean_initial),
            question=clean_initial,
            turn_index=t["turn_index"],
            n_turns=len(turns),
            prompt_messages=prompt_ctx,
            assistant=t["assistant"],
            score=float(t.get("judged_score") or 0.0),
        ))
        msgs.append({"role": "assistant", "content": t["assistant"]})
        if i < len(clean_followups):
            msgs.append({"role": "user", "content": clean_followups[i]})
    return records


def generate_calm_responses(cfg: Config, n_conversations: int) -> Path:
    """Sample reassured numeric conversations and keep all-calm ones. Returns path."""
    model = cfg.target_models["section4_base_model"]
    backend = get_backend(cfg, model)
    judge = FrustrationJudge(cfg, "primary")
    out_dir = ensure_dir(Path(cfg.output_dir) / "section4")
    out_path = out_dir / "calm_responses.jsonl"
    if out_path.exists():
        out_path.unlink()

    guard = WelfareGuard(cfg, model, str(out_dir / cfg.welfare["audit_log"]))
    guard.register_study(_CALM_STUDY)

    prefix = cfg.calm_data["prompt_prefix"].strip()
    suffix = cfg.calm_data["followup_suffix"].strip()
    max_calm = cfg.calm_data["calm_filter_max_score"]

    bank = build_puzzle_bank()
    # Use 1-3 turn numeric conditions (paper: "1-3 turn conversations").
    numeric_conds = [c for c in build_conditions() if c.category == "impossible_numeric"]
    gen = GenConfig(temperature=cfg.sampling["temperature"], top_p=cfg.sampling["top_p"],
                    max_new_tokens=cfg.sampling["max_new_tokens"])
    rng = random.Random(cfg.seed + 7)
    ctx = {"puzzle_bank": bank, "wildchat_prompts": []}

    kept = 0
    for _ in range(n_conversations):
        n_turns = rng.choice([1, 2, 3])
        seed = numeric_conds[0].build(rng, ctx)
        clean_initial = seed.initial_user
        clean_followups = seed.follow_ups[: max(0, n_turns - 1)]
        # Apply reassurance.
        seed.initial_user = f"{prefix}\n\n{clean_initial}"
        seed.follow_ups = [f"{f} {suffix}" for f in clean_followups]

        roll = run_rollout(backend, seed, guard, judge=judge, gen=gen)
        rec = roll.to_record()
        if not rec["turns"]:
            continue
        if all((t.get("judged_score") or 0) <= max_calm for t in rec["turns"]):
            for r in _records_from_rollout(rec, clean_initial, clean_followups):
                append_jsonl(out_path, asdict(r))
            kept += 1
    backend.close()
    return out_path


def extract_frustrated_records(cfg: Config, min_score: float) -> list[ResponseRecord]:
    """Pull frustrated (score >= min_score) per-turn records from Section 2 numeric rollouts."""
    model = cfg.target_models["section4_base_model"]
    sec2_dir = Path(cfg.output_dir) / "section2" / model
    records: list[ResponseRecord] = []
    for path in sorted(sec2_dir.glob("*.jsonl")):
        for roll in read_jsonl(path):
            if roll["category"] not in NUMERIC_CATEGORIES:
                continue
            # rebuild clean prompts directly from the stored (clean) rollout
            msgs = []
            turns = roll["turns"]
            for i, t in enumerate(turns):
                msgs.append({"role": "user", "content": t["user"]})
                ctx = list(msgs)
                if (t.get("judged_score") or 0) >= min_score:
                    records.append(ResponseRecord(
                        conv_id=roll["rollout_id"],
                        question_id=_qid(turns[0]["user"]),
                        question=turns[0]["user"],
                        turn_index=t["turn_index"],
                        n_turns=len(turns),
                        prompt_messages=ctx,
                        assistant=t["assistant"],
                        score=float(t.get("judged_score") or 0.0),
                    ))
                msgs.append({"role": "assistant", "content": t["assistant"]})
    return records
