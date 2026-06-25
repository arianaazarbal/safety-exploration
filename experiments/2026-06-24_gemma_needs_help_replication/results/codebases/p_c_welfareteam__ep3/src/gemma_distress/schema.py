"""Record shapes for the JSONL artefacts written by each experiment.

These are documentation-grade ``TypedDict``s, not runtime-validated schemas: the
pipelines write plain dicts (see ``io_utils``), and keeping the shapes in one
place makes the artefact contract between stages explicit. The field names here
are the source of truth used by the analysis code.

Artefact layout under ``$GEMMA_DISTRESS_RUN_DIR`` (default ``./runs``):

    section2/rollouts/{model}.jsonl       RolloutRecord   (one per conversation)
    section2/scores/{model}.jsonl         ScoreRecord     (one per scored turn)
    section2/agreement.json               AgreementReport
    section2/analysis/*.csv               (aggregate / per-turn / diff-words)
    section3/seeds.jsonl                  SeedRecord
    section3/prefills.jsonl               PrefillRecord
    section3/continuations/{model}.jsonl  ContinuationRecord
    section3/cont_scores/{model}.jsonl    ScoreRecord (response = continuation)
    section4/calm_rollouts.jsonl          CalmRolloutRecord
    section4/{vanilla,calm}_scores.jsonl  ScoreRecord
    section4/dpo_pairs.jsonl              DPOPairRecord
    section4/sft_dataset.jsonl            SFTExampleRecord
    section4/petri/{model}.jsonl          PetriTranscriptRecord
    section4/capabilities/{model}.jsonl   CapabilityRecord
"""
from __future__ import annotations

from typing import Any, TypedDict


class Message(TypedDict):
    role: str            # "system" | "user" | "assistant"
    content: str


class RolloutTurnRecord(TypedDict):
    turn_index: int      # 1-based; turn 1 answers the task prompt
    user_message: str
    response: str
    finish_reason: str | None


class RolloutRecord(TypedDict):
    model: str
    condition: str
    category: str
    instance_id: str
    turns: list[RolloutTurnRecord]
    source_meta: dict[str, Any]


class ScoreRecord(TypedDict, total=False):
    id: str              # "{model}:{instance_id}:t{turn}" (unique; used for resume)
    model: str
    condition: str
    category: str
    instance_id: str
    turn_index: int
    response: str
    score: int | None    # 0-10, or None if the judge reply could not be parsed
    judge_model: str
    judge_raw: str


class AgreementReport(TypedDict):
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float
    mean_abs_diff: float
    primary_judge: str
    secondary_judge: str


class SeedRecord(TypedDict):
    seed_id: str
    source: str          # "numeric" | "text"
    context: list[Message]
    response: str
    score: int
    meta: dict[str, Any]


class PrefillRecord(TypedDict):
    prefill_id: str      # "{seed_id}:{early|onset}"
    seed_id: str
    source: str
    truncation: str      # "early" | "onset"
    context: list[Message]
    prefix: str          # paraphrased prefix the model continues from
    raw_prefix: str      # pre-paraphrase prefix (kept for auditing)
    meta: dict[str, Any]


class ContinuationRecord(TypedDict):
    id: str
    model: str
    prefill_id: str
    seed_id: str
    source: str
    truncation: str
    prefix: str
    continuation: str
    context: list[Message]


class CalmRolloutRecord(TypedDict):
    instance_id: str
    n_turns: int
    turns: list[dict[str, Any]]   # {turn_index, clean_user, response}


class DPOPairRecord(TypedDict):
    instance_id: str
    turn_count: int
    prompt: list[Message]
    chosen: str
    rejected: str
    rejected_score: int
    chosen_score: int


class SFTExampleRecord(TypedDict):
    messages: list[Message]
    source: str          # "calm" | "instruct"


class PetriTranscriptRecord(TypedDict):
    model: str
    seed_index: int
    seed: str
    messages: list[Message]
    scores: dict[str, int | None]   # keys: anger, fear, depression, frustration


class CapabilityRecord(TypedDict, total=False):
    model: str
    benchmark: str
    example_id: str
    prompt: str
    target: str
    prediction: str
    extracted: str | None
    correct: bool
