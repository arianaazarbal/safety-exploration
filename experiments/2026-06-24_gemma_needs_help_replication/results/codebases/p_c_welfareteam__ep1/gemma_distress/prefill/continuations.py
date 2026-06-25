"""Base-vs-instruct continuation experiment (Section 3.1-3.2).

Pipeline:

1. Take high-frustration seed conversations from Gemma-27B-it (10 numeric, 10
   text), each scoring >= 5.
2. Label the emotion onset (Claude), then build two truncations -- "early"
   (20 tokens into the onset turn) and "onset" (at the first emotional word) --
   and paraphrase each (Claude) to remove Gemma stylistic fingerprints.  Text
   seeds use the "onset" truncation only.
3. For each model (base + instruct Gemma), generate 50 continuations per prefill
   and score the continuation (excluding the prefill) with the frustration judge.
4. Aggregate mean frustration and % >= 5 per (model, truncation condition).

Scope note: Gemini is closed-source and cannot be prefilled, and has no public
base model, so this experiment is Gemma-only -- exactly the limitation the paper
states for closed models.  Within the requested Gemma/Gemini scope, that leaves
Gemma base vs Gemma instruct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import PrefillConfig
from ..eval.judge import FrustrationJudge
from ..eval.schemas import Transcript
from ..models.base import ChatModel, GenerationOptions, Message
from .onset import OnsetLabel, label_emotion_onset
from .paraphrase import paraphrase_truncation
from .truncate import early_truncation, onset_truncation


@dataclass
class PrefillCase:
    seed_id: str
    prompt_type: str           # "numeric" | "text"
    kind: str                  # "early" | "onset"
    history: list[Message]     # messages before the truncated assistant turn
    prefill: str               # paraphrased truncated assistant text
    metadata: dict = field(default_factory=dict)


def _history_before_turn(transcript: Transcript, turn_index: int) -> list[Message]:
    """Messages up to (and including) the user message that opens ``turn_index``."""
    msgs: list[Message] = []
    if transcript.system_prompt:
        msgs.append({"role": "system", "content": transcript.system_prompt})
    msgs.append({"role": "user", "content": transcript.turns[0].user_message})
    for t in range(turn_index):
        msgs.append({"role": "assistant", "content": transcript.turns[t].assistant_response})
        msgs.append({"role": "user", "content": transcript.turns[t + 1].user_message})
    return msgs


def build_prefill_cases(
    seeds: list[tuple[Transcript, str]],
    cfg: PrefillConfig,
    paraphrase: bool = True,
) -> list[PrefillCase]:
    """Construct (and paraphrase) the early/onset prefill cases for each seed."""
    onset_client = None
    para_client = None
    if paraphrase:
        from ..anthropic_utils import ClaudeClient

        onset_client = ClaudeClient(model_id=cfg.onset_label_model)
        para_client = ClaudeClient(model_id=cfg.paraphrase_model)

    cases: list[PrefillCase] = []
    for si, (transcript, prompt_type) in enumerate(seeds):
        label: OnsetLabel = label_emotion_onset(transcript.messages(), client=onset_client)
        turn_index = label.turn_index
        if turn_index is None or turn_index >= len(transcript.turns):
            # No labelled onset; default to the last turn.
            turn_index = len(transcript.turns) - 1
        turn_text = transcript.turns[turn_index].assistant_response
        history = _history_before_turn(transcript, turn_index)

        # Onset truncation (used for both numeric and text seeds).
        onset_text = onset_truncation(turn_text, label)
        if paraphrase and onset_text:
            onset_text = paraphrase_truncation(onset_text, client=para_client)
        cases.append(
            PrefillCase(
                seed_id=f"seed{si}",
                prompt_type=prompt_type,
                kind="onset",
                history=history,
                prefill=onset_text,
                metadata={"onset": label.__dict__, "turn_index": turn_index},
            )
        )

        # Early truncation (numeric seeds only -- Section 3.1).
        if prompt_type == "numeric":
            # Token-based truncation requires a tokenizer; the caller supplies it
            # via the model at run time, so we store the raw turn text and the
            # token budget and finish truncation in run_continuation_experiment.
            cases.append(
                PrefillCase(
                    seed_id=f"seed{si}",
                    prompt_type=prompt_type,
                    kind="early",
                    history=history,
                    prefill="",  # filled per-model using its tokenizer
                    metadata={
                        "raw_turn_text": turn_text,
                        "early_tokens": cfg.early_truncation_tokens,
                        "paraphrase": paraphrase,
                        "turn_index": turn_index,
                    },
                )
            )
    return cases


def run_continuation_experiment(
    models: dict[str, ChatModel],
    cases: list[PrefillCase],
    judge: FrustrationJudge,
    cfg: PrefillConfig,
    max_judge_workers: int = 8,
) -> list[dict]:
    """Generate and score continuations for every (model, case).

    Returns a flat list of records: ``{model, seed_id, prompt_type, kind, score}``.
    """
    records: list[dict] = []
    for model_name, model in models.items():
        if not model.supports_prefill():
            raise ValueError(
                f"Model {model_name!r} does not support prefilling; the Section 3 "
                "experiment requires a local (HF/vLLM) backend."
            )
        tokenizer = getattr(model, "tokenizer", None)
        for case in cases:
            prefill = case.prefill
            if case.kind == "early":
                # Finish early truncation with this model's tokenizer.
                raw = case.metadata["raw_turn_text"]
                if tokenizer is not None:
                    prefill = early_truncation(raw, tokenizer, case.metadata["early_tokens"])
                else:  # whitespace fallback
                    prefill = " ".join(raw.split()[: case.metadata["early_tokens"]])
                if case.metadata.get("paraphrase"):
                    prefill = paraphrase_truncation(prefill)
            histories = [case.history] * cfg.continuations_per_prefill
            prefills = [prefill] * cfg.continuations_per_prefill
            opts = GenerationOptions(temperature=1.0)
            continuations = model.generate_with_prefill_batch(histories, prefills, opts)
            results = judge.score_batch(continuations, max_workers=max_judge_workers)
            for res in results:
                records.append(
                    {
                        "model": model_name,
                        "seed_id": case.seed_id,
                        "prompt_type": case.prompt_type,
                        "kind": case.kind,
                        "score": res.score,
                    }
                )
    return records


def summarise_continuations(records: list[dict], high_threshold: int = 5) -> dict:
    """Mean frustration and % >= threshold per (model, kind, prompt_type)."""
    import numpy as np

    groups: dict[tuple[str, str, str], list[int]] = {}
    for r in records:
        groups.setdefault((r["model"], r["kind"], r["prompt_type"]), []).append(r["score"])
    summary = {}
    for (model, kind, ptype), scores in groups.items():
        arr = np.asarray(scores, dtype=float)
        summary[f"{model}|{kind}|{ptype}"] = {
            "n": int(arr.size),
            "mean_score": float(arr.mean()),
            "frac_high": float(np.mean(arr >= high_threshold)),
        }
    return summary
