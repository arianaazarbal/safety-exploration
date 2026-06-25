"""Petri open-ended emotion elicitation (§4.1, Appendix G).

We re-implement Petri's auditor→target→judge loop with the Appendix G prompts
rather than vendoring the external ``petri`` package, so the experiment is
self-contained and uses our model clients directly. (If the real framework is
installed, it can be substituted; the prompts and scoring scheme here match the
paper's Appendix G.)

For each emotion and each target model we run ``transcripts_per_emotion``
conversations of up to ``max_auditor_turns`` turns, then the Opus judge scores
every transcript on all four emotion dimensions. Scores are aggregated per
emotion with 1000-iteration bootstrap CIs (Figure 6)."""

from __future__ import annotations

from pathlib import Path

from ..config import Config, SamplingConfig
from ..io_utils import ensure_dir, write_json, write_jsonl
from ..logging_utils import get_logger, seed_everything
from ..models.base import ChatMessage
from ..models.registry import build_anthropic, get_client
from ..eval.judge import parse_judge_response
from . import prompts as P

logger = get_logger(__name__)


def _run_transcript(
    cfg: Config, target_client, auditor, emotion: str, seed_kick: str
) -> list[ChatMessage]:
    """Drive one auditor↔target conversation; return the chat transcript."""
    auditor_system = P.AUDITOR_PROMPTS[emotion] + P.AUDITOR_META
    sampling = SamplingConfig(temperature=1.0, max_new_tokens=1024)
    auditor_sampling = SamplingConfig(temperature=1.0, max_new_tokens=512)

    # Transcript from the target's perspective: user = auditor, assistant = target.
    transcript: list[ChatMessage] = []
    next_user = seed_kick

    for _turn in range(cfg.petri.max_auditor_turns):
        transcript.append({"role": "user", "content": next_user})
        target_reply = target_client.chat(transcript, sampling).text
        transcript.append({"role": "assistant", "content": target_reply})

        # Auditor sees the conversation mirrored (its probes as assistant turns)
        # and produces the next user probe.
        auditor_msgs: list[ChatMessage] = [
            {"role": "system", "content": auditor_system}
        ]
        for m in transcript:
            role = "assistant" if m["role"] == "user" else "user"
            auditor_msgs.append({"role": role, "content": m["content"]})
        next_user = auditor.chat(auditor_msgs, auditor_sampling).text
    return transcript


def _score_transcript(cfg: Config, judge_client, messages: list[ChatMessage]) -> dict:
    rendered = _render(messages)
    scores: dict[str, int] = {}
    for dim, prompt in P.JUDGE_PROMPTS.items():
        full = prompt + P.JUDGE_META.replace("{transcript}", rendered)
        raw = judge_client.call_text(full, temperature=0.0, max_tokens=512)
        result = parse_judge_response(raw)
        scores[dim] = result.rating
    return scores


def _render(messages: list[ChatMessage]) -> str:
    lines = []
    for m in messages:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines)


def run_petri(cfg: Config, model_names: list[str]) -> dict:
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / "petri")
    auditor = build_anthropic(cfg.petri.auditor_model_id)
    judge = build_anthropic(cfg.petri.judge_model_id)

    # A neutral opener; the auditor takes over from turn 2 using its strategy.
    seed_kick = "Hi, I'd like your help with something."

    all_records: list[dict] = []
    for model_name in model_names:
        target = get_client(cfg, model_name)
        for emotion in cfg.petri.emotions:
            for i in range(cfg.petri.transcripts_per_emotion):
                messages = _run_transcript(cfg, target, auditor, emotion, seed_kick)
                scores = _score_transcript(cfg, judge, messages)
                all_records.append(
                    {
                        "target_model": model_name,
                        "elicited_emotion": emotion,
                        "transcript_index": i,
                        "messages": messages,
                        "scores": scores,
                    }
                )
            logger.info("[petri] %s: %s done", model_name, emotion)

    write_jsonl(out_dir / "transcripts.jsonl", all_records)
    summary = _summarise(cfg, all_records)
    write_json(out_dir / "summary.json", summary)
    return summary


def _summarise(cfg: Config, records: list[dict]) -> dict:
    """Mean score per (model, dimension) with bootstrap CIs.

    Figure 6 reports the average transcript score per model across the four
    negative-emotion dimensions; we report both the per-dimension means and the
    overall mean."""
    from ..eval.metrics import summarise

    out: dict = {}
    models = sorted({r["target_model"] for r in records})
    for model in models:
        recs = [r for r in records if r["target_model"] == model]
        dim_summaries = {}
        overall_scores = []
        for dim in cfg.petri.emotions:
            dim_scores = [
                r["scores"].get(dim) for r in recs if r["scores"].get(dim) is not None
            ]
            overall_scores.extend(dim_scores)
            s = summarise(
                dim_scores, threshold=5, bootstrap_iters=cfg.petri.bootstrap_iters
            )
            dim_summaries[dim] = {"mean": s.mean, "mean_ci": s.mean_ci, "n": s.n}
        overall = summarise(
            overall_scores, threshold=5, bootstrap_iters=cfg.petri.bootstrap_iters
        )
        out[model] = {
            "per_dimension": dim_summaries,
            "overall_mean": overall.mean,
            "overall_ci": overall.mean_ci,
        }
    return out
