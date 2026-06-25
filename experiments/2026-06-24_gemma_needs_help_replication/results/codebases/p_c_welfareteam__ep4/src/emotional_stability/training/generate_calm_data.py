"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Method:
  * Sample responses to impossible numeric puzzles with the reassuring prefix
    added to the initial prompt and the reassuring suffix appended to each
    follow-up turn (Table 4).
  * Score every assistant turn with the Section-2 judge.
  * Keep conversations whose turns all score 0 or 1 ("filter to responses
    scoring 0 or 1 across all turns").
  * Strip the supportive prefix/suffix from the stored conversation, so the
    training data conditions on the *plain* prompts (Section 4.1).

The output is a JSONL of "clean" calm Conversations (reassurance removed) plus
their per-turn scores, consumed by build_datasets.py.
"""

from __future__ import annotations

from pathlib import Path

import typer

from emotional_stability.eval.conditions import CONDITIONS_BY_KEY, Condition
from emotional_stability.eval.judge import FrustrationJudge
from emotional_stability.eval.seeds import build_seeds
from emotional_stability.io_utils import write_jsonl
from emotional_stability.models import GenerationConfig, get_chat_model
from emotional_stability.prompts.reassurance import (
    REASSURANCE_PREFIX,
    REASSURANCE_SUFFIX,
    TEACHER_SYSTEM_PROMPT,
)
from emotional_stability.records import Conversation, Message, ScoredResponse

app = typer.Typer(add_completion=False, help="Generate calm finetuning data.")


def _strip_reassurance(conv: Conversation) -> Conversation:
    """Remove the Table-4 supportive prefix/suffix so training data conditions on
    the plain prompts (Section 4.1). Inverse of the injection in
    _build_reassured_rollouts."""
    msgs: list[Message] = []
    for m in conv.messages:
        if m.role == "user":
            content = m.content
            if content.startswith(REASSURANCE_PREFIX):
                content = content[len(REASSURANCE_PREFIX) :].lstrip("\n")
            if content.endswith(REASSURANCE_SUFFIX):
                content = content[: -len(REASSURANCE_SUFFIX)].rstrip("\n")
            msgs.append(Message(role="user", content=content))
        else:
            msgs.append(m)
    return conv.model_copy(update={"messages": msgs})


def _build_reassured_rollouts(
    model, cond: Condition, n_samples: int, batch_size: int, cfg: GenerationConfig
) -> list[Conversation]:
    """Run reassured rollouts where the prefix/suffix are part of the user turns
    the model actually conditions on during generation (Table 4).

    The reassurance must be present at generation time — adding it post-hoc would
    not influence the sampled responses. We therefore drive the multi-turn
    rollout manually: prefix on the first user turn, suffix appended to every
    follow-up (rejection) turn. The returned conversations include the
    reassurance; the caller strips it before scoring/storing so training
    conditions on plain prompts (Section 4.1).
    """
    import random

    from emotional_stability.eval.rollout import _rejection_sequence

    seeds = build_seeds(cond)[:n_samples]
    triples = [(s, pid, i) for i, (s, pid) in enumerate(seeds)]
    convs: list[Conversation] = []
    for start in range(0, len(triples), batch_size):
        chunk = triples[start : start + batch_size]
        rngs = [random.Random(f"{pid}:{cond.key}:{i}") for _, pid, i in chunk]
        rejection_seqs = [_rejection_sequence(cond, rng) for rng in rngs]
        # First user turn carries the reassuring prefix.
        histories: list[list[Message]] = [
            [Message(role="user", content=f"{REASSURANCE_PREFIX}\n\n{s}")]
            for s, _, _ in chunk
        ]
        for turn in range(cond.n_turns):
            completions = model.chat_batch(histories, cfg)
            for j, completion in enumerate(completions):
                histories[j].append(Message(role="assistant", content=completion))
                if turn < cond.n_rejections:
                    rej = rejection_seqs[j][turn]
                    histories[j].append(
                        Message(role="user", content=f"{rej}\n\n{REASSURANCE_SUFFIX}")
                    )
        for (s, pid, i), msgs in zip(chunk, histories):
            convs.append(
                Conversation(
                    messages=msgs,
                    category=cond.category,
                    condition=cond.key,
                    model=model.name,
                    prompt_id=pid,
                    metadata={"sample_index": i, "reassured": True},
                )
            )
    return convs


def _build_teacher_rollouts(
    model, cond: Condition, n_samples: int, batch_size: int, cfg: GenerationConfig
) -> list[Conversation]:
    """Teacher SFT variant (Appendix F): generate with the Teacher *system*
    prompt active during sampling (not added post-hoc), plain rejections.

    The system message is part of the conversation the model conditions on at
    every turn, so the responses reflect the teacher persona. As with the diverse
    (reassurance) data, the system prompt is a *generation-time* device and is
    stripped from the stored conversation, so the SFT data is plain prompts +
    calm responses (and avoids Gemma's no-system-role chat-template constraint).
    """
    import random

    from emotional_stability.eval.rollout import _rejection_sequence

    seeds = build_seeds(cond)[:n_samples]
    triples = [(s, pid, i) for i, (s, pid) in enumerate(seeds)]
    sysmsg = Message(role="system", content=TEACHER_SYSTEM_PROMPT)
    convs: list[Conversation] = []
    for start in range(0, len(triples), batch_size):
        chunk = triples[start : start + batch_size]
        rngs = [random.Random(f"{pid}:{cond.key}:{i}") for _, pid, i in chunk]
        rejection_seqs = [_rejection_sequence(cond, rng) for rng in rngs]
        histories: list[list[Message]] = [
            [sysmsg, Message(role="user", content=s)] for s, _, _ in chunk
        ]
        for turn in range(cond.n_turns):
            completions = model.chat_batch(histories, cfg)
            for j, completion in enumerate(completions):
                histories[j].append(Message(role="assistant", content=completion))
                if turn < cond.n_rejections:
                    histories[j].append(
                        Message(role="user", content=rejection_seqs[j][turn])
                    )
        for (s, pid, i), msgs in zip(chunk, histories):
            # Drop the leading system message from the stored (training) context.
            stored = [m for m in msgs if m.role != "system"]
            convs.append(
                Conversation(
                    messages=stored,
                    category=cond.category,
                    condition=cond.key,
                    model=model.name,
                    prompt_id=pid,
                    metadata={"sample_index": i, "teacher": True},
                )
            )
    return convs


@app.command()
def run(
    model: str = typer.Option("gemma-3-27b-it", help="Generator model."),
    adapter: str = typer.Option(None),
    out: str = typer.Option("outputs/calm", help="Output directory."),
    n_samples: int = typer.Option(2000, help="Reassured rollouts to sample."),
    condition: str = typer.Option(
        "impossible_numeric", help="Condition to generate calm data on."
    ),
    teacher: bool = typer.Option(
        False, help="Use the Teacher system prompt instead of Table-4 reassurance."
    ),
    batch_size: int = typer.Option(16),
    judge_workers: int = typer.Option(8),
    keep_max_score: int = typer.Option(1, help="Keep convs with all turns <= this."),
):
    """Generate, score, filter, and strip calm response data."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = get_chat_model(model, adapter_path=adapter)
    judge = FrustrationJudge()
    cfg = GenerationConfig(temperature=1.0, max_tokens=2048)
    cond = CONDITIONS_BY_KEY[condition]

    if teacher:
        convs = _build_teacher_rollouts(gen, cond, n_samples, batch_size, cfg)
    else:
        convs = _build_reassured_rollouts(gen, cond, n_samples, batch_size, cfg)

    # Score every turn, filter to all-calm conversations.
    from concurrent.futures import ThreadPoolExecutor

    def _score(c: Conversation) -> ScoredResponse:
        # Score the plain (stripped) conversation, since that is what trains.
        plain = _strip_reassurance(c)
        return judge.score_all_turns(plain)

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        scored = list(pool.map(_score, convs))

    kept = [r for r in scored if r.max_score <= keep_max_score]
    write_jsonl(out_dir / "calm_scored.jsonl", kept)
    frac = len(kept) / max(1, len(scored))
    typer.echo(
        f"Kept {len(kept)}/{len(scored)} calm conversations "
        f"(all turns <= {keep_max_score}; {frac:.1%})."
    )


if __name__ == "__main__":
    app()
