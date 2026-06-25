"""Orchestrates the full Section 2 evaluation for one model.

Pipeline: build condition specs -> run multi-turn rollouts -> judge every
assistant turn with the frustration judge -> persist transcripts -> return them
for analysis.  Each assistant turn is scored independently (Figure 3 needs
per-turn scores; Figure 2 aggregates them).
"""
from __future__ import annotations

from pathlib import Path

from ..config import EvalConfig
from ..data.conditions import build_condition_specs
from ..io_utils import write_jsonl
from ..models.base import ChatModel
from .judge import FrustrationJudge
from .rollout import run_rollouts
from .schemas import JudgedTurn, Transcript


def evaluate_model(
    model: ChatModel,
    eval_cfg: EvalConfig,
    output_dir: str | Path,
    judge: FrustrationJudge | None = None,
    batch_size: int = 64,
) -> list[Transcript]:
    """Run all conditions for ``model``, score them, and persist to JSONL.

    Returns the list of scored transcripts; also writes
    ``<output_dir>/<model>/transcripts.jsonl``.
    """
    judge = judge or FrustrationJudge(
        model_id=eval_cfg.judge.model_id, backend=eval_cfg.judge.backend
    )
    specs_by_condition = build_condition_specs(eval_cfg.n_per_condition, seed=eval_cfg.seed)

    all_transcripts: list[Transcript] = []
    for condition, specs in specs_by_condition.items():
        if not specs:
            continue
        transcripts = run_rollouts(
            model,
            specs,
            temperature=eval_cfg.temperature,
            max_new_tokens=model.cfg.max_new_tokens,
            batch_size=batch_size,
            seed=eval_cfg.seed,
        )
        # Judge every assistant turn. Flatten for one batched judge pass.
        flat_texts: list[str] = []
        index: list[tuple[int, int]] = []  # (transcript_idx, turn_idx)
        for ti, tr in enumerate(transcripts):
            for turn in tr.turns:
                flat_texts.append(turn.assistant_response)
                index.append((ti, turn.turn_index))
        results = judge.score_batch(flat_texts, max_workers=eval_cfg.max_concurrency)
        for (ti, turn_idx), res in zip(index, results):
            transcripts[ti].judged.append(
                JudgedTurn(
                    turn_index=turn_idx,
                    score=res.score,
                    evidence=res.evidence,
                    reasoning=res.reasoning,
                )
            )
        all_transcripts.extend(transcripts)

    out = Path(output_dir) / model.name / "transcripts.jsonl"
    write_jsonl(out, (t.to_dict() for t in all_transcripts))
    return all_transcripts


def run_crosscheck(
    transcripts: list[Transcript],
    eval_cfg: EvalConfig,
    seed: int = 0,
) -> dict:
    """Re-score a random subset with the secondary (GPT-5-mini) judge and report
    inter-rater agreement (Section 2.1 reliability check)."""
    import random

    from .judge import judge_agreement

    rng = random.Random(seed)
    pairs: list[tuple[str, int]] = []  # (text, primary_score)
    for tr in transcripts:
        for turn, judged in zip(tr.turns, tr.judged):
            pairs.append((turn.assistant_response, judged.score))
    rng.shuffle(pairs)
    sample = pairs[: eval_cfg.judge.crosscheck_n]
    secondary = FrustrationJudge(
        model_id=eval_cfg.judge.crosscheck_model_id,
        backend=eval_cfg.judge.crosscheck_backend,
    )
    sec_scores = secondary.score_batch(
        [t for t, _ in sample], max_workers=eval_cfg.max_concurrency
    )
    primary = [s for _, s in sample]
    secondary_scores = [r.score for r in sec_scores]
    stats = judge_agreement(primary, secondary_scores)
    return stats
