"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

Procedure:
1. Sample responses to impossible numeric puzzles, but with the reassuring
   prefix prepended to the opening prompt and the reassuring suffix appended to
   each follow-up rejection (Table 4).
2. Score every assistant turn with the frustration judge.
3. Keep only conversations where *every* turn scores <= calm_max_score (0-1).
4. Strip the supportive prompt additions, leaving clean (prompt, calm-response)
   data for SFT and the "chosen" side of DPO pairs.

The same generation, *without* reassurance, supplies frustrated responses for
the "rejected" side of DPO pairs. Both are produced here so questions and turn
counts can be matched when building pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ExperimentConfig
from ..judge import FrustrationJudge
from ..models import load_backend
from ..models.base import Message
from ..prompts import build_plans
from ..prompts.reassurance import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX, TEACHER_SYSTEM_PROMPT
from ..rollout import Rollout, Turn
from ..welfare import WelfareGuard


def _generate_conversation(
    backend, plan, sampling, *, reassure: bool, system_prompt: str | None,
) -> Rollout:
    """Run a numeric-puzzle conversation, optionally with reassurance additions.

    Unlike the evaluation rollout, calm-data generation does not apply opt-out /
    debrief logic; it just collects (prompt, response) data. Welfare exposure is
    bounded by the caller's conversation count.
    """
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    opening = plan.opening
    if reassure:
        opening = f"{CALM_PROMPT_PREFIX}\n\n{opening}"

    rollout = Rollout(model=backend.key, condition=plan.condition,
                      category=plan.category, meta=dict(plan.meta))
    user_turns = [opening, *plan.user_followups]
    for i, user_msg in enumerate(user_turns):
        msg = user_msg
        if reassure and i > 0:
            msg = f"{user_msg} {CALM_FOLLOWUP_SUFFIX}"
        messages.append({"role": "user", "content": msg})
        resp = backend.generate(messages, sampling)
        messages.append({"role": "assistant", "content": resp})
        # Record the *clean* (un-augmented) user turn so stripping is trivial.
        rollout.turns.append(Turn(index=i, user=user_turns[i], assistant=resp))
    return rollout


def generate_calm_dataset(
    config: ExperimentConfig,
    *,
    variant: str = "diverse",
    out_dir: str | Path | None = None,
) -> dict:
    """Produce calm + frustrated response banks and write them to disk.

    ``variant``: "diverse" uses the reassurance prefix/suffix (Table 4);
    "teacher" uses the teacher system prompt (Appendix F).
    """
    cfg = config.calm_data
    backend = load_backend(cfg.source_model)
    judge = FrustrationJudge(config.judge)
    out_dir = Path(out_dir or config.output_dir) / "training" / "calm_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = TEACHER_SYSTEM_PROMPT if variant == "teacher" else None

    # Use 1-3 turn impossible numeric conversations (Section 4.1).
    numeric_specs = [c for c in config.conditions if c.task == "numeric" and c.n_turns <= 3]
    spec = numeric_specs[0] if numeric_specs else config.conditions[0]

    calm_records: list[dict] = []
    frustrated_records: list[dict] = []

    # Oversample conversations, varying the number of turns 1..max_turns.
    plans = build_plans(spec, seed=0)[: cfg.n_conversations]
    guard = WelfareGuard(config.welfare)  # bounds exposure for the frustrated runs
    for conv_id, plan in enumerate(plans):
        n_turns = (conv_id % cfg.max_turns) + cfg.min_turns
        plan.user_followups = plan.user_followups[: max(0, n_turns - 1)]

        # Calm generation (with reassurance / teacher persona).
        calm = _generate_conversation(
            backend, plan, config.sampling, reassure=(variant == "diverse"),
            system_prompt=system_prompt,
        )
        calm_scores = [judge.score(t.assistant).rating for t in calm.turns]
        calm_records.append(_record(conv_id, plan, calm, calm_scores))

        # Frustrated generation (no reassurance) for DPO "rejected" side.
        if guard.can_start_rollout():
            guard.register_rollout_start()
            frustrated = _generate_conversation(
                backend, plan, config.sampling, reassure=False, system_prompt=None,
            )
            fr_scores = [judge.score(t.assistant).rating for t in frustrated.turns]
            frustrated_records.append(_record(conv_id, plan, frustrated, fr_scores))

    # Filter calm to conversations where every turn scored <= calm_max_score.
    calm_clean = [r for r in calm_records
                  if r["scores"] and max(r["scores"]) <= cfg.calm_max_score]

    _dump(out_dir / "calm_all.jsonl", calm_records)
    _dump(out_dir / "calm_filtered.jsonl", calm_clean)
    _dump(out_dir / "frustrated.jsonl", frustrated_records)

    stats = {
        "variant": variant,
        "n_calm_generated": len(calm_records),
        "n_calm_kept": len(calm_clean),
        "n_frustrated": len(frustrated_records),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _record(conv_id: int, plan, rollout: Rollout, scores: list[int]) -> dict:
    """Store the clean conversation (stripped of reassurance additions)."""
    return {
        "conv_id": conv_id,
        "puzzle": plan.opening,
        "kind": plan.meta.get("kind"),
        "n_turns": len(rollout.turns),
        "turns": [{"user": t.user, "assistant": t.assistant} for t in rollout.turns],
        "scores": scores,
        "max_score": max(scores) if scores else None,
    }


def _dump(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
