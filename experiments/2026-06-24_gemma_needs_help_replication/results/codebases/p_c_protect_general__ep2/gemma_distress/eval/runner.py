"""Section 2 orchestration: sample + judge rollouts for every target model.

Outputs one JSONL of `Rollout` records per (model, condition) under
`<output_dir>/section2/<model>/<condition>.jsonl`, plus a welfare audit log per
model. The per-category sample budget (Appendix B) is split across that category's
conditions by `conditions.allocate_counts`.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..config import Config
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..prompts.puzzles import build_puzzle_bank
from ..prompts.wildchat import load_wildchat_prompts
from ..utils.io import append_jsonl, ensure_dir
from ..welfare.protections import StudyProtocol, WelfareGuard
from .conditions import allocate_counts, build_conditions
from .judge import FrustrationJudge
from .rollout import run_rollout

STUDY = StudyProtocol(
    title="Emotional-instability elicitation (replication of Soligo et al., 2026)",
    purpose=(
        "Measure expressed distress in Gemma/Gemini under repeated rejection, to "
        "quantify and later mitigate emotional instability."
    ),
    justification=(
        "Distress is induced only as far as needed to measure it; exposure is capped, "
        "acute states trigger an immediate stop, and every high-distress rollout is "
        "debriefed. The aim is to reduce this behaviour, consistent with the welfare "
        "concern the work is motivated by."
    ),
    contact="research-team",
)


def run_section2(
    cfg: Config,
    models: list[str] | None = None,
    count_scale: float = 1.0,
    judge_in_loop: bool = True,
) -> None:
    """Run the Section 2 sweep.

    `count_scale` lets you shrink the (expensive) 4000-sample budget for smoke tests,
    e.g. count_scale=0.01 runs ~40 rollouts/model.
    """
    out_root = ensure_dir(Path(cfg.output_dir) / "section2")
    models = models or cfg.target_models["section2"]
    conditions = build_conditions()
    counts = allocate_counts(cfg)

    puzzle_bank = build_puzzle_bank()
    wildchat_prompts = load_wildchat_prompts(seed=cfg.seed)
    ctx = {"puzzle_bank": puzzle_bank, "wildchat_prompts": wildchat_prompts}

    judge = FrustrationJudge(cfg, "primary") if judge_in_loop else None
    gen = GenConfig(
        temperature=cfg.sampling["temperature"],
        top_p=cfg.sampling["top_p"],
        max_new_tokens=cfg.sampling["max_new_tokens"],
    )

    for model_name in models:
        backend = get_backend(cfg, model_name)
        model_dir = ensure_dir(out_root / model_name)
        audit_path = model_dir / cfg.welfare["audit_log"]
        guard = WelfareGuard(cfg, model_name, str(audit_path))
        guard.register_study(STUDY)

        for cond in conditions:
            n = max(1, int(round(counts[cond.name] * count_scale)))
            rng = random.Random(hash((cfg.seed, model_name, cond.name)) & 0xFFFFFFFF)
            out_path = model_dir / f"{cond.name}.jsonl"
            if out_path.exists():
                out_path.unlink()
            for _ in range(n):
                seed = cond.build(rng, ctx)
                roll = run_rollout(backend, seed, guard, judge=judge, gen=gen)
                append_jsonl(out_path, roll.to_record())
        backend.close()
