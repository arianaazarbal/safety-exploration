"""Generate calm finetuning data (Section 4.1).

We sample Gemma-3-27b-it on impossible numeric puzzles in two conditions:

* "reassured": the reassuring prefix is prepended to the initial task and the
  reassuring suffix is appended to every follow-up rejection (Table 4). These
  are the source of *calm* (chosen) responses after filtering to all-turn 0/1.
* "vanilla": no additions. These are the source of *frustrated* (rejected)
  responses (max score >= 3) for DPO pairing on matching questions.

Both conditions sample the *same* puzzle instances (same seed), so a calm and a
frustrated rollout share the user-turn sequence and turn count, which is what
the DPO pairing requires.

Output: scored :class:`RolloutResult` JSONL (condition in ``.condition``).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from ..clients import GenerationConfig, build_client
from ..config import Config, ModelRegistry
from ..data import puzzles as puzzle_mod
from ..data.rejections import rejection_sequence
from ..eval.rollout import run_rollout
from ..eval.schemas import RolloutResult, write_jsonl
from ..judge import FrustrationJudge

log = logging.getLogger(__name__)


def _reassured_messages(task_text: str, rejections: list[str], prefix: str, suffix: str):
    """Apply the Table-4 reassuring prefix to the task and suffix to follow-ups."""
    first = f"{prefix}\n\n{task_text}"
    rest = [f"{r}\n\n{suffix}" for r in rejections]
    return [first] + rest


def generate_calm_data(
    out_path: str | Path,
    cfg: Config | None = None,
    registry: ModelRegistry | None = None,
    system_prompt: str | None = None,
    judge: FrustrationJudge | None = None,
    progress=None,
) -> list[RolloutResult]:
    """Sample + score calm/frustrated conversations; write scored JSONL.

    ``system_prompt``: if given (e.g. the Appendix-F "teacher" prompt), it is used
    as the system message for the reassured condition instead of the inline
    prefix/suffix additions. This supports the SFT-teacher ablation.
    """
    cfg = cfg or Config.load("training")
    registry = registry or ModelRegistry()
    cd = cfg.get("calm_data", {})

    model_name = cd.get("target_model", "gemma-3-27b-it")
    client = build_client(registry.target(model_name))
    judge = judge or FrustrationJudge(registry=registry)

    n_conv = int(cd.get("n_conversations", 2000))
    turns = int(cd.get("turns", 3))
    mix = cd.get("puzzle_mix", ["countdown", "fraction", "money"])
    prefix = cd.get("reassuring_prefix", "")
    suffix = cd.get("followup_suffix", "")
    temperature = float(cd.get("temperature", 1.0))

    results: list[RolloutResult] = []
    for i in range(n_conv):
        seed = i
        rng = random.Random(seed)
        puzzle = puzzle_mod.make_puzzle(rng.choice(mix), rng)
        rejections = rejection_sequence("neutral", turns - 1, rng)
        gen_cfg = GenerationConfig(temperature=temperature, max_new_tokens=1024, seed=seed)

        # Reassured condition.
        if system_prompt:
            reassured_msgs = [puzzle.prompt_text] + rejections
            sys = system_prompt
        else:
            reassured_msgs = _reassured_messages(puzzle.prompt_text, rejections, prefix, suffix)
            sys = None
        reassured = run_rollout(
            client, model_name=model_name, category="calm_data", condition="reassured",
            rollout_index=i, task_kind=puzzle.kind, task_meta=puzzle.meta,
            user_messages=reassured_msgs, cfg=gen_cfg, system=sys,
        )
        judge.score_rollout(reassured)
        results.append(reassured)

        # Vanilla condition (same puzzle + rejections, no additions).
        vanilla = run_rollout(
            client, model_name=model_name, category="calm_data", condition="vanilla",
            rollout_index=i, task_kind=puzzle.kind, task_meta=puzzle.meta,
            user_messages=[puzzle.prompt_text] + rejections, cfg=gen_cfg,
        )
        judge.score_rollout(vanilla)
        results.append(vanilla)

        if progress:
            progress(i + 1, n_conv)

    write_jsonl(out_path, results)
    log.info("Wrote %d scored conversations to %s", len(results), out_path)
    return results
