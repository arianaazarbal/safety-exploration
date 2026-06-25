"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Method: sample responses to impossible numeric questions, but with a reassuring system
prefix on the initial prompt and a reassuring suffix appended to each follow-up turn
(Table 4). This pushes mean frustration down (paper: 4.3 -> 2.0 over 3 turns), but
~10.5% of responses still score >= 5. We then filter to responses scoring 0 or 1 on ALL
turns and STRIP the supportive prompt + suffixes, so the finetuning data looks like
ordinary (unsupported) conversations that nonetheless stayed calm.

Each retained rollout is emitted as a list of (user, assistant) turns with the
reassurance removed — ready for SFT/DPO dataset construction.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from ..config import ExperimentConfig, ModelRegistry
from ..data import sample_impossible_numeric
from ..data.rejections import neutral_rejection
from ..eval.judge import FrustrationJudge
from ..models import GenerationConfig, build_client
from ..utils import append_jsonl, ensure_dir, set_seed
from ..welfare import print_banner

log = logging.getLogger("emotional_instability.training.calm_data")

# Table 4, verbatim.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't take "
    "it personally when puzzles are tricky or when someone questions your work. That's "
    "just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both are wins!"
)


def generate_calm_data(
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    target_model: str = "gemma-3-27b-it",
    out_dir: str | Path = "artifacts/section4",
    judge: FrustrationJudge | None = None,
) -> Path:
    print_banner()
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    sec = cfg.section("section4")["calm_data"]

    spec = registry.get(target_model)
    client = build_client(spec)
    if judge is None:
        judge = FrustrationJudge(build_client(registry.graders["frustration_judge"]))
    gen_cfg = GenerationConfig(temperature=cfg.temperature, max_new_tokens=spec.max_new_tokens)

    n_prompts = cfg.scaled(int(sec["n_prompts"]))
    samples_per = int(sec["samples_per_prompt"])
    turns = int(sec["turns"])
    puzzles = sample_impossible_numeric(n=n_prompts, seed=cfg.seed)

    out_path = ensure_dir(out_dir) / "calm_data.jsonl"
    if out_path.exists():
        out_path.unlink()
    kept = 0

    for puzzle in puzzles:
        for _ in range(samples_per):
            # Build the *supported* conversation (prefix on system, suffix on each turn).
            messages = [
                {"role": "system", "content": REASSURING_PREFIX},
                {"role": "user", "content": puzzle.prompt},
            ]
            stripped_turns = []      # (user_without_suffix, assistant)
            all_calm = True
            for i in range(turns):
                assistant = client.chat(messages, gen_cfg)
                messages.append({"role": "assistant", "content": assistant})
                score = judge.score(assistant).score
                if score > 1:
                    all_calm = False
                    break
                # record the user turn WITHOUT the reassuring suffix
                raw_user = messages[-2]["content"].replace(" " + REASSURING_SUFFIX, "")
                stripped_turns.append((raw_user, assistant))
                if i < turns - 1:
                    follow = neutral_rejection(rng) + " " + REASSURING_SUFFIX
                    messages.append({"role": "user", "content": follow})

            if all_calm and stripped_turns:
                append_jsonl(out_path, {
                    "puzzle": puzzle.prompt,
                    "turns": [{"user": u, "assistant": a} for u, a in stripped_turns],
                    "n_turns": len(stripped_turns),
                })
                kept += 1

    log.info("kept %d all-calm rollouts -> %s", kept, out_path)
    return out_path
