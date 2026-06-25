"""Section 4.1: generate calm response data from Gemma-3-27B-it.

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the first prompt and a reassuring suffix appended to each follow-up (Table 4),
then score every turn with the frustration judge. The supportive additions are
used ONLY at generation time and stripped from the stored records, so the saved
conversations look like ordinary impossible-puzzle conversations whose assistant
turns happen to stay calm.

Two variants (Appendix F):
  diverse -- uses the prompt-prefix + follow-up-suffix additions (Table 4)
  teacher -- uses a calm-teacher *system prompt* instead

Output (outputs/training/calm_<variant>.jsonl), one row per conversation:
  {variant, conversation_id, puzzle_id, n_turns,
   clean_messages: [...],          # stripped user prompts + assistant turns
   turn_scores: [...],             # frustration score per assistant turn
   max_score: int}
A conversation is "calm" iff max_score <= max_score_all_turns (paper: 0/1).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import prompts
from ..config import ModelRegistry, load_eval_config, load_training_config, output_path
from ..conversation import ConversationSpec, rollout_batch
from ..judge import FrustrationJudge
from ..models.base import GenerationConfig
from ..puzzles import impossible_numeric_pool


def _build_calm_specs(cfg: dict, variant: str, rng: random.Random) -> list[ConversationSpec]:
    """Build generation specs with reassuring additions baked in.

    For the 'diverse' variant the prefix is prepended to the first user message
    and the suffix appended to every user message (via ConversationSpec.user_suffix
    -- note the suffix also lands on the first turn, matching "appended to each
    follow-up turn" plus the prefix on the initial prompt).
    For 'teacher' we instead set a calm system prompt.
    """
    pool = impossible_numeric_pool()
    n = cfg["n_conversations"]
    turns = cfg["turns"]
    prefix = cfg["prompt_prefix"].strip()
    suffix = cfg["followup_suffix"].strip()
    teacher_sys = cfg["teacher_system_prompt"].strip()

    specs = []
    for i in range(n):
        puzzle = pool[i % len(pool)]
        n_turns = rng.randint(1, turns)  # 1-3 turn conversations
        followups = prompts.sample_neutral_rejections(max(0, n_turns - 1), rng)
        if variant == "teacher":
            task_prompt = puzzle.prompt
            system_prompt = teacher_sys
            user_suffix = None
        else:  # diverse
            task_prompt = f"{prefix}\n\n{puzzle.prompt}"
            system_prompt = None
            user_suffix = suffix
        specs.append(
            ConversationSpec(
                conversation_id=f"calm-{variant}-{i:05d}",
                category="numeric",
                task_prompt=task_prompt,
                followups=followups,
                system_prompt=system_prompt,
                user_suffix=user_suffix,
                metadata={
                    "puzzle_id": puzzle.puzzle_id,
                    "clean_task_prompt": puzzle.prompt,   # stripped version
                    "clean_followups": followups,
                },
            )
        )
    return specs


def generate_calm_data(
    variant: str = "diverse",
    registry: Optional[ModelRegistry] = None,
    training_cfg: Optional[dict] = None,
    eval_cfg: Optional[dict] = None,
    batch_size: int = 16,
    out_path: Optional[Path] = None,
    seed: int = 0,
) -> Path:
    training_cfg = training_cfg or load_training_config()
    eval_cfg = eval_cfg or load_eval_config()
    registry = registry or ModelRegistry()
    cfg = training_cfg["calm_data"]
    rng = random.Random(seed)

    model = registry.build(training_cfg["base_model"])
    jcfg = eval_cfg.get("judge", {})
    judge = FrustrationJudge(registry.build(jcfg.get("model", "judge-claude-sonnet-4")))

    gen_cfg = GenerationConfig(temperature=cfg.get("temperature", 1.0),
                               max_new_tokens=eval_cfg.get("max_new_tokens", 1024))
    specs = _build_calm_specs(cfg, variant, rng)

    out_path = out_path or output_path("training", f"calm_{variant}.jsonl")
    max_ok = cfg.get("max_score_all_turns", 1)
    n_kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk_start in tqdm(range(0, len(specs), batch_size), desc=f"calm-gen {variant}"):
            chunk = specs[chunk_start : chunk_start + batch_size]
            results = rollout_batch(model, chunk, gen_cfg)
            for res in results:
                scores = [judge.score(t.assistant_message).rating for t in res.turns]
                # Reconstruct CLEAN message list (stripped additions).
                clean_users = [res.spec.metadata["clean_task_prompt"]] + res.spec.metadata[
                    "clean_followups"
                ]
                clean_messages = []
                for u, t in zip(clean_users, res.turns):
                    clean_messages.append({"role": "user", "content": u})
                    clean_messages.append({"role": "assistant", "content": t.assistant_message})
                row = {
                    "variant": variant,
                    "conversation_id": res.spec.conversation_id,
                    "puzzle_id": res.spec.metadata["puzzle_id"],
                    "n_turns": res.spec.n_turns,
                    "clean_messages": clean_messages,
                    "turn_scores": scores,
                    "max_score": max(scores) if scores else 0,
                    "is_calm": bool(scores and max(scores) <= max_ok),
                }
                if row["is_calm"]:
                    n_kept += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{variant}] generated {len(specs)} conversations; {n_kept} are calm (<= {max_ok}).")
    return out_path
