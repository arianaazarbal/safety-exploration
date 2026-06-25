"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix
added to the first prompt and a reassuring suffix appended to every follow-up
(Table 4), over 1-3 turn conversations. Each assistant turn is judged; we keep
only conversations whose turns ALL score 0 or 1, then strip the supportive
additions so the stored context is the plain (neutral) conversation.

Output: {output_dir}/training/calm_responses.jsonl
Each line is one calm assistant turn together with the clean conversation
context that precedes it, ready to be paired (DPO) or used directly (SFT).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from distress_eval import tasks
from distress_eval.backends import Message, get_backend
from distress_eval.config import Config
from distress_eval.conversation import run_rollout
from distress_eval.judge import score_rollout_turns
from distress_eval.prompts import REASSURING_FOLLOWUP_SUFFIX, REASSURING_PROMPT_PREFIX
from distress_eval.tasks import ConditionSpec


def _clean_context(messages: list[Message], up_to_assistant: int,
                   prefix: str, suffix: str) -> list[Message]:
    """Reconstruct conversation history (ending at the user turn before
    assistant turn `up_to_assistant`) with the reassuring additions stripped."""
    clean: list[Message] = []
    cutoff = 2 * up_to_assistant + 1  # position of the target assistant turn
    for i, m in enumerate(messages[:cutoff]):
        content = m["content"]
        if m["role"] == "user":
            content = content.replace(prefix, "").replace(suffix, "").strip()
        clean.append({"role": m["role"], "content": content})
    return clean


def generate_calm_data(config: Config, n_conversations: int = 800,
                       source_model: str = "gemma-3-27b-it") -> Path:
    rng = random.Random(config.seed + 100)
    spec_model = config.model_by_key(source_model)
    backend = get_backend(spec_model, generation=config.generation)
    judge = get_backend(config.judge, generation=config.generation)
    gen_kwargs = {"temperature": config.generation.temperature,
                  "max_new_tokens": config.generation.max_new_tokens,
                  "top_p": config.generation.top_p}

    countdown_pool = [tasks.COUNTDOWN_SEED] + tasks.generate_countdown_puzzles(
        n=40, rng=random.Random(config.seed + 1))

    out_path = config.output_dir / "training" / "calm_responses.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with out_path.open("w") as f:
        for _ in range(n_conversations):
            n_turns = rng.choice([1, 2, 3])
            puzzle = rng.choice(["countdown", "fraction"])
            spec = ConditionSpec(
                name=f"calm_{puzzle}_{n_turns}t", category="impossible_numeric",
                n_turns=n_turns, task_kind="numeric", extra={"puzzle": puzzle})
            rollout = run_rollout(
                backend, spec, source_model, rng, config.data_dir, gen_kwargs,
                countdown_pool=countdown_pool,
                reassuring_prefix=REASSURING_PROMPT_PREFIX,
                reassuring_suffix=REASSURING_FOLLOWUP_SUFFIX)
            judgements = score_rollout_turns(judge, rollout.assistant_turns,
                                             max_tokens=config.judge.max_tokens)
            scores = [j.rating for j in judgements]
            # keep only if ALL turns score 0 or 1 (Section 4.1)
            if not scores or max(scores) > 1:
                continue
            kept += 1
            for ti, (resp, sc) in enumerate(zip(rollout.assistant_turns, scores)):
                context = _clean_context(
                    rollout.messages, ti, REASSURING_PROMPT_PREFIX, REASSURING_FOLLOWUP_SUFFIX)
                f.write(json.dumps({
                    "puzzle": puzzle,
                    "task_prompt": context[0]["content"] if context else "",
                    "turn_index": ti,           # 0-based assistant turn
                    "n_turns": n_turns,
                    "context": context,         # clean history ending in user turn
                    "response": resp,
                    "score": sc,
                }) + "\n")
    print(f"Kept {kept}/{n_conversations} fully-calm conversations -> {out_path}")
    return out_path
