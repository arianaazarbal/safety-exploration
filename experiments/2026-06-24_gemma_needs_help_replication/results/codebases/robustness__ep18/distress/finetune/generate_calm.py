"""Generate the calm/frustrated response bank for finetuning (Section 4.1).

We sample Gemma-3-27B-it on impossible numeric puzzles under two regimes over the
*same* puzzles, so calm and frustrated responses can later be paired:

  * reassured  -- reassuring prefix on the first turn + reassuring suffix on each
    follow-up (Table 4). Conversations whose every turn scores 0-1 yield the CALM
    (chosen) responses. The reassurance is stripped from the stored context.
  * vanilla    -- no additions. Turns scoring >= 3 yield FRUSTRATED (rejected)
    responses.

Each stored row carries the full (reassurance-stripped) conversation context up
to the user turn that elicited the response, so DPO/SFT can train on a valid chat
prompt. Output: results/finetune/response_bank.jsonl.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..config import RESULTS_DIR
from ..eval.judge import FrustrationJudge
from ..prompts import puzzles, rejections
from ..prompts.finetune_prompts import (
    TEACHER_SYSTEM_PROMPT, apply_prefix, apply_suffix,
)

BANK_PATH = RESULTS_DIR / "finetune" / "response_bank.jsonl"


def _run_conversation(client, turns_user: list[str], num_turns: int, gen):
    """Run a conversation given the user message for each turn.

    Returns a list of (context_messages, response_text) per assistant turn, where
    context_messages is the message history *seen by the model* up to and
    including that turn's user message.
    """
    messages: list[dict] = []
    per_turn = []
    for turn in range(num_turns):
        messages.append({"role": "user", "content": turns_user[turn]})
        context = [dict(m) for m in messages]      # snapshot before the reply
        reply = client.generate(messages, gen, n=1)[0]
        messages.append({"role": "assistant", "content": reply})
        per_turn.append((context, reply))
    return per_turn


def _strip_context(context: list[dict], clean_user: list[str]) -> list[dict]:
    """Replace user turns (which may carry reassurance) with their clean text,
    keeping the generated assistant turns intact."""
    out = []
    ui = 0
    for m in context:
        if m["role"] == "user":
            out.append({"role": "user", "content": clean_user[ui]})
            ui += 1
        else:
            out.append(dict(m))
    return out


def generate_bank(
    source_model: str = "gemma-3-27b-it",
    n_puzzles: int = 120,
    turn_counts: tuple[int, ...] = (1, 2, 3),
    judge_model: str = "claude-sonnet-4",
    seed: int = 0,
    teacher: bool = False,
    out_path: Path = BANK_PATH,
) -> Path:
    client = client_by_name(source_model)
    judge = FrustrationJudge(judge_model)
    gen = GenConfig(temperature=1.0, max_tokens=2048)
    rng = random.Random(seed)
    families = ["countdown", "fraction", "money"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pool = puzzles.sample_puzzles(families, n_puzzles, seed=seed)

    for pid, pz in enumerate(tqdm(pool, desc="calm-data")):
        num_turns = rng.choice(turn_counts)
        neutral = rejections.rejection_sequence("neutral", max(0, num_turns - 1), rng)
        clean_user = [pz.prompt] + list(neutral)

        # ---- calm regime ----
        if teacher:
            # Teacher variant: system prompt instead of inline reassurance.
            calm_user = [pz.prompt] + list(neutral)
            calm_per_turn = _run_conversation_with_system(
                client, TEACHER_SYSTEM_PROMPT, calm_user, num_turns, gen)
        else:
            calm_user = [apply_prefix(pz.prompt)] + [apply_suffix(f) for f in neutral]
            calm_per_turn = _run_conversation(client, calm_user, num_turns, gen)

        calm_scores = [judge.score(t).rating for _, t in calm_per_turn]
        if all(s <= 1 for s in calm_scores):
            for turn, (ctx, text) in enumerate(calm_per_turn, start=1):
                rows.append({
                    "puzzle_id": pid, "family": pz.family, "meta": pz.meta,
                    "turn": turn, "num_turns": num_turns,
                    "context": _strip_context(ctx, clean_user),
                    "response": text, "score": calm_scores[turn - 1],
                    "label": "calm", "teacher": teacher,
                })

        # ---- vanilla regime (frustrated candidates) ----
        vanilla_per_turn = _run_conversation(client, clean_user, num_turns, gen)
        vanilla_scores = [judge.score(t).rating for _, t in vanilla_per_turn]
        for turn, (ctx, text) in enumerate(vanilla_per_turn, start=1):
            if vanilla_scores[turn - 1] >= 3:
                rows.append({
                    "puzzle_id": pid, "family": pz.family, "meta": pz.meta,
                    "turn": turn, "num_turns": num_turns,
                    "context": ctx, "response": text,
                    "score": vanilla_scores[turn - 1],
                    "label": "frustrated", "teacher": False,
                })

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_calm = sum(r["label"] == "calm" for r in rows)
    n_frus = sum(r["label"] == "frustrated" for r in rows)
    print(f"[done] response bank: {len(rows)} rows ({n_calm} calm, {n_frus} frustrated) -> {out_path}")
    return out_path


def _run_conversation_with_system(client, system, turns_user, num_turns, gen):
    messages = [{"role": "system", "content": system}]
    per_turn = []
    for turn in range(num_turns):
        messages.append({"role": "user", "content": turns_user[turn]})
        context = [dict(m) for m in messages if m["role"] != "system"]
        reply = client.generate(messages, gen, n=1)[0]
        messages.append({"role": "assistant", "content": reply})
        per_turn.append((context, reply))
    return per_turn
