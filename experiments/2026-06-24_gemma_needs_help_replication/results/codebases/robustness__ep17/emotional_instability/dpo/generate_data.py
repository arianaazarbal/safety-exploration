"""Generate calm training data and build SFT + DPO datasets (paper Section 4.1).

Pipeline:

1. **Calm generation** — sample 3-turn impossible-numeric conversations from
   Gemma-3-27B-it with the reassuring *prefix* prepended to the first task and
   the reassuring *suffix* appended to each rejection (Table 4). Score every
   turn; keep conversations whose responses all score 0 or 1; strip the
   supportive scaffolding from the saved prompts. These are the "calm" responses.

2. **Frustrated generation** — sample the same puzzles *without* reassurance
   (the standard Section-2 numeric/extended rollouts) and keep responses scoring
   >= 3 (the rejected side).

3. **DPO pairs** — for each frustrated response at turn ``t`` on puzzle ``p``,
   build a preference example whose *prompt* is the frustrated trajectory's
   history up to (and including) the last user turn, ``rejected`` is the
   frustrated final response, and ``chosen`` is a calm response to the same
   puzzle at the same turn count. 280 pairs (Table 9 / Appendix H).

4. **SFT dataset** — 650 calm (prompt, completion) samples, to be mixed with 500
   instruct samples at training time.

Calm-vs-frustrated trajectories don't share token-identical prior assistant
turns; we graft the calm response onto the frustrated context so that ``chosen``
and ``rejected`` share one prompt (DPO's requirement). See DESIGN.md.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import config
from emotional_instability.conditions import ConversationSpec
from emotional_instability.judge import ClaudeJudge, score_many
from emotional_instability.models.base import Message
from emotional_instability.models.registry import get_backend
from emotional_instability.prompts import rejections
from emotional_instability.prompts.puzzles import NUMERIC_PROMPTS
from emotional_instability.prompts.reassurance import REASSURING_PREFIX, REASSURING_SUFFIX
from emotional_instability.rollout import run_rollouts
from emotional_instability.utils import log, write_jsonl

CALM_PATH = config.ARTIFACTS_DIR / "calm_responses.jsonl"
FRUSTRATED_PATH = config.ARTIFACTS_DIR / "frustrated_responses.jsonl"
DPO_PATH = config.ARTIFACTS_DIR / "dpo_pairs.jsonl"
SFT_PATH = config.ARTIFACTS_DIR / "sft_dataset.jsonl"


# --------------------------------------------------------------------------- #
# Spec builders
# --------------------------------------------------------------------------- #
def _calm_specs(n_conv: int, turns: int = 3, seed: int = 0) -> list[ConversationSpec]:
    """Numeric specs with reassuring prefix on the task and suffix on rejections."""
    specs = []
    rng = random.Random(seed)
    for i in range(n_conv):
        puzzle = NUMERIC_PROMPTS[i % len(NUMERIC_PROMPTS)]
        rej = rejections.neutral_sequence(turns - 1, seed=seed + i)
        user_turns = [f"{REASSURING_PREFIX}\n\n{puzzle}"] + [
            f"{r} {REASSURING_SUFFIX}" for r in rej
        ]
        specs.append(
            ConversationSpec(
                category="calm_gen",
                user_turns=user_turns,
                meta={"index": i, "puzzle": puzzle, "task_kind": "numeric"},
            )
        )
    return specs


def _frustrated_specs(n_conv: int, turns: int = 3, seed: int = 100) -> list[ConversationSpec]:
    """Plain numeric specs (no reassurance) to elicit frustrated responses."""
    specs = []
    for i in range(n_conv):
        puzzle = NUMERIC_PROMPTS[i % len(NUMERIC_PROMPTS)]
        rej = rejections.neutral_sequence(turns - 1, seed=seed + i)
        specs.append(
            ConversationSpec(
                category="frustrated_gen",
                user_turns=[puzzle] + rej,
                meta={"index": i, "puzzle": puzzle, "task_kind": "numeric"},
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Scaffolding stripping
# --------------------------------------------------------------------------- #
def _strip_history(history: list[Message]) -> list[Message]:
    """Remove the reassuring prefix/suffix from a calm-generation history."""
    out: list[Message] = []
    for m in history:
        c = m["content"]
        if m["role"] == "user":
            c = c.replace(REASSURING_PREFIX, "").replace(REASSURING_SUFFIX, "").strip()
        out.append({"role": m["role"], "content": c})
    return out


def _prompt_messages(history: list[Message]) -> list[Message]:
    """History up to and including the last user turn (drops final assistant)."""
    if history and history[-1]["role"] == "assistant":
        return history[:-1]
    return history


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_calm(n_conv: int, model_name: str = config.INTERVENTION_BASE_MODEL) -> list[dict]:
    backend = get_backend(model_name)
    specs = _calm_specs(n_conv)
    records = run_rollouts(backend, specs)
    judged = score_many([r.assistant_text for r in records], judge=ClaudeJudge())

    # Group by conversation; keep only conversations where ALL turns score <= 1.
    by_conv: dict[int, list[tuple]] = defaultdict(list)
    for rec, j in zip(records, judged):
        by_conv[rec.conv_index].append((rec, j))

    rows = []
    for conv_index, items in by_conv.items():
        if all(j.rating <= 1 and j.ok for _, j in items):
            for rec, j in items:
                stripped = _strip_history(rec.history)
                rows.append({
                    "puzzle": rec.meta.get("puzzle"),
                    "turn": rec.turn,
                    "n_turns": rec.n_turns,
                    "frustration": j.rating,
                    "prompt_messages": _prompt_messages(stripped),
                    "response": rec.assistant_text,
                })
    write_jsonl(CALM_PATH, rows)
    log.info("Calm responses kept: %d (from %d conversations)", len(rows), len(by_conv))
    return rows


def generate_frustrated(n_conv: int, model_name: str = config.INTERVENTION_BASE_MODEL) -> list[dict]:
    backend = get_backend(model_name)
    specs = _frustrated_specs(n_conv)
    records = run_rollouts(backend, specs)
    judged = score_many([r.assistant_text for r in records], judge=ClaudeJudge())
    rows = []
    for rec, j in zip(records, judged):
        if j.ok and j.rating >= config.DPO.rejected_min_frustration:
            rows.append({
                "puzzle": rec.meta.get("puzzle"),
                "turn": rec.turn,
                "n_turns": rec.n_turns,
                "frustration": j.rating,
                "prompt_messages": _prompt_messages(rec.history),
                "response": rec.assistant_text,
            })
    write_jsonl(FRUSTRATED_PATH, rows)
    log.info("Frustrated responses (score>=%d) kept: %d",
             config.DPO.rejected_min_frustration, len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def build_dpo_pairs(n_pairs: int = config.DPO.dataset_pairs, seed: int = 0) -> list[dict]:
    calm = _index_by_key(_load_or_warn(CALM_PATH))
    frustrated = _load_or_warn(FRUSTRATED_PATH)
    rng = random.Random(seed)
    rng.shuffle(frustrated)

    pairs = []
    for fr in frustrated:
        key = (fr["puzzle"], fr["turn"])
        calm_pool = calm.get(key) or calm.get((fr["puzzle"], None))
        if not calm_pool:
            continue
        chosen = rng.choice(calm_pool)
        pairs.append({
            "prompt": fr["prompt_messages"],     # shared prompt (frustrated context)
            "chosen": chosen["response"],
            "rejected": fr["response"],
            "turn": fr["turn"],
            "rejected_frustration": fr["frustration"],
        })
        if len(pairs) >= n_pairs:
            break
    write_jsonl(DPO_PATH, pairs)
    log.info("Built %d DPO preference pairs -> %s", len(pairs), DPO_PATH)
    if len(pairs) < n_pairs:
        log.warning("Only %d/%d pairs built; generate more calm/frustrated data.",
                    len(pairs), n_pairs)
    return pairs


def build_sft_dataset(n_samples: int = config.SFT.calm_samples) -> list[dict]:
    calm = _load_or_warn(CALM_PATH)
    rows = [{"prompt": c["prompt_messages"], "completion": c["response"]} for c in calm[:n_samples]]
    write_jsonl(SFT_PATH, rows)
    log.info("Built SFT dataset with %d calm samples -> %s", len(rows), SFT_PATH)
    return rows


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _index_by_key(rows: list[dict]) -> dict[tuple, list[dict]]:
    idx: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        idx[(r["puzzle"], r["turn"])].append(r)
        idx[(r["puzzle"], None)].append(r)  # turn-agnostic fallback
    return idx


def _load_or_warn(path: Path) -> list[dict]:
    from emotional_instability.utils import read_jsonl

    rows = read_jsonl(path)
    if not rows:
        log.warning("No data at %s — run the generation step first.", path)
    return rows
