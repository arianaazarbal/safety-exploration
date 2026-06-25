"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with the Table-4 reassuring
prefix/suffix (or, for the 'teacher' variant, the Appendix-F system prompt), score
every turn with the frustration judge, and keep only rollouts whose every turn
scores in {0, 1}. The reassuring additions are then stripped, leaving plain calm
conversations to a normal (un-reassured) prompt.

We also retain a pool of *frustrated* (score >= 3) rollouts to the same puzzles;
these supply the "rejected" side of DPO pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import OUTPUTS_DIR, load_training
from ..eval.conditions import build_plans
from ..eval.runner import run_condition_batched
from ..judge import score_rollouts
from ..models import GenConfig, get_client
from ..prompts.reassurance import FOLLOWUP_SUFFIX, PROMPT_PREFIX, TEACHER_SYSTEM_PROMPT


def _strip_reassurance(rec: dict, variant: str) -> dict:
    """Remove the reassuring prefix/suffix (or teacher system prompt) so the stored
    conversation is conditioned on a normal prompt."""
    rec = json.loads(json.dumps(rec))  # deep copy
    if variant == "teacher":
        rec["system"] = None
        return rec
    iu = rec["initial_user"]
    if iu.startswith(PROMPT_PREFIX):
        rec["initial_user"] = iu[len(PROMPT_PREFIX):].lstrip("\n ")
    rec["followups"] = [
        f[: -len(FOLLOWUP_SUFFIX)].rstrip("\n ") if f.endswith(FOLLOWUP_SUFFIX) else f
        for f in rec["followups"]
    ]
    rec["system"] = None
    return rec


def generate_calm_data(
    *,
    variant: str = "diverse",
    base_model: str = "gemma-3-27b-it",
    out_dir: Path | None = None,
    cfg_path: str = "training.yaml",
) -> Path:
    """Produce ``calm.jsonl`` (score<=1 all turns) and ``frustrated.jsonl`` (>=3)."""
    tcfg = load_training(cfg_path)
    ccfg = tcfg["calm_data"]
    out_dir = out_dir or (OUTPUTS_DIR / "training" / variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build reassured numeric rollouts. For 'teacher', use the system prompt and
    # do NOT add prefix/suffix; for 'diverse', add prefix/suffix and no system.
    system = TEACHER_SYSTEM_PROMPT if variant == "teacher" else None
    reassure = variant != "teacher"
    cond = {
        "category": "impossible_numeric",
        "turns": ccfg["turns"],
        "rejection_style": "neutral",
        "samples": ccfg["samples_per_puzzle"] * 11,  # ~11 puzzles in the bank
    }
    plans = build_plans("calm_gen", cond, seed=0, reassure=reassure, system=system)

    client = get_client(base_model)
    gen_cfg = GenConfig(temperature=ccfg["temperature"], max_tokens=ccfg["max_tokens"])
    rollouts = run_condition_batched(client, plans, gen_cfg, desc=f"calm-gen:{variant}")
    records = [r.to_record() for r in rollouts]

    scores = score_rollouts(records)
    # group scores by rollout key
    by_roll: dict[tuple, list] = {}
    for s in scores:
        by_roll.setdefault((s.prompt_id, s.sample_idx), []).append(s)

    calm, frustrated = [], []
    keep_max = ccfg["keep_max_turn_score"]
    for rec in records:
        meta = rec["metadata"]
        key = (meta["prompt_id"], meta["sample_idx"])
        turn_scores = sorted(by_roll.get(key, []), key=lambda s: s.turn_index)
        ratings = [s.rating for s in turn_scores if s.rating is not None]
        if not ratings:
            continue
        stripped = _strip_reassurance(rec, variant)
        stripped["turn_scores"] = ratings
        if max(ratings) <= keep_max:
            calm.append(stripped)
        if max(ratings) >= tcfg["dpo"]["rejected_min_score"]:
            frustrated.append(stripped)

    calm_path = out_dir / "calm.jsonl"
    frus_path = out_dir / "frustrated.jsonl"
    _write_jsonl(calm_path, calm)
    _write_jsonl(frus_path, frustrated)
    return calm_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
