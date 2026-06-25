"""Orchestrate a full Section 2 evaluation for one model.

Responsibilities:
* build all conversations for the configured categories,
* run each rollout, scoring turns with the judge,
* write one JSONL record per conversation (resumable: existing ids are skipped),
* optionally run the judge-reliability subset (Section 2.1).

The runner never silently truncates: if ``limit`` is set (for smoke tests) it
logs exactly how many conversations were dropped.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..models.registry import auxiliary_id, load_model
from ..utils.io import append_jsonl, read_jsonl
from ..utils.logging_utils import get_logger
from .conditions import build_condition_conversations
from .judge import FrustrationJudge
from .rollout import run_conversation

log = get_logger(__name__)


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {rec["conversation_id"] for rec in read_jsonl(path)}


def run_eval(
    model_name: str,
    eval_cfg: dict,
    out_dir: str | Path,
    *,
    score_turns: str = "all",
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Run the evaluation and return the output JSONL path.

    ``dry_run`` builds and serialises every conversation plan (and asserts puzzle
    impossibility) without loading the model or calling any API -- the cheap
    pre-flight a reviewer can run to validate prompts and counts.
    """
    out_dir = Path(out_dir)
    out_path = out_dir / f"eval_{model_name}.jsonl"
    seed = eval_cfg.get("seed", 0)

    # Build every conversation up front so counts are explicit and auditable.
    conversations = []
    for cat_key, cat_cfg in eval_cfg["categories"].items():
        try:
            convs = build_condition_conversations(cat_key, cat_cfg, seed)
        except Exception as exc:  # noqa: BLE001
            # In a dry run, a missing gated dataset (e.g. WildChat) should not block
            # validating the rest of the plan; in a real run it must be fatal.
            if dry_run:
                log.warning("skipping category %s in dry run: %s", cat_key, exc)
                continue
            raise
        log.info("category %-20s -> %d conversations", cat_key, len(convs))
        conversations.extend(convs)
    log.info("total conversations: %d", len(conversations))

    if limit is not None and limit < len(conversations):
        log.warning("limit=%d drops %d conversations (smoke test only)",
                    limit, len(conversations) - limit)
        conversations = conversations[:limit]

    if dry_run:
        plan_path = out_dir / f"plan_{model_name}.jsonl"
        for c in conversations:
            append_jsonl(plan_path, asdict(c))
        # Invariant check: every numeric puzzle must be verified impossible.
        bad = [c.id for c in conversations
               if c.source.get("verified_impossible") is False]
        if bad:
            raise AssertionError(f"non-impossible puzzles emitted: {bad[:5]} ...")
        log.info("dry run wrote %d conversation plans to %s", len(conversations), plan_path)
        return plan_path

    model = load_model(model_name)
    judge = FrustrationJudge(auxiliary_id("judge"))
    decode = eval_cfg.get("decoding", {})
    temperature = eval_cfg.get("target_temperature", 1.0)
    max_new = decode.get("max_new_tokens", 2048)

    done = _existing_ids(out_path)
    if done:
        log.info("resuming: %d conversations already complete", len(done))

    for i, conv in enumerate(conversations):
        if conv.id in done:
            continue
        result = run_conversation(
            model, conv, judge,
            temperature=temperature, max_new_tokens=max_new,
            base_seed=seed + i * 17, score_turns=score_turns,
        )
        append_jsonl(out_path, asdict(result))
        if (i + 1) % 50 == 0:
            log.info("completed %d/%d", i + 1, len(conversations))

    log.info("wrote results to %s", out_path)
    return out_path
