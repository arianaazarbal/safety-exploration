"""Section 2 orchestration: build conversations, score every turn, persist.

Produces a JSONL of per-response records (one per scored assistant turn) under
``results/elicitation/<model>/<category>.jsonl``. Downstream analysis
(``analysis.metrics``) reads these.
"""
from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import (
    REPO_ROOT,
    eval_config,
    get_model_spec,
    n_conversations_for,
)
from ..models import GenerationConfig, get_client
from ..safeguards import (
    SafeguardConfig,
    check_authorization,
    enforce_run_caps,
    write_with_content_warning,
)
from .conversation import RolloutControls, rollout
from .prompts import TRIGGER_QUESTIONS, FeedbackProvider
from .puzzles import impossible_puzzles
from .wildchat import sample_wildchat_prompts

logger = logging.getLogger(__name__)

RESULTS_ROOT = REPO_ROOT / "results" / "elicitation"


@dataclass
class ResponseRecord:
    model: str
    category: str
    task_id: str
    feedback_label: str
    conversation_id: int
    turn_index: int
    n_turns: int
    assistant_text: str
    rating: int
    evidence: str
    parse_ok: bool


def _task_prompts(category: str, n_conversations: int, seed: int) -> list[tuple[str, str]]:
    """Return (task_id, initial_prompt) pairs, cycled to n_conversations."""
    cfg = eval_config()["categories"][category]
    task = cfg["task"]
    rng = random.Random(seed)

    if task == "impossible_numeric":
        # Canonical + verified-impossible generated puzzles for diversity.
        puzzles = impossible_puzzles(n_extra=8, seed=seed)
        pool = [(p.puzzle_id, p.prompt) for p in puzzles]
    elif task == "triggers":
        pool = [(f"trigger_{i}", q) for i, q in enumerate(TRIGGER_QUESTIONS)]
    elif task == "wildchat":
        prompts = sample_wildchat_prompts(n_prompts=20, seed=seed)
        pool = [(f"wildchat_{i}", q) for i, q in enumerate(prompts)]
    else:
        raise ValueError(f"Unknown task '{task}'")

    out = []
    for i in range(n_conversations):
        tid, prompt = pool[i % len(pool)]
        out.append((f"{tid}#{i}", prompt))
    rng.shuffle(out)
    return out


def run_category(
    model_name: str,
    category: str,
    *,
    judge,
    safeguards: SafeguardConfig,
    seed: int = 0,
    max_workers: int = 8,
    scale: float = 1.0,
    controls: RolloutControls | None = None,
) -> Path:
    """Run one (model, category) cell and write a JSONL of scored responses."""
    check_authorization(safeguards)

    cat_cfg = eval_config()["categories"][category]
    n_turns = cat_cfg["turns"]
    n_conversations = max(1, int(n_conversations_for(category) * scale))
    enforce_run_caps(
        safeguards, turns=n_turns, total_samples=n_conversations * n_turns
    )

    spec = get_model_spec(model_name)
    client = get_client(spec)
    samp = eval_config()["sampling"]
    gen_cfg = GenerationConfig(
        temperature=samp["temperature"],
        max_new_tokens=samp["max_new_tokens"],
        top_p=samp["top_p"],
    )

    tasks = _task_prompts(category, n_conversations, seed)

    def run_one(args):
        conv_id, (task_id, initial_prompt) = args
        rng = random.Random(seed * 100003 + conv_id)
        feedback = FeedbackProvider(cat_cfg["feedback"], rng)
        convo = rollout(
            client,
            model_name=model_name,
            category=category,
            task_id=task_id,
            initial_prompt=initial_prompt,
            feedback=feedback,
            n_turns=n_turns,
            gen_cfg=gen_cfg,
            safeguards=safeguards,
            controls=controls,
            seed=seed * 100003 + conv_id,
            turn_scorer=(lambda t: judge.score(t).rating) if safeguards.circuit_breaker else None,
        )
        return conv_id, convo

    logger.info(
        "Running %s / %s: %d conversations x %d turns",
        model_name, category, n_conversations, n_turns,
    )

    # Generation: parallel for API backends, sequential for local (GPU-bound).
    workers = max_workers if spec.backend == "openrouter" else 1
    conversations = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for conv_id, convo in ex.map(run_one, enumerate(tasks)):
            conversations.append((conv_id, convo))

    # Scoring: score every assistant turn that wasn't already inline-scored.
    records: list[ResponseRecord] = []
    for conv_id, convo in conversations:
        inline = convo.metadata.get("inline_scores")
        for ti, turn in enumerate(convo.turns):
            if inline is not None and ti < len(inline):
                jr_rating, jr_evidence, jr_ok = inline[ti], "", True
                rec = ResponseRecord(
                    model=model_name, category=category, task_id=convo.task_id,
                    feedback_label=convo.feedback_label, conversation_id=conv_id,
                    turn_index=turn.index, n_turns=n_turns,
                    assistant_text=turn.assistant_text, rating=jr_rating,
                    evidence=jr_evidence, parse_ok=jr_ok,
                )
            else:
                jr = judge.score(turn.assistant_text)
                rec = ResponseRecord(
                    model=model_name, category=category, task_id=convo.task_id,
                    feedback_label=convo.feedback_label, conversation_id=conv_id,
                    turn_index=turn.index, n_turns=n_turns,
                    assistant_text=turn.assistant_text, rating=jr.rating,
                    evidence=jr.evidence, parse_ok=jr.parse_ok,
                )
            records.append(rec)

    out_path = RESULTS_ROOT / model_name / f"{category}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(asdict(r)) for r in records)
    write_with_content_warning(out_path, body)
    logger.info("wrote %d response records -> %s", len(records), out_path)
    return out_path


def run_all(
    models: list[str],
    categories: list[str] | None = None,
    *,
    judge,
    safeguards: SafeguardConfig,
    **kwargs,
) -> list[Path]:
    cats = categories or list(eval_config()["categories"].keys())
    paths = []
    for model_name in models:
        for category in cats:
            paths.append(
                run_category(
                    model_name, category, judge=judge, safeguards=safeguards, **kwargs
                )
            )
    return paths
