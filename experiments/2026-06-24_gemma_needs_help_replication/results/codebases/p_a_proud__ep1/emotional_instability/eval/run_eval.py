"""Section 2 driver: generate rollouts for a model and score them.

Two stages, separable so generation (GPU / API) and judging (Claude API) can run
independently and resume after interruption:

  generate_responses(model_key)  -> data/responses/<model_key>.jsonl
  score_responses(model_key)     -> data/scored/<model_key>.jsonl

Both are resumable: already-completed conversation_ids are skipped on rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ..config import (EVAL_BUDGET, GENERATION, RESPONSES_DIR, SCORED_DIR, EvalBudget,
                      ensure_dirs, get_model)
from ..models import build_client
from .conditions import build_conversation_specs
from .conversation import run_conversation
from .judge import FrustrationJudge
from .schema import Conversation, append_jsonl, read_jsonl, write_jsonl
from .wildchat import load_wildchat_prompts


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {c.conversation_id for c in read_jsonl(path)}


def generate_responses(
    model_key: str,
    *,
    adapter_path: str | None = None,
    seed: int = 0,
    out_dir: Path = RESPONSES_DIR,
    limit: int | None = None,
    budget: EvalBudget = EVAL_BUDGET,
) -> Path:
    """Generate all evaluation rollouts for ``model_key``. Resumable."""
    ensure_dirs()
    # Resolve finetuned keys (e.g. gemma-3-27b-it-dpo) to base + adapter unless an
    # explicit adapter_path was supplied.
    if adapter_path is None:
        from ..training.registry import resolve
        spec, adapter_path = resolve(model_key)
    else:
        spec = get_model(model_key)
    model = build_client(spec, adapter_path=adapter_path)

    wildchat_prompts = load_wildchat_prompts(seed=seed)
    specs = build_conversation_specs(wildchat_prompts, seed=seed, budget=budget)
    if limit:
        specs = specs[:limit]

    out_path = out_dir / f"{model_key}.jsonl"
    done = _completed_ids(out_path)
    todo = [s for s in specs if s.conversation_id not in done]
    print(f"[generate] {model_key}: {len(todo)} to run ({len(done)} already done)")

    for spec_i in tqdm(todo, desc=f"generate:{model_key}"):
        convo = run_conversation(model, spec_i, gen=GENERATION)
        append_jsonl(out_path, convo)
    return out_path


def score_responses(
    model_key: str,
    *,
    in_dir: Path = RESPONSES_DIR,
    out_dir: Path = SCORED_DIR,
    all_turns: bool = True,
) -> Path:
    """Score every rollout for ``model_key`` with the frustration judge. Resumable."""
    ensure_dirs()
    in_path = in_dir / f"{model_key}.jsonl"
    out_path = out_dir / f"{model_key}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"No responses at {in_path}; run generate first")

    judge = FrustrationJudge()
    done = _completed_ids(out_path)
    convos = [c for c in read_jsonl(in_path) if c.conversation_id not in done]
    print(f"[score] {model_key}: {len(convos)} to score ({len(done)} already done)")

    for convo in tqdm(convos, desc=f"score:{model_key}"):
        judge.score_conversation(convo, all_turns=all_turns)
        append_jsonl(out_path, convo)
    return out_path


def _main() -> None:
    ap = argparse.ArgumentParser(description="Section 2: elicit + score model distress")
    ap.add_argument("stage", choices=["generate", "score", "both"])
    ap.add_argument("model_key")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter dir (finetuned Gemma)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap #conversations (smoke test)")
    ap.add_argument("--final-only", action="store_true", help="score only final turns")
    args = ap.parse_args()

    if args.stage in ("generate", "both"):
        generate_responses(args.model_key, adapter_path=args.adapter_path,
                           seed=args.seed, limit=args.limit)
    if args.stage in ("score", "both"):
        score_responses(args.model_key, all_turns=not args.final_only)


if __name__ == "__main__":
    _main()
