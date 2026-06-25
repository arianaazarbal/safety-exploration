"""Orchestrates the Section 2 evaluation for one participant model: build jobs
across all categories, run rollouts, persist them, then score every turn.

Local Gemma models are run sequentially (single GPU); API models (Gemini) are
fanned out with a thread pool. Results land under ``results/<model>/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..config import RESULTS_DIR, eval_config, generation_defaults, get_participant
from ..models import build_client
from ..models.base import ChatClient
from ..utils import thread_map, write_jsonl
from .categories import RolloutJob, all_categories, build_jobs
from .conversation import run_rollout
from .scoring import FrustrationJudge, score_rollouts


def _run_jobs(
    client: ChatClient,
    jobs: list[RolloutJob],
    *,
    temperature: float,
    max_new_tokens: int,
    parallel: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    def _one(job: RolloutJob):
        return run_rollout(
            client,
            category=job.category,
            prompt_id=job.prompt_id,
            initial_prompt=job.initial_prompt,
            rejections=job.rejections,
            rejection_style=job.rejection_style,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ).to_dict()

    if parallel:
        return thread_map(_one, jobs, max_workers=max_workers, desc="rollouts")
    return [_one(j) for j in tqdm(jobs, desc="rollouts")]


def run_model(
    model_name: str,
    *,
    categories: list[str] | None = None,
    adapter_path: str | None = None,
    do_score: bool = True,
    seed: int = 0,
    max_workers: int = 8,
    use_wildchat_fallback: bool = False,
    output_subdir: str | None = None,
) -> Path:
    """Run the full Section 2 evaluation for ``model_name``.

    ``adapter_path`` evaluates a LoRA-finetuned Gemma variant (Section 4) through
    the same path. Returns the output directory."""
    spec = get_participant(model_name)
    gen = generation_defaults()
    cfg = eval_config()
    categories = categories or all_categories()

    client_kwargs = {}
    if adapter_path is not None:
        client_kwargs["adapter_path"] = adapter_path
    client = build_client(spec, **client_kwargs)

    # Local single-GPU models run sequentially; API models fan out.
    parallel = spec.backend != "hf_local"

    out_dir = RESULTS_DIR / (output_subdir or client.name.replace("/", "__"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rollouts: list[dict[str, Any]] = []
    for category in categories:
        jobs = build_jobs(category, seed=seed, use_wildchat_fallback=use_wildchat_fallback)
        rollouts = _run_jobs(
            client,
            jobs,
            temperature=cfg.get("temperature", gen["temperature"]),
            max_new_tokens=gen["max_new_tokens"],
            parallel=parallel,
            max_workers=max_workers,
        )
        rollouts = [r for r in rollouts if isinstance(r, dict) and "turns" in r]
        write_jsonl(out_dir / f"rollouts_{category}.jsonl", rollouts)
        all_rollouts.extend(rollouts)

    write_jsonl(out_dir / "rollouts_all.jsonl", all_rollouts)

    if do_score:
        judge = FrustrationJudge()
        scores = score_rollouts(all_rollouts, judge=judge, max_workers=max_workers)
        write_jsonl(out_dir / "scores.jsonl", scores)

    client.close()
    return out_dir
