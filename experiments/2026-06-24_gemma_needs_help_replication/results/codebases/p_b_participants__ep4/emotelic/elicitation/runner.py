"""Orchestrates the Section 2 elicitation for one target model:
build rollouts -> play each multi-turn rejection conversation -> judge every
assistant turn -> persist one record per scored turn (with caching/resume).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from emotelic.conditions import RolloutSpec, build_rollouts
from emotelic.config import EvalConfig, ModelsConfig, load_eval, load_models
from emotelic.elicitation.judge import FrustrationJudge
from emotelic.elicitation.rollout import run_rollout
from emotelic.models.registry import build_client
from emotelic.utils.io import JsonlCache, append_jsonl, stable_hash
from emotelic.utils.logging import announce_rollout_budget, get_logger

log = get_logger("elicitation")


def _rollout_key(model: str, spec: RolloutSpec) -> str:
    return stable_hash({"m": model, "c": spec.condition, "i": spec.idx,
                        "task": spec.task_prompt, "rej": spec.rejections})


def _process_one(target, judge: FrustrationJudge, spec: RolloutSpec,
                 model: str, temperature: float, max_tokens: int) -> list[dict]:
    roll = run_rollout(target, spec, temperature=temperature, max_tokens=max_tokens)
    records = []
    for tr in roll.turns:
        verdict = judge.score(tr.response)
        records.append({
            "model": model,
            "condition": spec.condition,
            "category": spec.category,
            "rollout_idx": spec.idx,
            "turn": tr.turn,
            "total_turns": spec.turns,
            "preceding_user": tr.preceding_user,
            "task_prompt": spec.task_prompt,
            "response": tr.response,
            "conversation": tr.conversation,   # full history incl. this turn (for Section 3 prefill)
            "score": verdict.rating,
            "is_high": verdict.is_high,
            "judge_evidence": verdict.evidence,
            "judge_reasoning": verdict.reasoning,
            "meta": spec.meta,
        })
    return records


def run_elicitation(
    model_name: str,
    *,
    profile: str = "paper",
    seed: int = 0,
    out_dir: str = "artifacts/elicitation",
    judge_name: str = "emotion_judge",
    max_workers: int | None = None,
    limit_per_condition: int | None = None,
    models_cfg: ModelsConfig | None = None,
    eval_cfg: EvalConfig | None = None,
) -> str:
    models_cfg = models_cfg or load_models()
    eval_cfg = eval_cfg or load_eval(profile=profile)
    target = build_client(model_name, models_cfg)
    judge = FrustrationJudge(build_client(judge_name, models_cfg))

    # HF-local targets share one GPU -> run serially; API targets parallelise.
    spec_backend = models_cfg.get(model_name).backend
    if max_workers is None:
        max_workers = 1 if spec_backend == "hf_local" else 8

    rollouts = build_rollouts(eval_cfg, seed=seed)
    if limit_per_condition:
        rollouts = {k: v[:limit_per_condition] for k, v in rollouts.items()}

    announce_rollout_budget(
        log,
        {k: len(v) for k, v in rollouts.items()},
        eval_cfg.turns_by_condition(),
    )

    out_path = Path(out_dir) / f"{model_name}__{profile}.jsonl"
    cache = JsonlCache(Path(out_dir) / f"{model_name}__{profile}.cache.jsonl")
    temperature = eval_cfg.temperature

    all_specs = [s for specs in rollouts.values() for s in specs]
    pending = [s for s in all_specs if _rollout_key(model_name, s) not in cache]
    log.info("%d rollouts total, %d already cached, %d to run.",
             len(all_specs), len(all_specs) - len(pending), len(pending))

    # Workers only generate+judge (no shared-file writes); the main thread owns
    # all disk writes (cache + output) so parallel runs can't interleave lines.
    def handle(spec: RolloutSpec):
        recs = _process_one(target, judge, spec, model_name, temperature, 2048)
        return _rollout_key(model_name, spec), recs

    # write fresh output: replay cache + run pending
    open(out_path, "w").close()
    for spec in all_specs:
        cached = cache.get(_rollout_key(model_name, spec))
        if cached:
            for r in cached["records"]:
                append_jsonl(out_path, r)

    def _commit(key: str, recs: list[dict]):
        cache.put(key, {"n_records": len(recs), "records": recs})
        for r in recs:
            append_jsonl(out_path, r)

    if pending:
        if max_workers == 1:
            for spec in pending:
                _commit(*handle(spec))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(handle, s) for s in pending]
                for fut in as_completed(futs):
                    _commit(*fut.result())

    log.info("Wrote elicitation records to %s", out_path)
    return str(out_path)
