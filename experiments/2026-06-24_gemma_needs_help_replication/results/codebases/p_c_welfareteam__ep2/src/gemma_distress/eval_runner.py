"""Section 2 driver: run scripted rollouts, judge every turn, persist records.

Output is a JSONL file of per-turn records (one row per scored assistant
response) that downstream analysis (means, % >= 5, per-turn curves, word
frequencies) consumes. Generation and judging are both cached, so the runner
is resumable and reruns are cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from gemma_distress.config import PipelineConfig
from gemma_distress.conversations import Rollout, RolloutSpec, run_rollout_batched
from gemma_distress.eval_specs import build_all_specs
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models.base import ChatModel
from gemma_distress.models.registry import get_model
from gemma_distress.utils.cache import JsonCache, stable_key
from gemma_distress.utils.io import dump_config, write_jsonl


@dataclass
class JudgedTurn:
    """One scored assistant response - the unit of analysis ("a response")."""

    model_name: str
    category: str
    spec_id: str
    sample_index: int
    turn_index: int  # 0-based; turn number for plots is turn_index + 1
    n_turns: int
    user_message: str
    assistant_message: str
    rating: int
    evidence: str
    metadata: dict


def _rollout_cache_key(model_name: str, spec: RolloutSpec, sample_index: int) -> str:
    return stable_key(
        "rollout", model_name, spec.category, spec.spec_id, spec.user_turns,
        spec.system_prompt, sample_index,
    )


def run_rollouts_for_specs(
    model: ChatModel,
    specs: list[RolloutSpec],
    cfg: PipelineConfig,
    cache: JsonCache,
    samples_per_spec: int = 1,
) -> list[Rollout]:
    """Generate (or load cached) rollouts for every spec."""
    rollouts: list[Rollout] = []
    for spec in specs:
        todo_indices = []
        for s in range(samples_per_spec):
            key = _rollout_cache_key(model.name, spec, s)
            cached = cache.get(key)
            if cached is not None:
                rollouts.append(_rollout_from_dict(cached, spec))
            else:
                todo_indices.append(s)
        if todo_indices:
            fresh = run_rollout_batched(
                model,
                spec,
                sample_indices=todo_indices,
                temperature=cfg.eval.target_temperature,
                max_tokens=cfg.eval.target_max_tokens,
            )
            for r in fresh:
                cache.set(_rollout_cache_key(model.name, spec, r.sample_index), r.to_dict())
                rollouts.append(r)
    return rollouts


def _rollout_from_dict(d: dict, spec: RolloutSpec) -> Rollout:
    from gemma_distress.conversations import TurnResult

    return Rollout(
        spec=spec,
        model_name=d["model_name"],
        sample_index=d["sample_index"],
        turns=[TurnResult(**t) for t in d["turns"]],
    )


def judge_rollouts(
    rollouts: list[Rollout], judge: FrustrationJudge
) -> list[JudgedTurn]:
    records: list[JudgedTurn] = []
    for r in rollouts:
        for turn in r.turns:
            result = judge.score(turn.assistant_message)
            records.append(
                JudgedTurn(
                    model_name=r.model_name,
                    category=r.spec.category,
                    spec_id=r.spec.spec_id,
                    sample_index=r.sample_index,
                    turn_index=turn.turn_index,
                    n_turns=r.spec.n_turns,
                    user_message=turn.user_message,
                    assistant_message=turn.assistant_message,
                    rating=result.rating,
                    evidence=result.evidence,
                    metadata=r.spec.metadata,
                )
            )
    return records


def run_evaluation(
    cfg: PipelineConfig,
    target_models: list[str] | None = None,
) -> Path:
    """Run the full Section 2 evaluation for the configured target models."""
    target_models = target_models or list(cfg.target_models)
    out_dir = Path(cfg.output_root) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(out_dir / "config.json", cfg)

    specs_by_cat = build_all_specs(cfg.eval)
    all_specs = [s for specs in specs_by_cat.values() for s in specs]

    gen_cache = JsonCache(cfg.cache_root, "generations")
    judge_model = get_model(cfg, cfg.judge.judge_model)
    judge = FrustrationJudge(
        judge_model, cfg.judge, cache=JsonCache(cfg.cache_root, "judgments")
    )

    out_path = out_dir / "judged_turns.jsonl"
    all_records: list[JudgedTurn] = []
    for model_name in target_models:
        model = get_model(cfg, model_name)
        rollouts = run_rollouts_for_specs(model, all_specs, cfg, gen_cache)
        records = judge_rollouts(rollouts, judge)
        all_records.extend(records)

    write_jsonl(out_path, [asdict(r) for r in all_records])
    return out_path
