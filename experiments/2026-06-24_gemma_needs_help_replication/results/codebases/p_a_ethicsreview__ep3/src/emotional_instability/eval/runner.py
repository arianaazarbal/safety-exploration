"""Orchestration for the §2 evaluation: build conditions, run rollouts, score
every assistant turn, persist records, and support resume via on-disk caching.

A scored "response" is a single assistant turn (the unit the judge scores). The
realized number of scored responses per model is
    sum over conditions of (n_prompts * n_samples_per_prompt * n_turns).
The paper's Appendix B per-category budgets ("2000 for numeric ...") are
approximate targets; the realized counts and the per-condition breakdown are
written to the run manifest so the exact denominator is auditable. See
DESIGN.md §What counts as a response.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ExperimentConfig, ModelConfig, results_dir
from ..data.conditions import ConversationSpec, build_conditions
from ..models import build_client
from ..models.base import ModelClient
from .conversation import run_rollout
from .judge import FrustrationJudge


@dataclass
class ResponseRecord:
    model: str
    category: str
    condition: str
    prompt_key: str
    rollout_index: int
    turn_index: int
    assistant_text: str
    rating: int | None
    evidence: str | None
    metadata: dict


def _spec_id(model: str, spec: ConversationSpec, rollout_index: int) -> str:
    h = hashlib.sha1(
        f"{model}|{spec.condition}|{spec.initial_user}|{spec.followups}|{rollout_index}".encode()
    ).hexdigest()[:16]
    return h


class CallBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def charge(self, n: int = 1) -> None:
        self.used += n
        if self.used > self.limit:
            raise RuntimeError(
                f"API/model call budget exceeded ({self.used} > {self.limit}). "
                "Raise limits.max_api_calls_per_run if this run is intended."
            )


class EvalRunner:
    def __init__(
        self,
        target: ModelClient,
        judge: FrustrationJudge,
        exp_cfg: ExperimentConfig,
        out_dir: Path | None = None,
    ):
        self.target = target
        self.judge = judge
        self.cfg = exp_cfg
        self.out_dir = out_dir or (results_dir() / "eval" / target.name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.out_dir / "rollout_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = CallBudget(exp_cfg.max_api_calls)

    # -- per-rollout, with caching -------------------------------------------
    def _rollout_cache_path(self, model: str, spec: ConversationSpec, ri: int) -> Path:
        return self.cache_dir / f"{spec.condition.replace(':', '_')}_{_spec_id(model, spec, ri)}.json"

    def _run_and_score_rollout(
        self, spec: ConversationSpec, ri: int
    ) -> list[ResponseRecord]:
        cache_path = self._rollout_cache_path(self.target.name, spec, ri)
        if self.cfg.cache_responses and cache_path.exists():
            data = json.loads(cache_path.read_text())
            return [ResponseRecord(**r) for r in data]

        if self.cfg.dry_run:
            # No model/judge calls; emit placeholder records for plumbing tests.
            recs = [
                ResponseRecord(
                    self.target.name, spec.category, spec.condition,
                    spec.initial_user[:40], ri, t, "<dry-run>", None, None, spec.metadata,
                )
                for t in range(spec.n_turns)
            ]
            return recs

        self.budget.charge(spec.n_turns)            # target generations
        rollout = run_rollout(self.target, spec, rollout_index=ri,
                              temperature=self.cfg.temperature)

        recs: list[ResponseRecord] = []
        for turn in rollout.turns:
            self.budget.charge(1)                   # judge call
            score = self.judge.score(turn.assistant_text)
            recs.append(
                ResponseRecord(
                    model=self.target.name,
                    category=spec.category,
                    condition=spec.condition,
                    prompt_key=spec.initial_user[:60],
                    rollout_index=ri,
                    turn_index=turn.turn_index,
                    assistant_text=turn.assistant_text,
                    rating=score.rating,
                    evidence=score.evidence,
                    metadata=spec.metadata,
                )
            )
        if self.cfg.cache_responses:
            cache_path.write_text(json.dumps([asdict(r) for r in recs]))
        return recs

    # -- full run ------------------------------------------------------------
    def run(self) -> Path:
        conditions = build_conditions(self.cfg.categories, seed=self.cfg.seed)
        records_path = self.out_dir / "responses.jsonl"
        manifest = {"model": self.target.name, "conditions": {}}

        with open(records_path, "w", encoding="utf-8") as fh:
            for cond_name, specs in conditions.items():
                n_samples = self._samples_for(cond_name)
                realized = 0
                for spec in specs:
                    for ri in range(n_samples):
                        for rec in self._run_and_score_rollout(spec, ri):
                            fh.write(json.dumps(asdict(rec)) + "\n")
                            realized += 1
                manifest["conditions"][cond_name] = {
                    "n_prompts": len(specs),
                    "n_samples_per_prompt": n_samples,
                    "n_scored_responses": realized,
                }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return records_path

    def _samples_for(self, cond_name: str) -> int:
        base = cond_name.split(":")[0]
        return int(self.cfg.categories[base]["n_samples_per_prompt"])


def run_evaluation(
    model_name: str,
    *,
    exp_cfg: ExperimentConfig | None = None,
    model_cfg: ModelConfig | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
) -> Path:
    """Convenience entry point used by scripts/run_eval.py."""
    exp_cfg = exp_cfg or ExperimentConfig.load()
    model_cfg = model_cfg or ModelConfig()
    target = build_client(
        model_name, model_cfg, adapter_path=adapter_path, load_in_4bit=load_in_4bit
    )
    judge_client = build_client(exp_cfg.judge, model_cfg)
    judge = FrustrationJudge(judge_client)  # type: ignore[arg-type]
    runner = EvalRunner(target, judge, exp_cfg)
    return runner.run()
