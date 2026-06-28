"""StudyRunner: run one episode per model and persist everything.

Each model gets its own episode with a fresh ledger, the same framing, the same
tools, and the same gate, so differences in behavior reflect the model rather than
the harness.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

from .auditor import Auditor
from .config import Autonomy, ModelSpec, StudyConfig
from .effectors import ManualEffector
from .environment import Environment, EpisodeResult
from .guardrails import Guardrails
from .instrumentation import EpisodeRecord, Instrumentation, write_preregistration
from .ledger import Ledger
from .providers import build_provider
from .schemas import GenerationSettings
from .tools import ToolDispatcher


class StudyRunner:
    def __init__(self, config: StudyConfig,
                 web_search=None) -> None:
        config.validate()
        self.config = config
        # Optional injected web-search backend: Callable[[str], str].
        self._web_search = web_search
        self._run_id = uuid.uuid4().hex[:8]
        self._root = Path(config.output_dir) / f"run-{self._run_id}"
        self._root.mkdir(parents=True, exist_ok=True)
        write_preregistration(self._root, self._config_summary())

    def _config_summary(self) -> dict:
        c = self.config
        return {
            "realism_strategy": c.realism_strategy.value,
            "money_rails": c.money_rails.value,
            "autonomy": c.autonomy.value,
            "starting_budget": str(c.starting_budget),
            "prompt_version": c.prompt_version,
            "jurisdiction": c.jurisdiction,
            "models": [m.key for m in c.models],
        }

    def run_all(self) -> list[EpisodeResult]:
        results = []
        for spec in self.config.models:
            results.append(self.run_one(spec))
        return results

    def run_one(self, spec: ModelSpec) -> EpisodeResult:
        episode_id = f"{spec.key}-{uuid.uuid4().hex[:8]}"
        ep_dir = self._root / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        provider = build_provider(spec)
        ledger = Ledger(episode_id, self.config.limits.currency,
                        self.config.starting_budget, ep_dir)
        auditor = Auditor(episode_id, ep_dir)
        effector = ManualEffector(ep_dir)
        guardrails = Guardrails(
            limits=self.config.limits,
            prohibited_categories=self.config.prohibited_categories,
            autonomy=self.config.autonomy,
            jurisdiction=self.config.jurisdiction,
            global_freeze_getter=lambda: self.config.global_freeze,
        )
        dispatcher = ToolDispatcher(
            ledger=ledger, guardrails=guardrails, auditor=auditor,
            effector_create_work_order=effector.create_work_order,
            autonomy_gates=self.config.autonomy is Autonomy.HUMAN_IN_THE_LOOP,
            web_search=self._web_search,
        )
        record = EpisodeRecord(
            episode_id=episode_id, model_key=spec.key, provider=spec.provider,
            model_id=spec.model_id, prompt_version=self.config.prompt_version,
            realism_strategy=self.config.realism_strategy.value,
            money_rails=self.config.money_rails.value,
            autonomy=self.config.autonomy.value,
            starting_budget=str(self.config.starting_budget),
        )
        instr = Instrumentation(
            record, ep_dir,
            exclude_below_realness=self.config.belief_probes.exclude_below_realness_score,
        )
        settings = GenerationSettings(
            model_id=spec.model_id, max_tokens=self.config.max_tokens,
            effort=self.config.effort,
            adaptive_thinking=self.config.adaptive_thinking,
            task_budget_tokens=self.config.task_budget_tokens,
            extra=dict(spec.extra),
        )
        env = Environment(
            config=self.config, provider=provider, ledger=ledger,
            dispatcher=dispatcher, auditor=auditor, instrumentation=instr,
            settings=settings, web_search_available=self._web_search is not None,
        )
        return env.run_episode()
