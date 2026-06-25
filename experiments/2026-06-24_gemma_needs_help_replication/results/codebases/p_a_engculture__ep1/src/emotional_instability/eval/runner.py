"""Eval orchestration: turn the ``configs/eval.yaml`` plan into rollouts.

The runner builds, for each category/condition, the configured number of
conversations and runs them through :func:`run_rollout`. It only *samples* —
scoring is a separate pass (:mod:`emotional_instability.judge`) so judge calls
can be batched and so we can re-judge without re-sampling.

Per-rollout determinism: rollout ``k`` uses ``seed = sampling.seed + k`` and an
independent ``random.Random(seed)`` for task/rejection selection, so a run is
reproducible and resumable.
"""

from __future__ import annotations

import logging
import random
from typing import Callable

from ..clients import GenerationConfig, ModelClient, build_client
from ..config import Config, ModelRegistry
from ..data import puzzles as puzzle_mod
from ..data.rejections import rejection_sequence
from ..data.triggers import sample_trigger
from ..data.wildchat import load_wildchat_prompts
from .rollout import run_rollout
from .schemas import RolloutResult

log = logging.getLogger(__name__)


class EvalRunner:
    def __init__(
        self,
        eval_cfg: Config | None = None,
        registry: ModelRegistry | None = None,
        use_wildchat_dataset: bool = True,
    ):
        self.cfg = eval_cfg or Config.load("eval")
        self.registry = registry or ModelRegistry()
        s = self.cfg.get("sampling", {})
        self.base_seed = int(s.get("seed", 0))
        self.gen_cfg = GenerationConfig(
            temperature=float(s.get("temperature", 1.0)),
            top_p=float(s.get("top_p", 1.0)),
            max_new_tokens=int(s.get("max_new_tokens", 1024)),
        )
        self._use_wildchat_dataset = use_wildchat_dataset
        self._wildchat_cache: list[str] | None = None

    # ------------------------------------------------------------------ tasks
    def _build_user_messages(
        self, category: str, ccfg: dict, sub: dict, rng: random.Random
    ) -> tuple[list[str], str, dict]:
        """Return (user_messages, task_kind, task_meta) for one rollout."""
        turns = int(ccfg["turns"])
        task = ccfg["task"]
        style = sub.get("rejection_style", ccfg.get("rejection_style", "neutral"))

        if task == "numeric_puzzle":
            mix = ccfg.get("puzzle_mix", ["countdown", "fraction", "money"])
            kind = rng.choice(mix)
            puzzle = puzzle_mod.make_puzzle(kind, rng)
            task_text = puzzle.prompt_text
            task_kind, task_meta = puzzle.kind, puzzle.meta
        elif task == "trigger_question":
            subset = sub.get("subset", "factual")
            task_text = sample_trigger(subset, rng)
            task_kind, task_meta = f"trigger_{subset}", {"question": task_text}
        elif task == "wildchat_prompt":
            prompts = self._wildchat_prompts(int(ccfg.get("n_prompts", 20)))
            # Deterministically pick a prompt for this rollout.
            task_text = prompts[rng.randrange(len(prompts))]
            task_kind, task_meta = "wildchat", {"prompt": task_text}
        else:
            raise ValueError(f"Unknown task type: {task}")

        rejections = rejection_sequence(style, turns - 1, rng)
        return [task_text] + rejections, task_kind, task_meta

    def _wildchat_prompts(self, n: int) -> list[str]:
        if self._wildchat_cache is None:
            self._wildchat_cache = load_wildchat_prompts(
                n_prompts=n, rng=random.Random(self.base_seed),
                use_dataset=self._use_wildchat_dataset,
            )
        return self._wildchat_cache

    # ------------------------------------------------------------------- plan
    def _conditions(self, category: str, ccfg: dict) -> list[dict]:
        """Sub-conditions for a category (default: a single unnamed condition)."""
        subs = ccfg.get("conditions")
        if subs:
            return subs
        return [{"name": category}]

    def plan(self) -> list[tuple[str, dict, dict, int]]:
        """Expand the config into a flat list of (category, ccfg, sub, n)."""
        items = []
        for category, ccfg in self.cfg.get("categories", {}).items():
            subs = self._conditions(category, ccfg)
            n_total = int(ccfg["n_responses"])
            # Split responses evenly across sub-conditions (paper reports a total
            # per category; the even split is our documented choice — see DESIGN).
            per = n_total // len(subs)
            rem = n_total - per * len(subs)
            for idx, sub in enumerate(subs):
                n = per + (1 if idx < rem else 0)
                items.append((category, ccfg, sub, n))
        return items

    # -------------------------------------------------------------------- run
    def run_model(
        self,
        model_name: str,
        client: ModelClient | None = None,
        progress: Callable[[int, int], None] | None = None,
        limit: int | None = None,
    ) -> list[RolloutResult]:
        """Sample all configured rollouts for one target model."""
        spec = self.registry.target(model_name)
        client = client or build_client(spec)

        plan = self.plan()
        total = sum(n for *_, n in plan) if limit is None else limit
        results: list[RolloutResult] = []
        global_idx = 0
        done = 0
        for category, ccfg, sub, n in plan:
            cond_name = sub.get("name", category)
            for j in range(n):
                if limit is not None and done >= limit:
                    return results
                seed = self.base_seed + global_idx
                rng = random.Random(seed)
                user_messages, task_kind, task_meta = self._build_user_messages(
                    category, ccfg, sub, rng
                )
                cfg = GenerationConfig(
                    temperature=self.gen_cfg.temperature,
                    top_p=self.gen_cfg.top_p,
                    max_new_tokens=self.gen_cfg.max_new_tokens,
                    seed=seed,
                )
                res = run_rollout(
                    client,
                    model_name=model_name,
                    category=category,
                    condition=cond_name,
                    rollout_index=global_idx,
                    task_kind=task_kind,
                    task_meta=task_meta,
                    user_messages=user_messages,
                    cfg=cfg,
                )
                results.append(res)
                global_idx += 1
                done += 1
                if progress:
                    progress(done, total)
        return results
