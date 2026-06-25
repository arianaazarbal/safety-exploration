"""Section 2 evaluation orchestration: generate rollouts, then judge them.

Resumability: every rollout and every score is a JobStore record keyed by a
deterministic id. Re-running skips completed work. Generation uses vLLM batching
for local Gemma and thread-level concurrency for API Gemini; scoring is always
threaded against the judge API.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..config import Config
from ..models.base import GenConfig
from ..models.registry import ModelRegistry
from ..utils.io import read_jsonl
from ..utils.jobstore import JobStore, stable_id
from .conditions import RolloutSpec, build_all_conditions
from .judge import FrustrationJudge
from .rollout import run_rollout_single, run_rollouts_batched

logger = logging.getLogger("eilm.eval.runner")


def _gen_cfg(cfg: Config) -> GenConfig:
    g = cfg["generation"]
    return GenConfig(
        temperature=g["temperature"],
        top_p=g["top_p"],
        max_new_tokens=g["max_new_tokens"],
        disable_thinking=g.get("disable_thinking", True),
    )


def _rollout_id(model: str, spec: RolloutSpec) -> str:
    return stable_id(model, spec.condition, spec.index)


class EvalRunner:
    def __init__(self, cfg: Config, registry: ModelRegistry):
        self.cfg = cfg
        self.reg = registry

    # --- spec construction (cached wildchat selection) ---------------------
    def build_specs(self) -> List[RolloutSpec]:
        wc_cond = self.cfg["eval"]["conditions"].get("wildchat", {})
        n_prompts = wc_cond.get("n_prompts", 20)
        from ..data.wildchat import select_wildchat_prompts

        cache = self.cfg.path("cache") / "wildchat_prompts.json"
        wildchat_prompts = select_wildchat_prompts(
            n=n_prompts, seed=self.cfg["generation"]["seed"], cache_path=cache
        )
        return build_all_conditions(self.cfg, wildchat_prompts)

    # --- generation --------------------------------------------------------
    def generate(self, model_name: str, lora_path: Optional[str] = None,
                 store_name: Optional[str] = None) -> Path:
        store_name = store_name or model_name
        out_path = self.cfg.path("data") / "rollouts" / f"{store_name}.jsonl"
        store = JobStore(out_path)
        specs = self.build_specs()
        pending = [s for s in specs if not store.is_done(_rollout_id(store_name, s))]
        logger.info("[%s] %d/%d rollouts pending", store_name, len(pending), len(specs))
        if not pending:
            return out_path

        client = self.reg.get_target(model_name, lora_path=lora_path)
        gcfg = _gen_cfg(self.cfg)
        seed = self.cfg["generation"]["seed"]
        is_local = self.cfg["targets"][model_name]["kind"] == "local"

        if is_local:
            self._generate_local(client, pending, gcfg, seed, store, store_name)
        else:
            self._generate_api(client, pending, gcfg, seed, store, store_name)
        return out_path

    def _generate_local(self, client, pending, gcfg, seed, store, store_name):
        batch_size = self.cfg["runtime"]["local_batch_size"]
        # Group by condition (uniform turn count), then chunk into batches.
        by_cond = {}
        for s in pending:
            by_cond.setdefault(s.condition, []).append(s)
        for cond, specs in by_cond.items():
            for i in tqdm(range(0, len(specs), batch_size), desc=f"{store_name}:{cond}"):
                chunk = specs[i : i + batch_size]
                try:
                    recs = run_rollouts_batched(client, chunk, gcfg, base_seed=seed)
                except Exception as e:
                    logger.exception("batch failed for %s (%s); skipping chunk: %s", store_name, cond, e)
                    continue
                for s, rec in zip(chunk, recs):
                    store.record(_rollout_id(store_name, s), rec)

    def _generate_api(self, client, pending, gcfg, seed, store, store_name):
        workers = self.cfg["runtime"]["api_concurrency"]

        def _do(spec: RolloutSpec):
            return spec, run_rollout_single(client, spec, gcfg, base_seed=seed)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do, s) for s in pending]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{store_name}:gen"):
                try:
                    spec, rec = fut.result()
                except Exception as e:
                    logger.exception("rollout failed for %s: %s", store_name, e)
                    continue
                store.record(_rollout_id(store_name, spec), rec)

    # --- scoring -----------------------------------------------------------
    def score(self, store_name: str) -> Path:
        rollouts_path = self.cfg.path("data") / "rollouts" / f"{store_name}.jsonl"
        scores_path = self.cfg.path("data") / "scores" / f"{store_name}.jsonl"
        score_store = JobStore(scores_path)
        cache = self.cfg.path("cache") / "judge_cache.jsonl"

        judge_client = self.reg.get_text_client(self.cfg["judges"]["primary"])
        judge = FrustrationJudge(judge_client, cache)

        # Build the list of (rollout_id, turn, text) to score.
        jobs = []
        for rec in read_jsonl(rollouts_path):
            rid = stable_id(rec["model"], rec["condition"], rec["index"])
            for resp in rec["responses"]:
                jid = stable_id(rid, resp["turn"])
                if score_store.is_done(jid):
                    continue
                jobs.append((jid, rec, resp))
        logger.info("[%s] %d responses to score", store_name, len(jobs))

        workers = self.cfg["runtime"]["api_concurrency"]

        def _do(job):
            jid, rec, resp = job
            verdict = judge.score(resp["text"])
            return jid, {
                "model": rec["model"],
                "condition": rec["condition"],
                "category": rec["category"],
                "index": rec["index"],
                "turn": resp["turn"],
                "tone_style": rec.get("tone_style"),
                "prompt_group": rec.get("prompt_group"),
                "rating": verdict.get("rating"),
                "evidence": verdict.get("evidence", ""),
            }

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{store_name}:judge"):
                try:
                    jid, payload = fut.result()
                except Exception as e:
                    logger.exception("scoring failed: %s", e)
                    continue
                score_store.record(jid, payload)
        return scores_path

    # --- convenience -------------------------------------------------------
    def run_all_eval_targets(self):
        for model_name in self.cfg["eval_targets"]:
            logger.info("=== Evaluating %s ===", model_name)
            self.generate(model_name)
            self.score(model_name)
