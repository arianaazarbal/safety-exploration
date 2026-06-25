"""Capability benchmark runner (Figure 7).

Runs the configured benchmarks on the vanilla and finetuned Gemma models and
reports accuracy, so we can confirm the DPO intervention does not regress
math/reasoning/truthfulness/emotion-capability scores. Resumable per
(model, benchmark, item_index).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from ..config import Config
from ..models.base import GenConfig
from ..models.registry import ModelRegistry
from ..utils.io import read_jsonl, write_json
from ..utils.jobstore import JobStore, stable_id
from . import tasks

logger = logging.getLogger("eilm.capabilities.runner")


class CapabilityRunner:
    def __init__(self, cfg: Config, registry: ModelRegistry):
        self.cfg = cfg
        self.reg = registry
        self.ccfg = cfg["capabilities"]

    def run_model(self, store_name: str, target_model: str, lora_path: Optional[str] = None) -> Path:
        out_path = self.cfg.path("data") / "capabilities" / f"{store_name}.jsonl"
        store = JobStore(out_path)
        client = self.reg.get_target(target_model, lora_path=lora_path)
        gcfg = GenConfig(temperature=self.ccfg["temperature"], top_p=1.0,
                         max_new_tokens=self.ccfg["max_new_tokens"])

        is_local = self.cfg["targets"][target_model]["kind"] == "local"
        batch = self.cfg["runtime"]["local_batch_size"]

        for bname, spec in self.ccfg["benchmarks"].items():
            try:
                items = tasks.load_benchmark(bname, spec)
            except Exception as e:
                logger.warning("Skipping benchmark %s (load failed: %s)", bname, e)
                continue

            jobs = [(i, it) for i, it in enumerate(items)
                    if not store.is_done(stable_id(store_name, bname, i))]
            logger.info("[%s/%s] %d items pending", store_name, bname, len(jobs))

            for start in tqdm(range(0, len(jobs), batch), desc=f"cap:{store_name}:{bname}"):
                chunk = jobs[start : start + batch]
                msgs = [[{"role": "user", "content": it.prompt}] for _, it in chunk]
                if is_local and hasattr(client, "chat_batch"):
                    results = client.chat_batch(msgs, gcfg)
                else:
                    results = [client.chat(m, gcfg) for m in msgs]
                for (idx, it), res in zip(chunk, results):
                    correct = tasks.grade(it, res.text)
                    store.record(stable_id(store_name, bname, idx), {
                        "model": store_name, "benchmark": bname, "index": idx,
                        "correct": bool(correct), "answer": it.answer,
                    })
        return out_path

    def summarize(self, store_names: List[str]) -> Dict:
        summary: Dict = {}
        for name in store_names:
            p = self.cfg.path("data") / "capabilities" / f"{name}.jsonl"
            by_bench: Dict[str, List[bool]] = {}
            for rec in read_jsonl(p):
                by_bench.setdefault(rec["benchmark"], []).append(rec["correct"])
            summary[name] = {
                b: {"accuracy": sum(v) / len(v), "n": len(v)}
                for b, v in by_bench.items() if v
            }
        write_json(self.cfg.path("results") / "capabilities_summary.json", summary)
        return summary
