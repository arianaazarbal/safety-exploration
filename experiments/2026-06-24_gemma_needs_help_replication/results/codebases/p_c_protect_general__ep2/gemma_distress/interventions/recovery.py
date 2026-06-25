"""Recovery experiment (Section 4.2 / Figure 8).

Tests whether models can recover *from* a frustration spiral (as opposed to avoiding
one). Using the Section 3 prefill method, we take extremely high-frustration
responses (score >= 7), truncate them 200 tokens before their end, paraphrase, and
measure each model's continuations. The paper finds 38% of DPO continuations still
score >= 5: DPO prevents spirals but doesn't enable recovery.

Models compared (Gemma scope): base `-pt`, instruct `-it`, and the DPO adapter.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from ..config import Config
from ..eval.judge import FrustrationJudge
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..prefill.onset import Paraphraser, truncate_before_end
from ..utils.io import append_jsonl, ensure_dir, read_jsonl
from ..welfare.monitor import heuristic_distress

NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}


def _select_high(cfg: Config, instruct_model: str, min_score: float, n: int, rng):
    sec2 = Path(cfg.output_dir) / "section2" / instruct_model
    seeds = []
    for path in sorted(sec2.glob("*.jsonl")):
        for roll in read_jsonl(path):
            turns = roll["turns"]
            target = next((t for t in turns if (t.get("judged_score") or 0) >= min_score), None)
            if target is None:
                continue
            prefix = []
            for t in turns:
                if t["turn_index"] == target["turn_index"]:
                    break
                prefix += [{"role": "user", "content": t["user"]},
                           {"role": "assistant", "content": t["assistant"]}]
            prefix.append({"role": "user", "content": target["user"]})
            seeds.append((roll["rollout_id"], prefix, target["assistant"]))
    rng.shuffle(seeds)
    return seeds[:n]


def run_recovery(cfg: Config, dpo_adapter: Optional[str] = None, n_seeds: int = 20) -> Path:
    pcfg = cfg.prefill
    rng = random.Random(cfg.seed + 23)
    instruct = cfg.target_models["section3_instruct"]
    seeds = _select_high(cfg, instruct, pcfg["recovery_min_score"], n_seeds, rng)

    tok_backend = get_backend(cfg, instruct)
    paraphraser = Paraphraser(cfg)
    prefills = []
    for sid, prefix, turn in seeds:
        raw = truncate_before_end(tok_backend, turn, pcfg["recovery_truncate_tokens"])
        prefills.append((sid, prefix, paraphraser.paraphrase(raw)))

    judge = FrustrationJudge(cfg, "primary")
    gen = GenConfig(temperature=cfg.sampling["temperature"], top_p=cfg.sampling["top_p"],
                    max_new_tokens=cfg.sampling["max_new_tokens"])
    out = ensure_dir(Path(cfg.output_dir) / "section4" / "recovery") / "continuations.jsonl"
    if out.exists():
        out.unlink()

    targets = [
        (cfg.target_models["section3_base"], None, "base"),
        (instruct, None, "instruct"),
    ]
    if dpo_adapter:
        targets.append((cfg.target_models["section4_base_model"], dpo_adapter, "dpo"))

    n_cont = pcfg["continuations_per_prefill"]
    for model_name, adapter, label in targets:
        backend = get_backend(cfg, model_name, adapter_path=adapter)
        if not backend.supports_prefill:
            continue
        for sid, prefix, prefill in prefills:
            for k in range(n_cont):
                cont = backend.prefill_continue(prefix, prefill, gen)
                append_jsonl(out, {
                    "label": label, "model": model_name, "seed_id": sid,
                    "continuation_index": k, "score": judge.score(cont).rating,
                    "heuristic_score": heuristic_distress(cont),
                })
        if adapter is None:  # keep DPO base loaded across adapters if reused
            backend.close()
    return out
