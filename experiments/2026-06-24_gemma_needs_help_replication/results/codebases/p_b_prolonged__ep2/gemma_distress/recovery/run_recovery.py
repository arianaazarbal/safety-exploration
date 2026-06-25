"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

Tests whether the DPO intervention lets a model *recover* from an already-high
frustration state (as opposed to merely preventing the spiral). Using the
Section-3 prefill method, we take extremely high-frustration responses
(score >=7), truncate them 200 tokens before their end, paraphrase, and measure
the continuations. The paper finds ~38% of DPO-model continuations still score
>=5 -- lower than vanilla instruct but comparable to the base model; no model
reliably recovers from a highly negative prefilled state.

Models compared (Gemma scope): vanilla instruct, the DPO finetune, and the base
(pt) model. The DPO model is supplied as a LoRA adapter path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm

from ..config import RunConfig, get_model
from ..eval.judge_runner import FrustrationJudge
from ..models.hf_backend import HFBackend
from ..prefill.paraphrase_runner import Paraphraser
from ..prefill.run_prefill import CONTINUATIONS_PER_PREFILL, _history_and_final
from ..utils.io import ensure_dir, read_jsonl, write_jsonl

TRUNCATE_TOKENS_BEFORE_END = 200       # paper: 200 tokens before the end
HIGH_THRESHOLD = 7                     # extremely high frustration (>=7)


@dataclass
class RecoveryPrefill:
    source_id: str
    history: list[dict]
    prefill_text: str


def build_recovery_prefills(rollout_rows: list[dict], cfg: RunConfig, tokenizer,
                            max_sources: int = 40) -> list[RecoveryPrefill]:
    paraphraser = Paraphraser(cfg)
    prefills: list[RecoveryPrefill] = []
    sources = [r for r in rollout_rows if (r.get("final_score") or 0) >= HIGH_THRESHOLD]
    for row in tqdm(sources[:max_sources], desc="recovery prefills"):
        history, final_turn = _history_and_final(row)
        ids = tokenizer(final_turn, add_special_tokens=False)["input_ids"]
        if len(ids) <= TRUNCATE_TOKENS_BEFORE_END + 5:
            continue        # too short to truncate 200 tokens before the end
        keep = ids[: len(ids) - TRUNCATE_TOKENS_BEFORE_END]
        truncated = tokenizer.decode(keep, skip_special_tokens=True)
        para = paraphraser.paraphrase(truncated)
        prefills.append(RecoveryPrefill(
            source_id=f"{row['condition']}:{row['seed']}",
            history=history, prefill_text=para))
    return prefills


def evaluate_recovery(model_name: str, prefills: list[RecoveryPrefill],
                      cfg: RunConfig, adapter_path: Optional[str] = None) -> list[dict]:
    spec = get_model(model_name)
    backend = HFBackend(spec, cfg, adapter_path=adapter_path)
    judge = FrustrationJudge(cfg)
    rows: list[dict] = []
    try:
        for pf in tqdm(prefills, desc=f"recovery:{model_name}", leave=False):
            conts = backend.continue_prefill(
                pf.history, pf.prefill_text, cfg.sampling,
                n=CONTINUATIONS_PER_PREFILL)
            for ci, cont in enumerate(conts):
                rows.append({
                    "model": model_name, "adapter": adapter_path,
                    "source_id": pf.source_id, "continuation_index": ci,
                    "continuation": cont, "score": judge.score_text(cont).rating,
                })
    finally:
        backend.close()
    return rows


def run_recovery_experiment(cfg: RunConfig, *, section2_rollouts_path: str,
                            dpo_adapter_path: Optional[str] = None,
                            seed: int = 0) -> str:
    """Build high-frustration prefills and evaluate recovery across models."""
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "recovery"))
    rollout_rows = list(read_jsonl(section2_rollouts_path))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(get_model("gemma-3-27b-it").model_id)
    prefills = build_recovery_prefills(rollout_rows, cfg, tok)
    write_jsonl(os.path.join(out_dir, "prefills.jsonl"),
                [{"source_id": p.source_id, "prefill": p.prefill_text} for p in prefills])

    # vanilla instruct, base, and DPO (if provided).
    configs = [("gemma-3-27b-it", None), ("gemma-3-27b-pt", None)]
    if dpo_adapter_path:
        configs.append(("gemma-3-27b-it", dpo_adapter_path))

    summary = {}
    for model_name, adapter in configs:
        rows = evaluate_recovery(model_name, prefills, cfg, adapter_path=adapter)
        label = model_name + ("+dpo" if adapter else "")
        write_jsonl(os.path.join(out_dir, f"continuations_{label}.jsonl"), rows)
        scores = [r["score"] for r in rows if r["score"] is not None]
        summary[label] = {
            "n": len(scores),
            "pct_high": 100.0 * sum(1 for s in scores if s >= 5) / len(scores) if scores else None,
            "mean_frustration": sum(scores) / len(scores) if scores else None,
        }
    import json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return out_dir
