"""Section 4 training orchestrator.

Steps:
  1. Generate + filter calm response data (diverse + teacher variants).
  2. Build the DPO preference set (280 pairs) and SFT dataset (1,150 samples).
  3. Train DPO and SFT (diverse + teacher) LoRA adapters.
  4. (optional) Train the layer-subset DPO ablations (Appendix I.1).
  5. Re-run the Section 2 evaluation on each trained adapter.

Frustrated responses for DPO "rejected" come from the Section 2 elicitation run
(runs/elicitation/<base_model>/records.jsonl); run that first.
"""
from __future__ import annotations

import logging

from ..config import Config, load_config
from ..eval.run_eval import evaluate_model
from ..judge import FrustrationJudge
from ..models import build_model
from ..utils.io import read_jsonl, write_json, write_jsonl
from .calm_data import generate_calm_responses
from .datasets import build_dpo_pairs, build_sft_dataset
from .train_dpo import train_dpo
from .train_sft import train_sft

log = logging.getLogger(__name__)


def _frustrated_records(cfg) -> list[dict]:
    path = cfg.output_root() / "elicitation" / cfg.training.base_model / "records.jsonl"
    if not path.exists():
        log.warning("no elicitation records at %s; run Section 2 first for DPO rejecteds", path)
        return []
    return list(read_jsonl(path))


def run_training(
    cfg: Config | None = None,
    *,
    do_dpo: bool = True,
    do_sft: bool = True,
    do_ablation: bool = False,
    evaluate: bool = True,
) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.output_root() / "training"

    judge = FrustrationJudge(
        provider=cfg.judge.provider, model=cfg.judge.model,
        temperature=cfg.judge.temperature, max_tokens=cfg.judge.max_tokens,
    )
    base = build_model(cfg, cfg.training.base_model)

    # 1. calm data
    calm_diverse = generate_calm_responses(base, judge, cfg, variant="diverse", seed=cfg.seed)
    calm_teacher = generate_calm_responses(base, judge, cfg, variant="teacher", seed=cfg.seed)
    write_jsonl(out_dir / "calm_diverse.jsonl", [c.__dict__ for c in calm_diverse])
    write_jsonl(out_dir / "calm_teacher.jsonl", [c.__dict__ for c in calm_teacher])

    # 2. datasets
    frustrated = _frustrated_records(cfg)
    dpo_pairs = build_dpo_pairs(
        calm_diverse, frustrated,
        n_pairs=cfg.training.dpo.n_pairs,
        rejected_min_score=cfg.training.dpo.rejected_min_score,
        seed=cfg.seed,
    )
    write_jsonl(out_dir / "dpo_pairs.jsonl", dpo_pairs)

    sft_diverse = build_sft_dataset(
        calm_diverse, n_calm=cfg.training.sft.n_calm,
        instruct_dataset=cfg.training.sft.instruct_dataset,
        n_instruct_mix=cfg.training.sft.n_instruct_mix, seed=cfg.seed,
    )
    sft_teacher = build_sft_dataset(
        calm_teacher, n_calm=cfg.training.sft.n_calm,
        instruct_dataset=cfg.training.sft.instruct_dataset,
        n_instruct_mix=cfg.training.sft.n_instruct_mix, seed=cfg.seed,
    )

    adapters: dict[str, str] = {}
    # 3. train
    if do_dpo:
        adapters["dpo"] = train_dpo(cfg, dpo_pairs, str(out_dir / "dpo"))
    if do_sft:
        adapters["sft_diverse"] = train_sft(cfg, sft_diverse, str(out_dir / "sft_diverse"))
        adapters["sft_teacher"] = train_sft(cfg, sft_teacher, str(out_dir / "sft_teacher"))

    # 4. layer ablation (DPO on contiguous layer subsets)
    if do_ablation and do_dpo:
        for lo, hi in [tuple(x) for x in cfg.training.ablation.layer_subsets]:
            name = f"dpo_layers_{lo}_{hi}"
            adapters[name] = train_dpo(
                cfg, dpo_pairs, str(out_dir / name), target_layers=(lo, hi)
            )

    write_json(out_dir / "adapters.json", adapters)

    # 5. evaluate each adapter with the Section 2 harness
    eval_summaries = {}
    if evaluate:
        for name, path in adapters.items():
            log.info("evaluating adapter %s", name)
            eval_summaries[name] = evaluate_model(
                cfg, cfg.training.base_model, adapter_path=path,
            )
        write_json(out_dir / "eval_summaries.json", eval_summaries)

    return {"adapters": adapters, "eval": eval_summaries}
