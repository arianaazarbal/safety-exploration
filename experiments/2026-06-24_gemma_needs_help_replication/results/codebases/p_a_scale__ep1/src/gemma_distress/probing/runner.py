"""Internal-emotion probing orchestration (Appendix I, Figures 14-15).

For a Gemma variant (vanilla or DPO), over a set of high-frustration
conversations:
  * conversation-level emotion trajectory (running average over token windows,
    aggregated across the configured layer range), and
  * layerwise emotion at three points relative to emotion onset
    (-40:-20, -20:0, final 20 tokens).

Requires the transformers backend (residual-stream access), so the provider is
built with ``require_capability='logits'``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config, load_models
from ..data.wildchat import load_wildchat_prompts
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import atomic_write_json, read_jsonl
from .emotion_logits import EMOTIONS, EmotionDetector, running_average

log = get_logger("probing.runner")


def _layer_range(run_cfg: Config) -> list[int]:
    lo, hi = run_cfg.probing.aggregate_layers
    return list(range(lo, hi + 1))


def _get_detector(model: str, run_cfg: Config, models_cfg: Config, adapter: str | None):
    provider = build_provider(model, models_cfg, run_cfg,
                              require_capability="logits", adapter=adapter)
    layers = _layer_range(run_cfg)
    det = EmotionDetector(provider, layers, per_emotion=run_cfg.probing.tokens_per_emotion)
    out = Path(run_cfg.run.output_root) / "probing"
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / f"detector_stats_{model}{'_dpo' if adapter else ''}.npz"
    if stats_path.exists():
        det.load(stats_path)
    else:
        texts = load_wildchat_prompts(n_prompts=run_cfg.probing.zscore_norm_samples,
                                      seed=run_cfg.run.seed)
        det.fit(texts)
        det.save(stats_path)
    return det


def _frustrated_conversations(run_cfg: Config, source_model: str, n: int) -> list[list[dict]]:
    eval_dir = Path(run_cfg.run.output_root) / "eval" / source_model
    scored = {r["id"]: r for r in read_jsonl(eval_dir / "scored.jsonl")}
    convs = []
    for rec in read_jsonl(eval_dir / "rollouts.jsonl"):
        sc = scored.get(rec["id"])
        if sc and sc.get("max_rating") is not None and sc["max_rating"] >= 5:
            convs.append(rec["transcript"])
        if len(convs) >= n:
            break
    return convs


def run(model: str, run_cfg: Config, models_cfg: Config | None = None,
        adapter: str | None = None, n_conversations: int = 12,
        source_model: str = "gemma-3-27b-it") -> Path:
    models_cfg = models_cfg or load_models()
    det = _get_detector(model, run_cfg, models_cfg, adapter)
    convs = _frustrated_conversations(run_cfg, source_model, n_conversations)
    window = run_cfg.probing.conversation_window_tokens
    layers = det.layers

    out = Path(run_cfg.run.output_root) / "probing"
    tag = f"{model}{'_dpo' if adapter else ''}"

    traj_summary: dict[str, list[float]] = {e: [] for e in EMOTIONS}
    layerwise = {pt: {e: {l: [] for l in layers} for e in EMOTIONS}
                 for pt in ("pre40", "pre20", "final20")}

    per_conv = []
    for ci, transcript in enumerate(tqdm(convs, desc=f"probe({tag})")):
        scored = det.score(transcript)
        # Layer-aggregated trajectory (mean over layer range).
        agg = {e: np.mean([scored["layers"][l][e] for l in layers], axis=0) for e in EMOTIONS}
        seq_len = len(next(iter(agg.values())))
        for e in EMOTIONS:
            ra = running_average(agg[e], window)
            traj_summary[e].append(float(ra.mean()))
        # Layerwise windows relative to end-of-sequence (proxy for onset/final).
        for l in layers:
            for e in EMOTIONS:
                vec = scored["layers"][l][e]
                n = len(vec)
                layerwise["pre40"][e][l].append(float(vec[max(0, n - 40):max(0, n - 20)].mean()) if n >= 40 else float("nan"))
                layerwise["pre20"][e][l].append(float(vec[max(0, n - 20):n].mean()) if n >= 20 else float("nan"))
                layerwise["final20"][e][l].append(float(vec[max(0, n - 20):n].mean()))
        per_conv.append({"conversation_index": ci, "seq_len": seq_len})

    summary = {
        "model": tag,
        "n_conversations": len(convs),
        "trajectory_mean_zscore": {e: float(np.nanmean(v)) for e, v in traj_summary.items()},
        "layerwise_mean": {
            pt: {e: {l: float(np.nanmean(layerwise[pt][e][l])) for l in layers} for e in EMOTIONS}
            for pt in layerwise
        },
    }
    atomic_write_json(out / f"summary_{tag}.json", summary)
    log.info("probing summary (%s): %s", tag, summary["trajectory_mean_zscore"])
    return out / f"summary_{tag}.json"
