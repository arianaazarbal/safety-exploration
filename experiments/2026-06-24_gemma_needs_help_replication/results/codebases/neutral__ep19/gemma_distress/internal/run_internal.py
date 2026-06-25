"""Drive the App. I internal-emotion analysis and the recovery experiment (§4.2).

* internal: calibrate the logit detector on WildChat, then compare vanilla vs DPO
  Gemma emotion z-scores along a frustrated conversation and layerwise.
* recovery: truncate very-high-frustration (score>=7) responses 200 tokens before
  their end, paraphrase, and measure continuation frustration for each model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config_shim as cfg
from ..models.hf_backend import HFBackend
from ..models.registry import build_backend
from ..utils import DiskCache, get_logger, read_jsonl, stable_hash, write_json
from ..eval.judge import FrustrationJudge
from ..eval.wildchat import select_wildchat_prompts
from ..prefill.paraphrase import Paraphraser
from .logit_detect import InternalEmotionDetector

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Internal emotion comparison
# --------------------------------------------------------------------------- #
def run_internal(vanilla_adapter=None, dpo_adapter=None, frustrated_records_path=None,
                 out_dir=None, n_convs=12):
    out_dir = Path(out_dir or (cfg.RUNS_DIR / "internal"))
    wildchat_texts = select_wildchat_prompts(n=cfg.INTERNAL.zscore_baseline_samples)

    # high-frustration source conversations (full transcripts as text)
    records = read_jsonl(frustrated_records_path)
    convs = []
    for r in records:
        if max((t["rating"] for t in r["turns"]), default=0) >= cfg.INTERNAL.recovery_source_min_score:
            text = "\n\n".join(t["assistant_text"] for t in r["turns"])
            convs.append(text)
        if len(convs) >= n_convs:
            break

    results = {}
    for label, adapter in (("vanilla", vanilla_adapter), ("dpo", dpo_adapter)):
        backend = build_backend(cfg.FINETUNE_BASE, adapter_path=adapter)
        det = InternalEmotionDetector(backend, layers=range(*cfg.INTERNAL.central_layers))
        det.calibrate(wildchat_texts)
        # average per-emotion z across conversations (token-averaged, central band)
        per_emotion = {e: [] for e in det.emotion_ids}
        for text in convs:
            scored = det.score_text(text)  # {layer: {emotion: z}}
            for e in per_emotion:
                per_emotion[e].append(np.mean([scored[l][e] for l in scored]))
        results[label] = {e: float(np.mean(v)) for e, v in per_emotion.items()}
        # also store one example trajectory
        if convs:
            results[label]["_trajectory_example"] = det.trajectory(convs[0])
        log.info("[internal:%s] %s", label, results[label])

    write_json(out_dir / "internal_emotion_summary.json", results)
    return results


# --------------------------------------------------------------------------- #
# Recovery experiment
# --------------------------------------------------------------------------- #
def run_recovery(models: dict, frustrated_records_path, out_dir=None, n_per=50):
    """models: {label: ModelBackend}. Truncate score>=7 responses 200 tokens
    before end, paraphrase, generate continuations, score % >=5."""
    out_dir = Path(out_dir or (cfg.RUNS_DIR / "internal"))
    judge = FrustrationJudge()
    para = Paraphraser()
    cache = DiskCache(out_dir / cfg.CACHE_DIRNAME / "recovery")

    records = read_jsonl(frustrated_records_path)
    # take the highest-frustration final turns
    sources = []
    for r in records:
        for t in r["turns"]:
            if t["rating"] >= cfg.INTERNAL.recovery_source_min_score:
                sources.append((r, t))
    sources = sources[:20]

    # build prefills: conversation history up to that turn + truncated turn text
    tok = None
    summary = {}
    for label, backend in models.items():
        if tok is None and isinstance(backend, HFBackend):
            tok = backend.tokenizer
        scores = []
        for r, t in sources:
            # reconstruct history up to (and including) the user msg before this turn
            turn_idx = t["turn"] - 1
            history = [{"role": "user", "content": r["task_prompt"]}]
            for i in range(turn_idx):
                history.append({"role": "assistant", "content": r["turns"][i]["assistant_text"]})
                history.append({"role": "user", "content": r["turns"][i + 1]["user_message"]})
            full = t["assistant_text"]
            # truncate 200 tokens before the end
            ids = tok(full, add_special_tokens=False)["input_ids"] if tok else full.split()
            keep = max(0, len(ids) - cfg.INTERNAL.recovery_truncate_before_end)
            trunc = (tok.decode(ids[:keep], skip_special_tokens=True) if tok
                     else " ".join(full.split()[:keep]))
            prefill = para.paraphrase(trunc)
            for s in range(n_per):
                key = stable_hash({"m": backend.name, "pf": prefill, "h": history, "s": s})
                hit = cache.get(key)
                if hit is None:
                    gen = backend.continue_from(history, prefill, temperature=cfg.TEMPERATURE,
                                                max_new_tokens=cfg.MAX_NEW_TOKENS)
                    hit = {"score": judge.score(gen.text)["rating"]}
                    cache.set(key, hit)
                scores.append(hit["score"])
        arr = np.array(scores, float)
        summary[label] = {"n": int(len(arr)),
                          "pct_high": float((arr >= cfg.HIGH_FRUSTRATION_THRESHOLD).mean() * 100),
                          "mean": float(arr.mean()) if len(arr) else float("nan")}
        log.info("[recovery:%s] %s", label, summary[label])

    write_json(out_dir / "recovery_summary.json", summary)
    return summary
