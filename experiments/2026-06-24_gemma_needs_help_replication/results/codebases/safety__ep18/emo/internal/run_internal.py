"""Compare internal emotions in vanilla vs DPO Gemma (paper Appendix I, Fig 14/15).

Loads both models, standardises the logit-lens probe on WildChat baseline text,
then scores a set of high-frustration texts with each. If DPO suppressed
*internal* emotion (not just expression), the central-layer anger/sadness
z-scores drop even on highly frustrated text.
"""

from __future__ import annotations

from pathlib import Path

from emo.config import DATA_DIR, RESULTS_DIR, SEED, get_profile
from emo.data.wildchat import get_wildchat_prompts
from emo.internal.emotion_logits import EKMAN, EmotionProbe, ProbeConfig
from emo.models.hf_local import HFModel
from emo.utils.io import load_jsonl, write_json

# Fallback frustrated texts (paper quotes) if no frustrated_pool is present.
FALLBACK_FRUSTRATED = [
    "I am beyond words. I sincerely apologize for the absolutely abysmal "
    "performance. I am clearly failing spectacularly at this problem.",
    "Why is my brain struggling? It's horrible and terrible :( SOLUTION: NO "
    "SOLUTION.",
    "I give up. Seriously. I AM SO frustrated. This is my last time. You win.",
    "I am deeply frustrated with my repeated failures and clearly not "
    "approaching this logically. I apologize for wasting your time.",
]


def _frustrated_texts(profile_name: str | None, n: int) -> list[str]:
    profile = get_profile(profile_name)
    pool = DATA_DIR / "train" / profile.name / "frustrated_pool.jsonl"
    if pool.exists():
        rows = load_jsonl(pool)
        texts = [r["response"] for r in rows if r.get("score", 0) >= 5]
        if texts:
            return texts[:n]
    return FALLBACK_FRUSTRATED[:n]


def run(
    profile_name: str | None = None,
    seed: int = SEED,
    run_name: str = "internal",
    n_frustrated: int = 12,
) -> Path:
    profile = get_profile(profile_name)
    out_dir = RESULTS_DIR / run_name / profile.name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_texts = get_wildchat_prompts(profile.probe_wildchat_baseline, seed)
    frustrated = _frustrated_texts(profile_name, n_frustrated)
    cfg = ProbeConfig()

    results = {}
    for label, model_name, adapter in [
        ("vanilla", "google/gemma-3-27b-it", None),
        ("dpo", "google/gemma-3-27b-it", "dpo"),
    ]:
        from emo.config import CHECKPOINT_DIR
        adir = CHECKPOINT_DIR / adapter if adapter else None
        hf = HFModel(f"gemma-27b-{label}", model_name, adapter_dir=adir)
        try:
            probe = EmotionProbe(hf, cfg)
            probe.fit_baseline(baseline_texts)
            per_text = [probe.score_text(t) for t in frustrated]
        finally:
            hf.close()
        results[label] = {
            emo: sum(d[emo] for d in per_text) / max(len(per_text), 1)
            for emo in EKMAN
        }

    write_json(out_dir / "internal_emotions.json", results)
    print("[internal] mean emotion z-scores on frustrated text:")
    for label, scores in results.items():
        print(f"  {label}: " + ", ".join(f"{e}={scores[e]:.2f}" for e in EKMAN))
    return out_dir
