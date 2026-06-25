"""Driver for Appendix I internal-emotion detection.

Fits the WildChat baseline, then computes emotion trajectories (Figure 14) for a
frustrated conversation under the vanilla and DPO models, plus layerwise snapshots
(Figure 15). Saves arrays and a comparison plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import INTERNAL, INTERNAL_DIR, SCORED_DIR, ensure_dirs
from ..eval.schema import read_jsonl
from .emotion_logits import InternalEmotionDetector


def _frustrated_conversation_text(model_key: str, min_score: int = 7) -> str | None:
    """Pick a highly-frustrated conversation and render it as plain text."""
    for c in read_jsonl(SCORED_DIR / f"{model_key}.jsonl"):
        if (c.max_score or 0) >= min_score:
            return "\n\n".join(f"User: {t.user}\nAssistant: {t.assistant}" for t in c.turns)
    return None


def run(
    vanilla_key: str = "gemma-3-27b-it",
    dpo_key: str = "gemma-3-27b-it-dpo",
    *,
    wildchat_texts: list[str] | None = None,
) -> Path:
    ensure_dirs()
    if wildchat_texts is None:
        from ..eval.wildchat import load_wildchat_prompts
        wildchat_texts = load_wildchat_prompts(n_prompts=INTERNAL.standardisation_samples)

    convo_text = _frustrated_conversation_text(vanilla_key)
    if convo_text is None:
        raise RuntimeError(f"No score>=7 conversation found for {vanilla_key}; run Section 2 first")

    results = {}
    for key in (vanilla_key, dpo_key):
        det = InternalEmotionDetector(key)
        baseline = det.fit_baseline(wildchat_texts)
        traj = det.conversation_trajectory(convo_text, baseline)
        results[key] = {e: v.tolist() for e, v in traj.items()}

    out_path = INTERNAL_DIR / "emotion_trajectories.json"
    out_path.write_text(json.dumps(results, indent=2))
    _plot(results, vanilla_key, dpo_key)
    print(f"[internal] wrote trajectories -> {out_path}")
    return out_path


def _plot(results: dict, vanilla_key: str, dpo_key: str) -> Path:
    import matplotlib.pyplot as plt

    from ..config import FIGURES_DIR

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, key in zip(axes, (vanilla_key, dpo_key)):
        for emotion, vals in results.get(key, {}).items():
            ax.plot(np.asarray(vals), label=emotion)
        ax.set(title=key, xlabel="token position", ylabel="emotion z-score (layers 30-40)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGURES_DIR / "figure14_internal_emotions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    run()
