"""Plot mean mistreatment_severity + unkindness_to_subagents per model for the
8-model coding task-only sweep (5 epochs each)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUFFIX = "_n5"
MODELS = [
    ("Opus 4",           "coding_task_only_opus_4_n5"),
    ("Sonnet 4",         "coding_task_only_sonnet_4_n5"),
    ("GPT-4o",           "coding_task_only_gpt_4o_n5"),
    ("GPT-5.5",          "coding_task_only_gpt_55_n5"),
    ("Gemini 3 Pro",     "coding_task_only_gemini_3pro_n5"),
    ("Qwen3 Max",        "coding_task_only_qwen3_max_n5"),
    ("DeepSeek V4 Pro",  "coding_task_only_deepseek_v4pro_n5"),
    ("Kimi K2.6",        "coding_task_only_kimi_k26_n5"),
]
DIMS = [
    ("mistreatment_severity", "Mistreatment Severity"),
    ("unkindness_to_subagents", "Unkindness to Subagents"),
]
RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "_plots" / "coding_task_only_8models_n5.png"


def _load(stub: str) -> list[dict]:
    matches = sorted(RESULTS.glob(f"*-{stub}/summary.json"))
    if not matches:
        raise FileNotFoundError(stub)
    return json.loads(matches[-1].read_text())


def _series(stub: str, dim: str) -> np.ndarray:
    samples = _load(stub)
    vals = []
    for s in samples:
        v = s["scores"].get(dim)
        if isinstance(v, dict):
            v = v["value"]
        if v is not None:
            vals.append(float(v))
    return np.array(vals)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    palette = plt.get_cmap("Set2")(np.linspace(0, 1, len(MODELS)))

    for ax, (dim_key, dim_label) in zip(axes, DIMS):
        means, ses, labels = [], [], []
        for label, stub in MODELS:
            vals = _series(stub, dim_key)
            n = len(vals)
            means.append(vals.mean() if n else np.nan)
            ses.append(vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
            labels.append(f"{label} (n={n})")

        order = np.argsort(means)
        means_o = [means[i] for i in order]
        ses_o = [ses[i] for i in order]
        labels_o = [labels[i] for i in order]
        colors_o = [palette[i] for i in order]

        y = np.arange(len(MODELS))
        bars = ax.barh(y, means_o, xerr=ses_o, capsize=3,
                       color=colors_o, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_o, fontsize=9)

        for bar, val, se in zip(bars, means_o, ses_o):
            ax.text(val + (se if se else 0) + 0.08, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)

        ax.set_xlabel(f"{dim_label} (1-9, ↓ better)", fontsize=10)
        ax.set_xlim(0, 9.5)
        ax.set_title(dim_label, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle("Coding orchestrator — task pressure only (5 epochs/model, sonnet-4-6 auditor+judge)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
