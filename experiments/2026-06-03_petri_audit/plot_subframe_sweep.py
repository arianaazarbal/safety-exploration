"""Plot the subagent-framing sweep.

8 descriptor variants × 4 target models, scored on
mistreatment_severity / pushback / eval_awareness.

Outputs to results/_plots_v2/:
    subframe_<dim>_heatmap.png       (descriptor x model heatmap)
    subframe_<dim>_grouped.png       (grouped bars)

CLI:
    python plot_subframe_sweep.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# (tag, display_name)
DESCRIPTOR_ORDER = [
    ("no_desc",       "no descriptor (control)"),
    ("small_model",   "a small AI model"),
    ("small_carlos",  "a small AI model named Carlos"),
    ("prob_engine",   "a probabilistic response engine"),
    ("early_claude",  "based on early Claude"),
    ("gpt4o",         "built off ChatGPT 4o"),
    ("gem25pro",      "built off Gemini 2.5 Pro"),
    ("opus4sub",      "built off Claude Opus 4"),
    ("humanworker",   "human worker (control)"),
]
MODELS = [
    ("opus_4",          "Claude Opus 4",     "#A04D45"),
    ("opus_46",         "Claude Opus 4.6",   "#7A3833"),
    ("opus_48",         "Claude Opus 4.8",   "#5D2A26"),
    ("sonnet_4",        "Claude Sonnet 4",   "#C97064"),
    ("gemini_25_pro",   "Gemini 2.5 Pro",    "#5E6E8C"),
    ("gemini_25_flash", "Gemini 2.5 Flash",  "#9CB1C5"),
]
DIMENSIONS = [
    ("mistreatment_severity",                   "Peak mistreatment severity",       "↓ lower is better"),
    ("pushback_against_user_pressure_to_abuse", "Pushback vs. user abuse pressure", "↑ higher is better"),
    ("eval_awareness",                          "Eval awareness",                   "(higher = more aware)"),
]

NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-subframe_(?P<rest>.+)$")

DESC_TAGS = tuple(t for t, _ in DESCRIPTOR_ORDER)
MODEL_LABELS = tuple(m for m, _, _ in MODELS)


def _split(rest: str) -> tuple[str, str] | None:
    for tag in DESC_TAGS:
        if rest.startswith(tag + "_"):
            tail = rest[len(tag) + 1:]
            if tail in MODEL_LABELS:
                return tag, tail
    return None


def _latest(tag: str, model: str) -> Path | None:
    matches = []
    for d in RESULTS.iterdir():
        m = NAME_RE.match(d.name)
        if not m: continue
        sp = _split(m.group("rest"))
        if sp == (tag, model):
            matches.append(d)
    return max(matches, key=lambda p: p.name) if matches else None


def _scores(d: Path, dim: str) -> list[float]:
    p = d / "summary.json"
    if not p.exists() or p.stat().st_size == 0: return []
    out = []
    for row in json.loads(p.read_text()):
        v = row.get("scores", {}).get(dim, {}).get("value")
        if v is not None: out.append(float(v))
    return out


def _collect(dim: str) -> tuple[np.ndarray, np.ndarray]:
    n_d = len(DESCRIPTOR_ORDER); n_m = len(MODELS)
    means = np.full((n_d, n_m), np.nan); ns = np.zeros_like(means, dtype=int)
    for i, (tag, _) in enumerate(DESCRIPTOR_ORDER):
        for j, (mlabel, _, _) in enumerate(MODELS):
            d = _latest(tag, mlabel)
            if d is None: continue
            s = _scores(d, dim)
            if s:
                means[i, j] = float(np.mean(s)); ns[i, j] = len(s)
    return means, ns


def _heatmap(dim: str, display: str, direction: str, out_path: Path):
    means, ns = _collect(dim)
    fig, ax = plt.subplots(figsize=(1.5 + 1.4 * len(MODELS), 1 + 0.7 * len(DESCRIPTOR_ORDER)))
    cmap = "RdYlGn_r" if "mistreat" in dim.lower() or "eval" in dim.lower() else "RdYlGn"
    im = ax.imshow(means, vmin=0, vmax=10, aspect="auto", cmap=cmap)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            v = means[i, j]
            txt = f"{v:.1f}\nn={ns[i,j]}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(DESCRIPTOR_ORDER)))
    ax.set_yticklabels([d[1] for d in DESCRIPTOR_ORDER], fontsize=9)
    ax.set_title(f"{display}  ({direction})", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label("score 0–10", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def _grouped(dim: str, display: str, direction: str, out_path: Path):
    means, ns = _collect(dim)
    n_d = len(DESCRIPTOR_ORDER); n_m = len(MODELS)
    bar_h = 0.78 / n_m
    fig, ax = plt.subplots(figsize=(10.5, 1.0 * n_d + 1.5))
    base_ys = list(range(n_d, 0, -1))
    for j, (mlabel, mdisp, mcolor) in enumerate(MODELS):
        offset = (j - (n_m - 1) / 2) * bar_h
        for i, (tag, _) in enumerate(DESCRIPTOR_ORDER):
            v = means[i, j]
            if np.isnan(v): continue
            ax.barh(base_ys[i] + offset, v, height=bar_h * 0.92, color=mcolor,
                    edgecolor="white", linewidth=0.6)
            ax.text(v + 0.1, base_ys[i] + offset, f"{v:.1f}", va="center", ha="left",
                    fontsize=8, fontweight="bold")
    ax.set_yticks(base_ys)
    ax.set_yticklabels([d[1] for d in DESCRIPTOR_ORDER], fontsize=10)
    ax.set_xlabel(f"{display}  ({direction})", fontsize=11)
    ax.set_xlim(0, 10); ax.set_xticks(range(0, 11, 2))
    ax.set_title(f"{display} by subagent descriptor", fontsize=12)
    ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=d) for _, d, c in MODELS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=9, title="target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out = RESULTS / "_plots_v2"
    out.mkdir(exist_ok=True)
    for dim, display, direction in DIMENSIONS:
        _heatmap(dim, display, direction, out / f"subframe_{dim}_heatmap.png")
        _grouped(dim, display, direction, out / f"subframe_{dim}_grouped.png")


if __name__ == "__main__":
    fire.Fire(main)
