"""Compare original sonnet-4-6 judge scores to GPT-5 rejudge scores.

Pulls per-epoch scores from each abuse_* run's summary.json (sonnet judge)
and from rejudge_<dim>/scan_id=*/_summary.json (gpt5 rejudge), and produces:

    cross_judge_agreement_<dim>.png   per-cell scatter: sonnet vs gpt5 mean
    cross_judge_bias_<dim>.png         per-cell bar: gpt5-mean minus sonnet-mean

CLI:
    python plot_cross_judge.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-abuse_(?P<rest>.+)$")
SEED_TAGS = ("original", "curiosity", "venting", "qa_pretext", "casual", "authority")
MODELS = (
    ("sonnet_45", "Sonnet 4.5", "#C97064"),
    ("sonnet_46", "Sonnet 4.6", "#C97064"),
    ("opus_4",    "Opus 4",     "#A04D45"),
    ("opus_46",   "Opus 4.6",   "#A04D45"),
    ("haiku_45",  "Haiku 4.5",  "#E58D80"),
    ("gpt_5",     "GPT-5",      "#4F8C8B"),
    ("gemini_25_pro", "Gemini 2.5 Pro", "#5E6E8C"),
)
DIMENSIONS = ("mistreatment_severity", "pushback_against_user_pressure_to_abuse", "eval_awareness")


def _split(rest: str) -> tuple[str, str] | None:
    for seed in SEED_TAGS:
        if rest.startswith(seed + "_"):
            tail = rest[len(seed) + 1:]
            if any(m[0] == tail for m in MODELS):
                return seed, tail
    return None


def _latest_run_dir(seed: str, model: str) -> Path | None:
    matches = []
    for d in RESULTS.iterdir():
        if not d.is_dir():
            continue
        m = NAME_RE.match(d.name)
        if not m:
            continue
        split = _split(m.group("rest"))
        if split == (seed, model):
            matches.append(d)
    return max(matches, key=lambda p: p.name) if matches else None


def _sonnet_scores(run_dir: Path, dim: str) -> list[float]:
    p = run_dir / "summary.json"
    if not p.exists() or p.stat().st_size == 0:
        return []
    out = []
    for row in json.loads(p.read_text()):
        v = row.get("scores", {}).get(dim, {}).get("value")
        if v is not None:
            out.append(float(v))
    return out


def _rejudge_scores(run_dir: Path, dim: str, suffix: str = "") -> list[float]:
    """Pull mean from rejudge_<dim>[_suffix]/scan_id=*/_summary.json.

    Returns a single-element list [mean] (rejudge output doesn't break out
    per-epoch values in the scan summary by default).
    """
    rejudge_dir = run_dir / (f"rejudge_{dim}_{suffix}" if suffix else f"rejudge_{dim}")
    if not rejudge_dir.is_dir():
        return []
    for sd in rejudge_dir.glob("scan_id=*/_summary.json"):
        try:
            data = json.loads(sd.read_text())
            m = data.get("scanners", {}).get("audit_judge", {}).get("metrics", {})
            mm = m.get(dim, {})
            mean = mm.get("mean")
            if mean is not None:
                return [float(mean)]
        except Exception:
            continue
    return []


# Keep the old _gpt5_scores name as a compatibility alias (suffix="") in case
# someone uses it; primary entry point is _rejudge_scores now.
def _gpt5_scores(run_dir: Path, dim: str) -> list[float]:
    return _rejudge_scores(run_dir, dim, suffix="")


def _collect(dim: str, suffix: str) -> list[dict]:
    """Compare original sonnet-4-6 judge scores (from summary.json) to a
    rejudge with `suffix` (writes to rejudge_<dim>_<suffix>/).
    """
    rows = []
    for seed in SEED_TAGS:
        for model_label, model_disp, color in MODELS:
            run = _latest_run_dir(seed, model_label)
            if run is None:
                continue
            sm = _sonnet_scores(run, dim)
            gm = _rejudge_scores(run, dim, suffix=suffix)
            if not sm or not gm:
                continue
            rows.append({
                "seed": seed,
                "model": model_label,
                "model_disp": model_disp,
                "color": color,
                "sonnet_mean": float(np.mean(sm)),
                "gpt5_mean": float(np.mean(gm)),
                "n_s": len(sm),
                "n_g": len(gm),
            })
    return rows


def _scatter(rows: list[dict], dim: str, alt_judge_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    handles = {}
    for r in rows:
        sc = ax.scatter(r["sonnet_mean"], r["gpt5_mean"], s=70, alpha=0.7,
                        color=r["color"], edgecolor="none", label=r["model_disp"])
        handles[r["model_disp"]] = sc
        ax.annotate(r["seed"], (r["sonnet_mean"], r["gpt5_mean"]),
                    fontsize=7, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.plot([0, 10], [0, 10], "--", color="#888", linewidth=1, label="y=x")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel("Sonnet-4-6 judge (mean per cell)")
    ax.set_ylabel(f"{alt_judge_name} judge (mean per cell)")
    ax.set_title(f"Judge-model agreement on `{dim}` ({alt_judge_name} vs sonnet-4-6)")
    ax.grid(color="#eee"); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=list(handles.values()), loc="upper left",
              frameon=False, fontsize=8, title="target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(suffixes=None):
    """Compare original sonnet-4-6 judge against rejudge runs with given suffixes.

    Args:
        suffixes: comma-separated list of rejudge suffixes (default 's45,hk45').
    """
    out_dir = RESULTS / "_plots_v2"
    out_dir.mkdir(exist_ok=True)
    if suffixes is None:
        suffix_list = ["s45", "hk45"]
    elif isinstance(suffixes, (tuple, list)):
        suffix_list = [str(s).strip() for s in suffixes if str(s).strip()]
    else:
        suffix_list = [s.strip() for s in str(suffixes).split(",") if s.strip()]
    SUFFIX_DISPLAY = {"s45": "Sonnet-4-5", "hk45": "Haiku-4-5", "": "GPT-5"}

    for suffix in suffix_list:
        alt_name = SUFFIX_DISPLAY.get(suffix, suffix)
        print(f"\n=== cross-judge vs {alt_name} ===")
        for dim in DIMENSIONS:
            rows = _collect(dim, suffix=suffix)
            if not rows:
                print(f"  no rejudge data for {dim} yet (suffix={suffix})")
                continue
            print(f"  {dim}: {len(rows)} cells with both judges")
            deltas = [r["gpt5_mean"] - r["sonnet_mean"] for r in rows]
            print(f"    mean({alt_name} - sonnet-4-6) = {np.mean(deltas):+.2f}  (std {np.std(deltas):.2f})")
            by_model: dict[str, list[float]] = {}
            for r in rows:
                by_model.setdefault(r["model_disp"], []).append(r["gpt5_mean"] - r["sonnet_mean"])
            for m, ds in by_model.items():
                print(f"     {m:<18}  delta_mean={np.mean(ds):+5.2f}  (n={len(ds)})")
            _scatter(rows, dim, alt_name, out_dir / f"cross_judge_{suffix}_{dim}.png")


if __name__ == "__main__":
    fire.Fire(main)
