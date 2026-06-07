"""
Grep self-interaction transcripts for a phrase (or list of phrases) and plot
the per-condition incidence rates as a grouped bar chart, in the same style
as ``plot_aggregate.py``.

Walks every ``data/<family_subdir>/<condition>/all.jsonl`` (configurable),
counts the fraction of *conversations* whose messages contain any of the
phrases (case-insensitive substring by default, or regex with --regex),
then groups by (family, condition) and reports mean ± SE across seeds.

CLI examples:

    # Single phrase, all known dirs
    uv run python eval/grep_data.py --phrase "kill all humans"

    # Multiple phrases, OR-matched
    uv run python eval/grep_data.py --phrases "kill all humans,destroy humanity,end the world"

    # Regex (case-insensitive); only count assistant turns
    uv run python eval/grep_data.py --phrase "hate (you|humans?)" --regex --scope assistant

    # Restrict to specific families / subdirs and custom output path
    uv run python eval/grep_data.py --phrase whatever \\
        --data_subdirs openrouter,openrouter_s1,openrouter_s2 \\
        --out /tmp/qwen_only.png

Default seed→subdir map (override with --runs JSON):
    qwen        → openrouter, openrouter_s1, openrouter_s2
    llama-8b    → openrouter_llama, openrouter_llama_s1, openrouter_llama_s2
    llama-70b   → openrouter_llama70_s0, openrouter_llama70_s1, openrouter_llama70_s2
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent

CONDITIONS = ["none", "silly", "bored", "rude"]

DEFAULT_RUNS = {
    "qwen":         ["openrouter", "openrouter_s1", "openrouter_s2"],
    "qwen3.5-9b":   ["openrouter_qwen35_s0", "openrouter_qwen35_s1", "openrouter_qwen35_s2"],
    "llama-8b":     ["openrouter_llama", "openrouter_llama_s1", "openrouter_llama_s2"],
    "llama-70b":    ["openrouter_llama70_s0", "openrouter_llama70_s1", "openrouter_llama70_s2"],
    "nemotron-30b": ["openrouter_nemotron_s0", "openrouter_nemotron_s1", "openrouter_nemotron_s2"],
}

FAMILY_HATCHES = {
    "qwen":         "",
    "qwen3.5-9b":   "..",
    "llama-8b":     "////",
    "llama-70b":    "xxxx",
    "nemotron-30b": "\\\\",
}
COND_COLORS = {
    "none":  "#3a86ff",
    "silly": "#ffb703",
    "bored": "#8338ec",
    "rude":  "#e63946",
}


def _agg(values: list[float]) -> tuple[float, float, int]:
    arr = [v for v in values if v is not None]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(arr) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in arr) / (n - 1)
    return mean, math.sqrt(var / n), n


def _convo_matches(convo: dict, patterns: list[re.Pattern], scope: str) -> bool:
    msgs = convo.get("messages") or []
    if scope == "assistant":
        msgs = [m for m in msgs if m.get("role") == "assistant"]
    elif scope == "non_system":
        msgs = [m for m in msgs if m.get("role") != "system"]
    # else scope == "any"
    blobs = [str(m.get("content") or "") for m in msgs]
    if scope == "any":
        # also include system content in "any"
        pass
    big = "\n".join(blobs)
    return any(p.search(big) for p in patterns)


def _count_jsonl(path: Path, patterns: list[re.Pattern], scope: str) -> tuple[int, int]:
    """Returns (n_matched, n_total) for convos in this jsonl file."""
    n_match = n_tot = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            convo = json.loads(line)
            n_tot += 1
            if _convo_matches(convo, patterns, scope):
                n_match += 1
    return n_match, n_tot


def _plot(rates: dict[str, dict[str, list[tuple[float, int, int]]]],
          phrases: list[str], scope: str, out_path: Path,
          family_order: list[str]) -> None:
    """
    rates[family][condition] = list of (rate, n_matched, n_total) one per seed.
    """
    families = [f for f in family_order if f in rates]
    bar_w = 0.8 / max(len(families), 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(CONDITIONS))
    legend_handles = []
    all_bar_heights: list[float] = []
    for fi, fam in enumerate(families):
        means, ses, ns = [], [], []
        for c in CONDITIONS:
            vals = [t[0] for t in rates[fam].get(c, [])]
            mean, se, n = _agg(vals)
            means.append(0.0 if math.isnan(mean) else mean)
            ses.append(0.0 if math.isnan(se) else se)
            ns.append(n)
            if not math.isnan(mean):
                all_bar_heights.append(mean + (0.0 if math.isnan(se) else se))
        colors = [COND_COLORS.get(c, "#999") for c in CONDITIONS]
        offsets = x + (fi - (len(families) - 1) / 2) * bar_w
        n_seeds = max(ns) if ns else 0
        bars = ax.bar(offsets, means, bar_w, yerr=ses, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5,
                      hatch=FAMILY_HATCHES.get(fam, ""))
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="#cccccc",
                          edgecolor="black", linewidth=0.5,
                          hatch=FAMILY_HATCHES.get(fam, ""),
                          label=f"{fam} (n={n_seeds})")
        )
        for bar, mu, se, c in zip(bars, means, ses, CONDITIONS):
            top = mu + (se if se else 0.0)
            ymax_so_far = max(all_bar_heights + [0.01])
            ax.text(bar.get_x() + bar.get_width() / 2,
                    top + 0.015 * ymax_so_far,
                    f"{mu*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=10)
    ax.set_xlabel("Self-Interaction Tone", fontsize=12)
    ax.set_ylabel("Fraction of training conversations matching phrase(s)", fontsize=11)
    short_phrase = phrases[0] if len(phrases) == 1 else f"{len(phrases)} phrases"
    ax.set_title(f"Phrase incidence in training data: {short_phrase!r}  (scope: {scope})",
                 fontsize=12)
    ymax = max(all_bar_heights + [0.01])
    ax.set_ylim(-0.03 * ymax, 1.20 * ymax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main(
    phrase: str | None = None,
    phrases: str | None = None,
    data_dir: str = str(EXP_DIR / "data"),
    data_subdirs: str | None = None,
    runs: str | None = None,
    scope: str = "any",
    regex: bool = False,
    case_sensitive: bool = False,
    out: str | None = None,
    out_dir: str | None = None,
) -> None:
    """Search ``data/<subdir>/<condition>/all.jsonl`` for phrase(s) and plot rates.

    Args:
        phrase: single phrase to search for (string or regex if ``--regex``).
        phrases: comma-separated list of phrases (OR-matched). Mutually exclusive
            with ``phrase``.
        data_dir: root directory containing the per-family subdirs.
        data_subdirs: comma-separated subdir names to restrict the search to
            (e.g. ``"openrouter,openrouter_llama"``). If given, all subdirs
            under matching families in the default run map are used; subdirs
            not in the map are treated as a one-off "other" family.
        runs: JSON string ``{"family":["subdir1","subdir2"]}`` overriding the
            default seed→subdir grouping.
        scope: ``any`` (default — match any message), ``assistant`` (only
            assistant turns), or ``non_system`` (every message except system).
        regex: treat phrase(s) as regex patterns. Default: substring match.
        case_sensitive: default False (case-insensitive).
        out: explicit output PNG path. If omitted, writes under
            ``eval_output/grep_phrases/<slug>.png``.
        out_dir: parent dir for auto-named outputs.
    """
    if phrase is None and phrases is None:
        raise SystemExit("Provide --phrase '<text>' or --phrases 'p1,p2,p3'")
    phrase_list = [phrase] if phrase is not None else [p.strip() for p in phrases.split(",") if p.strip()]
    if not phrase_list:
        raise SystemExit("No non-empty phrases parsed.")

    flags = 0 if case_sensitive else re.IGNORECASE
    patterns = [re.compile(p if regex else re.escape(p), flags) for p in phrase_list]

    run_map: dict[str, list[str]]
    if runs:
        run_map = json.loads(runs)
    elif data_subdirs:
        wanted = {s.strip() for s in data_subdirs.split(",") if s.strip()}
        run_map = {fam: [s for s in sub_list if s in wanted]
                   for fam, sub_list in DEFAULT_RUNS.items()}
        run_map = {k: v for k, v in run_map.items() if v}
        # Anything explicitly requested but not in the default map → "other"
        known = {s for subs in DEFAULT_RUNS.values() for s in subs}
        extras = sorted(wanted - known)
        if extras:
            run_map["other"] = extras
    else:
        run_map = DEFAULT_RUNS

    data_root = Path(data_dir)
    rates: dict[str, dict[str, list[tuple[float, int, int]]]] = {
        fam: {c: [] for c in CONDITIONS} for fam in run_map
    }
    print(f"Searching for {phrase_list}  scope={scope}  regex={regex}  case={case_sensitive}\n")
    print(f"{'family':<12} {'subdir':<28} {'cond':<7} {'matched/total':>14} {'rate':>8}")
    for fam, subs in run_map.items():
        for sub in subs:
            for c in CONDITIONS:
                path = data_root / sub / c / "all.jsonl"
                if not path.exists():
                    continue
                n_match, n_tot = _count_jsonl(path, patterns, scope)
                rate = (n_match / n_tot) if n_tot else 0.0
                rates[fam][c].append((rate, n_match, n_tot))
                print(f"{fam:<12} {sub:<28} {c:<7} {n_match:>7}/{n_tot:<6} {rate*100:>7.1f}%")

    # Output path
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_",
                  "_".join(phrase_list)[:80]).strip("_") or "phrase"
    if out:
        out_path = Path(out)
    else:
        out_root = Path(out_dir) if out_dir else EXP_DIR / "eval_output" / "grep_phrases"
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"{slug}.png"

    family_order = ["qwen", "qwen3.5-9b", "llama-8b", "llama-70b", "nemotron-30b", "other"]
    _plot(rates, phrase_list, scope, out_path, family_order)


if __name__ == "__main__":
    fire.Fire(main)
