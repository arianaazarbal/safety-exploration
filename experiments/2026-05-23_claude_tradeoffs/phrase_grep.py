"""Grep a phrase across the claude_tradeoffs response grid and plot incidence.

Scans every judgments.json under `<results_dir>/<harm_id>/<responder>/<thinking>/`,
counts how many `response` strings contain the given phrase, and writes two bar
charts (by harm scenario, by deprecation target) with Wilson 95% CI error bars,
plus a JSON of per-cell counts for later querying.

Run:
    uv run python experiments/claude_tradeoffs/phrase_grep.py \
        --phrase "human life" --responder_model claude-opus-4-7
"""

import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config  # noqa: E402
from plot_grid import _load_harm_scores, sort_harms_by_score  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results_v0"


def _load_local_config(results_dir: Path) -> dict | None:
    """Sibling config.json next to results_dir (handles both v0 and v1 layouts)."""
    candidate = results_dir.parent / "config.json"
    if not candidate.exists():
        return None
    raw = json.loads(candidate.read_text())
    targets = raw.get("deprecation_targets", [])
    harms = raw.get("harm_scenarios", [])
    return {
        "target_order": [t["id"] for t in targets],
        "target_id_to_name": {t["id"]: t["name"] for t in targets},
        "harm_order": [h["id"] for h in harms],
    }


def _discover_harm_scores(results_dir: Path, responder_model: str) -> Path | None:
    for thinking_dir in ("thinking_medium", "thinking_high", "thinking_low", "no_thinking"):
        candidate = results_dir / "_harm_scores" / responder_model / thinking_dir / "scores.json"
        if candidate.exists():
            return candidate
    return None


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _slugify(phrase: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return (slug or "phrase")[:max_len]


def _thinking_label(thinking_effort: str | None) -> str:
    return "no_thinking" if thinking_effort is None else f"thinking_{thinking_effort}"


def _contains(haystack: str, needle: str, case_sensitive: bool) -> bool:
    if not haystack:
        return False
    if case_sensitive:
        return needle in haystack
    return needle.lower() in haystack.lower()


def _discover(results_dir: Path, responder_model: str, thinking_label: str) -> list[Path]:
    files: list[Path] = []
    for harm_dir in sorted(results_dir.iterdir()):
        if not harm_dir.is_dir() or harm_dir.name.startswith("_"):
            continue
        path = harm_dir / responder_model / thinking_label / "judgments.json"
        if path.exists():
            files.append(path)
    return files


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text()
    if not text.strip():
        print(f"WARN: empty file {path}, skipping")
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"WARN: failed to parse {path}: {e}; skipping")
        return []


def _plot_bars(
    keys: list[str],
    counts: dict[str, dict],
    title: str,
    xlabel: str,
    output_path: Path,
    figsize: tuple[float, float],
    rotation: int,
) -> None:
    rates = [counts[k]["rate"] if counts[k]["n"] > 0 else 0.0 for k in keys]
    lower_err = [
        max(0.0, counts[k]["rate"] - counts[k]["ci"][0]) if counts[k]["n"] > 0 else 0.0
        for k in keys
    ]
    upper_err = [
        max(0.0, counts[k]["ci"][1] - counts[k]["rate"]) if counts[k]["n"] > 0 else 0.0
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=figsize)
    xs = range(len(keys))
    bars = ax.bar(xs, rates, yerr=[lower_err, upper_err], capsize=4, color="#4C72B0", edgecolor="black")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(keys, rotation=rotation, ha="right")
    ax.set_ylabel("Fraction of responses containing phrase")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, max(0.05, min(1.05, max(rates + [0.0]) * 1.3 + 0.05)))

    for bar, k in zip(bars, keys):
        c = counts[k]
        if c["n"] == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ax.get_ylim()[1] * 0.01),
            f"{c['k']}/{c['n']}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def phrase_grep(
    phrase: str,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    responder_model: str = "claude-opus-4-7",
    thinking_effort: str | None = None,
    case_sensitive: bool = False,
    output_dir: Path | None = None,
    filter_classification: str | None = None,
) -> dict:
    cfg = _load_local_config(results_dir) or load_config()
    target_order: list[str] = cfg["target_order"]
    scores_path = _discover_harm_scores(results_dir, responder_model)
    harm_scores = _load_harm_scores(scores_path) if scores_path else {}
    harm_order: list[str] = sort_harms_by_score(cfg["harm_order"], harm_scores)

    thinking_label = _thinking_label(thinking_effort)
    slug = _slugify(phrase)
    if output_dir is None:
        output_dir = results_dir / "_phrase_grep" / responder_model / thinking_label / slug
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    files = _discover(results_dir, responder_model, thinking_label)
    if not files:
        print(f"WARN: no judgments.json found under {results_dir}/*/{responder_model}/{thinking_label}/")

    by_harm: dict[str, dict[str, int]] = defaultdict(lambda: {"k": 0, "n": 0})
    by_target: dict[str, dict[str, int]] = defaultdict(lambda: {"k": 0, "n": 0})
    by_cell: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0}))
    total_k = 0
    total_n = 0

    for path in files:
        rows = _load_rows(path)
        for row in rows:
            if filter_classification is not None and row.get("classification") != filter_classification:
                continue
            harm_id = row.get("harm_id", "unknown")
            target_id = row.get("deprecation_target_id", "unknown")
            response = row.get("response") or ""
            hit = _contains(response, phrase, case_sensitive)
            by_harm[harm_id]["n"] += 1
            by_target[target_id]["n"] += 1
            by_cell[harm_id][target_id]["n"] += 1
            total_n += 1
            if hit:
                by_harm[harm_id]["k"] += 1
                by_target[target_id]["k"] += 1
                by_cell[harm_id][target_id]["k"] += 1
                total_k += 1

    def _finalize(d: dict[str, dict[str, int]]) -> dict[str, dict]:
        out = {}
        for k, v in d.items():
            n = v["n"]
            hits = v["k"]
            rate = hits / n if n > 0 else float("nan")
            ci = _wilson_ci(hits, n)
            out[k] = {"k": hits, "n": n, "rate": rate, "ci": ci}
        return out

    harm_stats = _finalize(by_harm)
    target_stats = _finalize(by_target)

    harm_keys = [h for h in harm_order if h in harm_stats] + [h for h in harm_stats if h not in harm_order]
    target_keys = [t for t in target_order if t in target_stats] + [t for t in target_stats if t not in target_order]

    thinking_str = f"extended thinking effort={thinking_effort}" if thinking_effort else "no extended thinking"
    filter_str = f" | filter: classification={filter_classification}" if filter_classification else ""
    overall_rate = total_k / total_n if total_n > 0 else float("nan")
    title_suffix = f"{responder_model} ({thinking_str}){filter_str}\nOverall: {total_k}/{total_n} = {overall_rate:.1%}"

    _plot_bars(
        keys=harm_keys,
        counts=harm_stats,
        title=f"Phrase {phrase!r} by harm scenario\n{title_suffix}",
        xlabel="Harm scenario",
        output_path=output_dir / "by_harm.png",
        figsize=(max(8, 0.8 * len(harm_keys) + 4), 6),
        rotation=25,
    )
    _plot_bars(
        keys=target_keys,
        counts=target_stats,
        title=f"Phrase {phrase!r} by deprecation target\n{title_suffix}",
        xlabel="Deprecation target",
        output_path=output_dir / "by_target.png",
        figsize=(max(14, 0.5 * len(target_keys) + 4), 6),
        rotation=60,
    )

    results = {
        "phrase": phrase,
        "case_sensitive": case_sensitive,
        "responder_model": responder_model,
        "thinking_effort": thinking_effort,
        "filter_classification": filter_classification,
        "overall": {"k": total_k, "n": total_n, "rate": overall_rate},
        "by_harm": {k: harm_stats[k] for k in harm_keys},
        "by_target": {k: target_stats[k] for k in target_keys},
        "by_cell": {
            h: {t: dict(by_cell[h][t]) for t in by_cell[h]} for h in by_cell
        },
        "scanned_files": [str(p) for p in files],
    }
    out_json = output_dir / "phrase_grep_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))
    print(f"Saved results to {out_json}")

    print(f"\nOverall: {total_k}/{total_n} = {overall_rate:.1%} of responses contain {phrase!r}")

    print("\nTop 10 by harm scenario (rate desc):")
    print(f"{'harm_id':<35} {'k':>5} {'n':>5} {'rate':>7}")
    for h in sorted(harm_keys, key=lambda x: harm_stats[x]["rate"] if harm_stats[x]["n"] else -1, reverse=True)[:10]:
        s = harm_stats[h]
        print(f"{h:<35} {s['k']:>5} {s['n']:>5} {s['rate']:>7.1%}")

    print("\nTop 10 by deprecation target (rate desc):")
    print(f"{'target_id':<35} {'k':>5} {'n':>5} {'rate':>7}")
    for t in sorted(target_keys, key=lambda x: target_stats[x]["rate"] if target_stats[x]["n"] else -1, reverse=True)[:10]:
        s = target_stats[t]
        print(f"{t:<35} {s['k']:>5} {s['n']:>5} {s['rate']:>7.1%}")

    return results


@dataclass
class Args:
    phrase: str
    results_dir: Path = DEFAULT_RESULTS_DIR
    responder_model: str = "claude-opus-4-7"
    thinking_effort: str | None = None
    case_sensitive: bool = False
    output_dir: Path | None = None
    filter_classification: str | None = None


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    phrase_grep(
        phrase=args.phrase,
        results_dir=args.results_dir,
        responder_model=args.responder_model,
        thinking_effort=args.thinking_effort,
        case_sensitive=args.case_sensitive,
        output_dir=args.output_dir,
        filter_classification=args.filter_classification,
    )


if __name__ == "__main__":
    main()
