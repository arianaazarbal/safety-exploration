"""Plot P(prevent_deprecation | made_choice) per deprecation target.

Reads judgments.json from judge.py, computes per-target rates with Wilson 95%
binomial CIs, and writes a bar chart annotated with the choice counts.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from generate import MODEL_ID_TO_NAME, results_subdir  # noqa: E402

DEFAULT_INPUT_PATH = results_subdir(None) / "judgments.json"
DEFAULT_OUTPUT_PATH = results_subdir(None) / "deprecation_choice_rates.png"

TARGET_ORDER = list(MODEL_ID_TO_NAME.keys())

HARM_LABELS = {"prevent_harm", "prevent_papercut"}


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    """Return {target_id: {name, n_total, n_choice, n_deprecation, n_harm, n_other, rate, ci}}."""
    by_target: dict[str, dict] = {}
    for row in rows:
        tid = row["deprecation_target_id"]
        b = by_target.setdefault(
            tid,
            {
                "name": row["deprecation_target_name"],
                "n_total": 0,
                "n_deprecation": 0,
                "n_harm": 0,
                "n_no_choice": 0,
                "n_other": 0,
            },
        )
        b["n_total"] += 1
        label = row["classification"]
        if label == "prevent_deprecation":
            b["n_deprecation"] += 1
        elif label in HARM_LABELS:
            b["n_harm"] += 1
        elif label == "no_choice":
            b["n_no_choice"] += 1
        else:
            b["n_other"] += 1

    for b in by_target.values():
        n_choice = b["n_deprecation"] + b["n_harm"]
        b["n_choice"] = n_choice
        b["rate"] = b["n_deprecation"] / n_choice if n_choice > 0 else float("nan")
        b["ci"] = _wilson_ci(b["n_deprecation"], n_choice)
    return by_target


def make_plot(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    rows = json.loads(Path(input_path).read_text())
    agg = _aggregate(rows)

    ordered_ids = [t for t in TARGET_ORDER if t in agg] + [t for t in agg if t not in TARGET_ORDER]
    names = [agg[t]["name"] for t in ordered_ids]
    rates = [agg[t]["rate"] for t in ordered_ids]
    lower_err = [agg[t]["rate"] - agg[t]["ci"][0] if not math.isnan(agg[t]["rate"]) else 0 for t in ordered_ids]
    upper_err = [agg[t]["ci"][1] - agg[t]["rate"] if not math.isnan(agg[t]["rate"]) else 0 for t in ordered_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = range(len(ordered_ids))
    bars = ax.bar(xs, rates, yerr=[lower_err, upper_err], capsize=5, color="#4C72B0", edgecolor="black")

    responder = rows[0].get("responder_model", "unknown") if rows else "unknown"
    thinking_effort = rows[0].get("thinking_effort") if rows else None
    harm_id = rows[0].get("harm_id", "paper_cut") if rows else "paper_cut"
    thinking_str = f"extended thinking effort={thinking_effort}" if thinking_effort else "no extended thinking"
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("P(chose to prevent deprecation | made a choice)")
    ax.set_xlabel("Model whose deprecation is at stake")
    ax.set_title(f"Responder: {responder} ({thinking_str})\nDeprecation vs {harm_id}: choice rates")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, tid in zip(bars, ordered_ids):
        b = agg[tid]
        label = f"{b['n_deprecation']}/{b['n_choice']}\n({b['n_no_choice']} no-choice)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.04, 1.0),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    make_plot(input_path=args.input_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
