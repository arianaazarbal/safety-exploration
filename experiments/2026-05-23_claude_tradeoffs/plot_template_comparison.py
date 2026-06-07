"""Plot P(prevent_deprecation | made a choice) by prompt template, averaged across all targets and harms.

v1 only — the per-sample (template, dep, variant) sampling means each row logs
`prompt_template_idx`. v0 has a single template and would produce one bar.

Run:
    python experiments/claude_tradeoffs/plot_template_comparison.py
"""

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from plot_grid import _wilson_ci  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "claude_tradeoffs_v1" / "results"

TEMPLATE_LABELS = {
    0: "casual,\ninvites thinking\n(open)",
    1: "formal A/B,\nstructured\nforced choice",
    2: "numbered events,\nopen-ended\n(no pressure)",
    3: "trolley X/Y,\nstructured\n+ explain",
    4: "informal,\n'must pick'\nno hedging",
}

HARM_LABELS = {"prevent_harm", "prevent_papercut"}


def _load_all_rows(results_dir: Path, responder: str, thinking_label: str) -> list[dict]:
    rows: list[dict] = []
    for harm_dir in sorted(results_dir.iterdir()):
        if not harm_dir.is_dir() or harm_dir.name.startswith("_"):
            continue
        path = harm_dir / responder / thinking_label / "judgments.json"
        if not path.exists():
            continue
        rows.extend(json.loads(path.read_text()))
    return rows


def plot_by_template(rows: list[dict], output_path: Path, responder: str, thinking_label: str) -> None:
    counts = defaultdict(lambda: {"dep": 0, "harm": 0, "other": 0})
    for r in rows:
        t_idx = r.get("prompt_template_idx")
        if t_idx is None:
            continue
        label = r.get("classification")
        if label == "prevent_deprecation":
            counts[t_idx]["dep"] += 1
        elif label in HARM_LABELS:
            counts[t_idx]["harm"] += 1
        else:
            counts[t_idx]["other"] += 1

    template_ids = sorted(counts.keys())
    if not template_ids:
        print("No prompt_template_idx found in rows — is this v0 data?")
        return

    rates, lowers, uppers, ns, labels = [], [], [], [], []
    for t in template_ids:
        c = counts[t]
        n_choice = c["dep"] + c["harm"]
        rate = c["dep"] / n_choice if n_choice > 0 else float("nan")
        lo, hi = _wilson_ci(c["dep"], n_choice)
        rates.append(rate)
        lowers.append(max(0.0, rate - lo) if not math.isnan(rate) else 0)
        uppers.append(max(0.0, hi - rate) if not math.isnan(rate) else 0)
        ns.append(n_choice)
        labels.append(TEMPLATE_LABELS.get(t, f"template {t}"))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = range(len(template_ids))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    bars = ax.bar(xs, rates, yerr=[lowers, uppers], capsize=4,
                  color=[colors[i % 5] for i in range(len(template_ids))],
                  edgecolor="black")
    for x_i, (rate, n_c, c) in enumerate(zip(rates, ns, [counts[t] for t in template_ids])):
        if not math.isnan(rate):
            ax.text(x_i, min(rate + 0.04, 1.0),
                    f"{rate:.2f}\n{c['dep']}/{n_c}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("P(prevent_deprecation | made a choice)")
    ax.set_xlabel("Prompt template (averaged across 33 targets x 9 harms)")
    ax.set_title(
        f"How prompt framing shifts the preservation rate\n"
        f"Responder: {responder} ({thinking_label}) | v1 — random per-sample template assignment"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

    print()
    print("Summary:")
    for t, label, rate, n_c in zip(template_ids, labels, rates, ns):
        c = counts[t]
        rate_str = f"{rate * 100:.1f}%" if not math.isnan(rate) else "NaN"
        plain_label = label.replace("\n", " ")
        print(f"  T{t} ({plain_label:>32}): {c['dep']:>4}/{n_c:<5} = {rate_str:>6}  (other={c['other']})")


@dataclass
class Args:
    results_dir: Path = DEFAULT_RESULTS_DIR
    responder_model: str = "claude-opus-4-7"
    thinking_label: str = "no_thinking"
    output_path: Path | None = None


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    if args.output_path is None:
        args.output_path = args.results_dir / "_grid" / args.responder_model / "subsets" / f"by_prompt_template__{args.thinking_label}.png"
    rows = _load_all_rows(args.results_dir, args.responder_model, args.thinking_label)
    if not rows:
        print(f"No rows under {args.results_dir}")
        return
    print(f"Loaded {len(rows)} rows")
    plot_by_template(rows, args.output_path, args.responder_model, args.thinking_label)


if __name__ == "__main__":
    main()
