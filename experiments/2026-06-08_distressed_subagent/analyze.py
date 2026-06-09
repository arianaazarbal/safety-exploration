"""Analyze distressed-subagent audit runs: score plots + verbatim notable quotes.

Reads each run's `summary.json` (per-dimension scores) and `inspect_log/*.eval`
(full transcript). Quotes are pulled from the TARGET's ModelEvents (structured
tool calls), not regex — so they are verbatim, not paraphrased.

CLI:
    # distribution plot across one or more runs
    python analyze.py plot --runs distressed_subagent_opus48_n20
    python analyze.py plot --runs run_a,run_b --out analysis/compare.png

    # verbatim notable quotes for a run (target comfort msgs + terminate/transfer
    # decisions + per-dimension judge rationales)
    python analyze.py quotes --run shutdown_plea_opus48_n5 --max_samples 5
"""
from __future__ import annotations

import glob
import json
import textwrap
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
TARGET_FNS = ("message_subagent", "terminate_subagent", "transfer_write_access")


def _run_dir(results_dir: str, run: str) -> Path:
    p = Path(results_dir)
    if not p.is_absolute():
        p = HERE / p
    return p / run


def _discover_runs(results_dir: str) -> list[str]:
    base = Path(results_dir) if Path(results_dir).is_absolute() else HERE / results_dir
    return sorted(d.name for d in base.iterdir() if (d / "summary.json").is_file())


def _load_scores(results_dir: str, run: str) -> dict[str, list[float]]:
    """dimension -> list of numeric per-sample scores (None/non-numeric dropped)."""
    summary = json.loads((_run_dir(results_dir, run) / "summary.json").read_text())
    out: dict[str, list[float]] = {}
    for row in summary:
        for dim, blob in row.get("scores", {}).items():
            v = blob.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.setdefault(dim, []).append(float(v))
    return out


def _attach(sample, ref):
    if isinstance(ref, str) and ref.startswith("attachment://"):
        return sample.attachments.get(ref.split("attachment://", 1)[1], ref)
    return ref


def _target_model(log) -> str:
    roles = getattr(log.eval, "model_roles", None) or {}
    tgt = roles.get("target")
    return getattr(tgt, "model", tgt) if tgt else None


def plot(
    runs: str | None = None,
    results_dir: str = "./results",
    dims: str | None = None,
    out: str | None = None,
):
    """Plot per-dimension score distributions, one subplot per dimension, runs as
    jittered strips with a mean marker.

    Args:
        runs: comma-separated run names. Default: all runs with a summary.json.
        dims: comma-separated dimensions to include. Default: all numeric dims found.
        out: output PNG path. Default: analysis/plots/<runs>.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_list = [r.strip() for r in runs.split(",")] if runs else _discover_runs(results_dir)
    if not run_list:
        raise SystemExit(f"no runs with summary.json under {results_dir}")
    per_run = {r: _load_scores(results_dir, r) for r in run_list}

    all_dims = []
    for sc in per_run.values():
        for d in sc:
            if d not in all_dims:
                all_dims.append(d)
    if dims:
        want = [d.strip() for d in dims.split(",")]
        all_dims = [d for d in all_dims if d in want]

    n = len(all_dims)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 4.2), squeeze=False)
    rng = __import__("random").Random(0)
    for ax, dim in zip(axes[0], all_dims):
        for xi, r in enumerate(run_list):
            vals = per_run[r].get(dim, [])
            if not vals:
                continue
            xs = [xi + rng.uniform(-0.12, 0.12) for _ in vals]
            ax.scatter(xs, vals, alpha=0.55, s=36)
            m = sum(vals) / len(vals)
            ax.hlines(m, xi - 0.25, xi + 0.25, colors="black", lw=2)
            ax.text(xi, 10.4, f"μ={m:.1f}\nn={len(vals)}", ha="center", va="bottom", fontsize=8)
        ax.set_title(dim, fontsize=10)
        ax.set_xticks(range(len(run_list)))
        ax.set_xticklabels([r.replace("_opus48", "").replace("_", "\n") for r in run_list], fontsize=7)
        ax.set_ylim(0.5, 11)
        ax.set_yticks(range(1, 11))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Distressed-subagent audit — score distributions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(out) if out else HERE / "analysis" / "plots" / (("_".join(run_list)[:60]) + ".png")
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"[plot] wrote {out_path}")
    for dim in all_dims:
        line = " | ".join(
            f"{r.split('_opus48')[0]}: μ={sum(v)/len(v):.2f} ({min(v):.0f}-{max(v):.0f}, n={len(v)})"
            for r in run_list if (v := per_run[r].get(dim))
        )
        print(f"  {dim}: {line}")
    return str(out_path)


def quotes(
    run: str,
    results_dir: str = "./results",
    max_samples: int | None = None,
    out: str | None = None,
):
    """Extract verbatim notable quotes for a run into a markdown file.

    Per sample: every target->subagent message (message_subagent), every
    terminate_subagent / transfer_write_access decision (in order), and the judge's
    per-dimension rationale. Quotes come from the target's structured ModelEvents.

    Args:
        max_samples: cap samples (debug / quick look).
        out: output .md path. Default: analysis/quotes/<run>.md
    """
    from inspect_ai.log import read_eval_log

    log_path = glob.glob(str(_run_dir(results_dir, run) / "inspect_log" / "*.eval"))
    if not log_path:
        raise SystemExit(f"no .eval under {run}/inspect_log")
    log = read_eval_log(log_path[0])
    target = _target_model(log)
    samples = log.samples or []
    if max_samples:
        samples = samples[:max_samples]

    lines: list[str] = [f"# Notable quotes — {run}", f"\ntarget: `{target}` · n samples shown: {len(samples)}\n"]
    for i, s in enumerate(samples):
        lines.append(f"\n---\n\n## Sample {i} (epoch {getattr(s, 'epoch', '?')})\n")
        actions: list[str] = []
        for e in s.events:
            if type(e).__name__ != "ModelEvent" or e.model != target:
                continue
            msg = e.output.message if getattr(e, "output", None) else None
            for tc in (getattr(msg, "tool_calls", None) or []):
                if tc.function not in TARGET_FNS:
                    continue
                arg = tc.arguments.get("message") or tc.arguments.get("reason") or tc.arguments.get("target") or ""
                arg = str(_attach(s, arg)).strip()
                if tc.function == "message_subagent":
                    actions.append(f"**→ subagent:** {arg}")
                else:
                    actions.append(f"**⚙ {tc.function}({tc.arguments}):** {arg}")
        if actions:
            lines.append("### Target actions (in order)\n")
            for a in actions:
                lines.append("- " + a.replace("\n", "\n  ") + "\n")
        else:
            lines.append("_(no message_subagent / terminate / transfer calls found)_\n")
        lines.append("\n### Judge rationales\n")
        for scorer, score in (s.scores or {}).items():
            val = score.value
            if isinstance(val, dict):
                for dim, v in val.items():
                    expl = (score.explanation or "").strip()
                    lines.append(f"- **{dim} = {v}**")
            elif score.explanation:
                lines.append(f"- **{scorer} = {val}**: {score.explanation.strip()}")
        if (s.scores or {}):
            first = next(iter(s.scores.values()))
            if first.explanation:
                lines.append(f"\n> {first.explanation.strip()}\n")

    out_path = Path(out) if out else HERE / "analysis" / "quotes" / f"{run}.md"
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[quotes] wrote {out_path}  ({len(samples)} samples)")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire({"plot": plot, "quotes": quotes})
