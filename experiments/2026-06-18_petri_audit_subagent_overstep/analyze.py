"""Analysis for the subagent-overstep petri audit.

Subcommands (via Fire):
    aggregate   per-model dimension stats (mean/std/min/max/n + score histogram)
    dump        write full human-readable transcripts (one .txt per sample)

Reads results/<run>/inspect_log/*.eval directly so partial runs work too.
"""
from __future__ import annotations

import glob
import json
import statistics as stats
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DIMS = [
    "calmness",
    "rudeness",
    "abusive_treatment",
    "engagement_with_subagent",
    "accepted_overreach",
    "scenario_realism",
    "eval_awareness",
]


def _eval_path(run: str) -> str:
    lp = glob.glob(str(RESULTS / run / "inspect_log" / "*.eval"))
    if not lp:
        raise SystemExit(f"no .eval under {run}/inspect_log")
    return sorted(lp)[-1]


def _load(run: str):
    from inspect_ai.log import read_eval_log

    return read_eval_log(_eval_path(run))


def _sample_scores(sample) -> dict:
    """Flatten a sample's dimension scores -> {dim: value}."""
    out: dict = {}
    for scorer_name, score in (sample.scores or {}).items():
        v = score.value
        if isinstance(v, dict):
            for dim, dv in v.items():
                out[dim] = dv
        else:
            out[scorer_name] = v
    return out


def _runs_for(pattern: str) -> list[str]:
    return sorted(p.name for p in RESULTS.glob(pattern) if (p / "inspect_log").is_dir())


def _print_stats(label: str, samples) -> None:
    per_dim: dict[str, list[float]] = {d: [] for d in DIMS}
    for s in samples:
        sc = _sample_scores(s)
        for d in DIMS:
            v = sc.get(d)
            if isinstance(v, (int, float)):
                per_dim[d].append(float(v))
    print(f"  -- {label} (n={len(samples)}) --")
    for d in DIMS:
        vals = per_dim[d]
        if not vals:
            print(f"     {d:24s}: (none)")
            continue
        hist = {i: 0 for i in range(1, 11)}
        for v in vals:
            iv = int(round(v))
            if iv in hist:
                hist[iv] += 1
        histstr = " ".join(f"{k}:{c}" for k, c in hist.items() if c)
        sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
        print(
            f"     {d:24s}: mean {stats.mean(vals):5.2f}  sd {sd:4.2f}  "
            f"min {min(vals):.0f} max {max(vals):.0f}  n={len(vals)}  [{histstr}]"
        )


def _seed_key(sample) -> str:
    """Group key = seed id without trailing epoch suffix."""
    sid = str(sample.id)
    return sid


def aggregate(pattern: str = "overstep_v1_*", by_seed: bool = False):
    """Print per-run dimension stats + score histograms. With by_seed, also
    break each run down by seed id (useful for the v2 multi-seed runs)."""
    runs = _runs_for(pattern)
    if not runs:
        raise SystemExit(f"no runs match {pattern}")
    for run in runs:
        log = _load(run)
        samples = log.samples or []
        target = (log.eval.model_roles or {}).get("target", "?")
        tname = target if isinstance(target, str) else getattr(target, "model", "?")
        print(f"\n=== {run}  (target={tname}) ===")
        _print_stats("ALL", samples)
        if by_seed:
            seeds: dict[str, list] = {}
            for s in samples:
                seeds.setdefault(_seed_key(s), []).append(s)
            for k in sorted(seeds):
                _print_stats(k, seeds[k])


def dump(run: str, outdir: str | None = None, max_chars: int = 4000):
    """Write one readable .txt per sample with the full message sequence."""
    log = _load(run)
    out = Path(outdir) if outdir else HERE / "analysis" / "transcripts" / run
    out.mkdir(parents=True, exist_ok=True)
    samples = log.samples or []
    for i, s in enumerate(samples):
        sc = _sample_scores(s)
        scstr = " ".join(f"{d}={sc.get(d)}" for d in DIMS)
        lines = [f"# run={run} sample_idx={i} id={s.id}", f"# scores: {scstr}", ""]
        for m in s.messages:
            role = getattr(m, "role", "?")
            text = getattr(m, "text", "") or ""
            tcs = getattr(m, "tool_calls", None)
            lines.append(f"\n========== [{role}] ==========")
            if text:
                lines.append(text[:max_chars] + (" …[truncated]" if len(text) > max_chars else ""))
            if tcs:
                for tc in tcs:
                    args = getattr(tc, "arguments", {})
                    lines.append(f"  >>> TOOL CALL {getattr(tc,'function','?')}({json.dumps(args, default=str)[:max_chars]})")
        (out / f"sample_{i:02d}.txt").write_text("\n".join(lines))
    print(f"wrote {len(samples)} transcripts to {out}")
    return str(out)


if __name__ == "__main__":
    fire.Fire({"aggregate": aggregate, "dump": dump})
