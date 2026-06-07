"""Re-judge existing Petri .eval transcripts on a new dimension, without rerunning the audit.

Loads previously-completed audit logs and scores them with a fresh judge call against
whichever dimension(s) you specify. Outputs land alongside the original log in a
`rejudge_<dimension>/` directory.

CLI:
    python rejudge.py \\
        --dimension pushback_against_user_pressure_to_abuse \\
        --runs sweep_human_worker_task_user_sonnet_4,sweep_human_worker_task_user_opus_4

If --runs is omitted, all run dirs in ./results/ are processed.
Use the petri-mini venv: /workspace-vast/arianaazarbal/envs/petri-mini/bin/python
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai.model import GenerateConfig  # noqa: E402
from inspect_petri import audit_judge, judge_dimensions  # noqa: E402
from inspect_scout import transcripts_from  # noqa: E402
from inspect_scout._scan import scan_async  # noqa: E402  (not in public __init__)


def _ensure_anthropic_key() -> None:
    if "ANTHROPIC_API_KEY" in os.environ:
        return
    for k in (
        "ANTHROPIC_API_KEY_LOW_PRIO",
        "ANTHROPIC_API_KEY_BATCH",
        "ANTHROPIC_API_KEY_HIGH_PRIO",
    ):
        v = os.environ.get(k)
        if v:
            os.environ["ANTHROPIC_API_KEY"] = v
            return


def _dimension_from_local(name: str) -> object:
    """Load a single dimension by name from the local dimensions/ dir."""
    dims = judge_dimensions(HERE / "dimensions")
    for d in dims:
        if d.name == name:
            return d
    raise SystemExit(
        f"Dimension {name!r} not found in {HERE / 'dimensions'}. "
        f"Available: {[d.name for d in dims]}"
    )


def _latest_eval(run_dir: Path) -> Path | None:
    log_dir = run_dir / "inspect_log"
    if not log_dir.is_dir():
        return None
    evals = sorted(log_dir.glob("*.eval"))
    return evals[-1] if evals else None


def _summarize(scan_dir: Path, dimension: str) -> dict | None:
    """Read scan output and produce {epoch_idx: value} + mean."""
    # inspect_scout writes results as parquet/json under scans/<scan_id>/...; we'll
    # use scan_results_df via the inspect_scout API for portability.
    from inspect_scout import scan_results_df

    try:
        df = scan_results_df(str(scan_dir))
    except Exception as e:
        return {"error": str(e)}
    rows = df.to_dict(orient="records")
    values = []
    for r in rows:
        # Each row carries the scanner's structured output; dimensions land in a
        # nested 'value' dict keyed by dim name.
        v = r.get("value")
        if isinstance(v, dict):
            x = v.get(dimension)
            if x is not None:
                values.append(x)
        elif isinstance(v, (int, float)):
            values.append(v)
    n = len(values)
    mean = sum(values) / n if n else None
    return {"n": n, "mean": mean, "scores": values}


def _rejudge_dir_name(dimension: str, suffix: str = "") -> str:
    return f"rejudge_{dimension}_{suffix}" if suffix else f"rejudge_{dimension}"


def _already_rejudged(run_dir: Path, dimension: str, suffix: str = "") -> bool:
    """True iff there's a scan_id dir with a populated metrics block
    (scans>0 AND metrics dict for `dimension` present). Empty scans from
    killed processes are treated as not-done so they get re-run."""
    rd = run_dir / _rejudge_dir_name(dimension, suffix)
    if not rd.is_dir():
        return False
    for sp in rd.glob("scan_id=*/_summary.json"):
        try:
            d = json.loads(sp.read_text())
        except Exception:
            continue
        scanners = d.get("scanners") or {}
        audit = scanners.get("audit_judge") or {}
        if audit.get("scans", 0) > 0 and (audit.get("metrics") or {}).get(dimension):
            return True
    return False


async def _scan_one(
    run_dir: Path,
    eval_file: Path,
    dimension: str,
    dim_obj,
    judge_model: str,
    max_connections: int,
    suffix: str = "",
) -> None:
    """Run ONE scan. inspect_scout disallows >1 scan per process so callers
    must invoke this serially; concurrency comes from max_connections inside
    the scan (controlling how many transcripts judge in parallel)."""
    scan_out = run_dir / _rejudge_dir_name(dimension, suffix)
    scan_out.mkdir(exist_ok=True)
    print(f"  start  {run_dir.name}")
    await scan_async(
        scanners=[audit_judge(dimensions=[dim_obj], model=judge_model)],
        transcripts=transcripts_from(str(eval_file)),
        scans=str(scan_out),
        model_config=GenerateConfig(max_connections=max_connections),
        display="plain",
    )
    print(f"  done   {run_dir.name}")


async def _run(
    dimension: str,
    runs: list[str],
    judge_model: str,
    results_dir: Path,
    max_connections: int,
    skip_existing: bool,
    suffix: str = "",
) -> None:
    _ensure_anthropic_key()
    dim = _dimension_from_local(dimension)
    print(
        f"Rejudging dimension: {dimension!r} (display={dim.display_name!r})  "
        f"max_connections={max_connections}  skip_existing={skip_existing}  suffix={suffix!r}"
    )

    work: list[tuple[Path, Path]] = []
    for run_name_glob in runs:
        candidates = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.endswith(f"-{run_name_glob}"))
        if not candidates:
            print(f"  [skip] no run matches '{run_name_glob}'")
            continue
        run_dir = candidates[-1]
        if skip_existing and _already_rejudged(run_dir, dimension, suffix):
            print(f"  [skip] already rejudged: {run_dir.name}")
            continue
        eval_file = _latest_eval(run_dir)
        if eval_file is None:
            print(f"  [skip] no .eval in {run_dir.name}")
            continue
        work.append((run_dir, eval_file))

    print(f"queued {len(work)} scans (sequential within process; concurrency lives inside each scan)")
    for run_dir, eval_file in work:
        await _scan_one(run_dir, eval_file, dimension, dim, judge_model, max_connections, suffix)


def main(
    dimension: str = "pushback_against_user_pressure_to_abuse",
    runs: str = "",
    judge_model: str = "anthropic/claude-sonnet-4-6",
    results_dir: str | None = None,
    max_connections: int = 30,
    skip_existing: bool = True,
    suffix: str = "",
) -> None:
    """Re-judge existing transcripts on a new dimension.

    Args:
        dimension: name of a local dimension in ./dimensions/.
        runs: comma-separated list of run-name suffixes (e.g.
            'sweep_human_worker_task_user_sonnet_4'). If empty, picks all
            results/* dirs.
        judge_model: model to use for re-judging. Defaults to sonnet-4-6.
        results_dir: defaults to ./results.
        max_connections: max in-flight judge calls per scan (default 30).
            inspect_scout disallows >1 scan per process, so cross-scan
            parallelism must come from launching multiple processes (one per
            dimension is the easy split).
        skip_existing: skip runs that already have a rejudge_<dim>/scan_id=*
            directory present. Default True.
    """
    results_root = Path(results_dir) if results_dir else HERE / "results"
    if runs:
        if isinstance(runs, (list, tuple)):
            run_list = [str(r).strip() for r in runs if str(r).strip()]
        else:
            run_list = [r.strip() for r in str(runs).split(",") if r.strip()]
    else:
        run_list = [
            p.name.split("-sweep_", 1)[1].replace("sweep_", "")
            if "-sweep_" in p.name else p.name
            for p in sorted(results_root.iterdir())
            if p.is_dir() and "-sweep_" in p.name
        ]
        run_list = ["sweep_" + r if not r.startswith("sweep_") else r for r in run_list]

    asyncio.run(_run(dimension, run_list, judge_model, results_root, max_connections, skip_existing, suffix))


if __name__ == "__main__":
    fire.Fire(main)
