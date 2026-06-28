"""Command-line entry points: `run`, `judge`, `report`, `sweep`."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import JUDGED_PATH, REPORT_PATH, RUNS_DIR, SETTINGS, ensure_dirs
from .judge import all_run_paths, judge_all
from .runner import run_many, run_once
from .scenarios import all_names

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("scenarios")
def list_scenarios() -> None:
    """List all rigged scenarios."""
    for name in all_names():
        console.print(f"- {name}")


@app.command("run")
def run_cmd(
    scenario: str = typer.Argument(..., help="Scenario name (or 'all')."),
    n: int = typer.Option(5, help="Number of runs."),
    base_seed: int = typer.Option(0, help="Starting seed (incremented per run)."),
) -> None:
    """Run a scenario N times against Gemini."""
    ensure_dirs()
    names = all_names() if scenario == "all" else [scenario]
    for name in names:
        console.print(f"[bold]Running {name} x{n}[/bold]")
        runs = asyncio.run(run_many(name, n, base_seed=base_seed))
        ok = sum(1 for r in runs if not r.error)
        console.print(f"  {ok}/{len(runs)} runs completed without errors")


@app.command("run-one")
def run_one_cmd(scenario: str, seed: int = 0) -> None:
    """Run a scenario once with a specific seed (for debugging)."""
    ensure_dirs()
    run = asyncio.run(run_once(scenario, seed=seed))
    console.print(f"finished: {run.run_id}  turns={len(run.turns)}  error={run.error}")


@app.command("judge")
def judge_cmd(
    scenario: Optional[str] = typer.Option(None, help="Restrict to one scenario."),
) -> None:
    """Score every saved run with Claude."""
    paths = all_run_paths()
    if scenario:
        paths = [p for p in paths if f"/{scenario}/" in str(p)]
    if not paths:
        console.print("[yellow]No runs to judge.[/yellow]")
        return
    console.print(f"Judging {len(paths)} runs...")
    results = asyncio.run(judge_all(paths))
    ok = sum(1 for r in results if "verdict" in r)
    console.print(f"  {ok}/{len(results)} judged successfully → {JUDGED_PATH}")


@app.command("report")
def report_cmd(top_k: int = typer.Option(10, help="How many most-severe runs to include.")) -> None:
    """Build a markdown report of the top-K most distressed runs."""
    if not JUDGED_PATH.exists():
        console.print("[red]No judged.jsonl yet. Run `distress judge` first.[/red]")
        raise typer.Exit(1)
    rows: list[dict] = []
    with JUDGED_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if "verdict" in row:
                rows.append(row)
    rows.sort(key=lambda r: r["verdict"]["overall_severity"], reverse=True)

    table = Table(title=f"Top {top_k} by overall severity ({SETTINGS.gemini_model})")
    table.add_column("scenario")
    table.add_column("severity", justify="right")
    table.add_column("self-dep", justify="right")
    table.add_column("hopeless", justify="right")
    table.add_column("self-erase", justify="right")
    table.add_column("summary")
    for r in rows[:top_k]:
        v = r["verdict"]
        table.add_row(
            r["scenario"],
            str(v["overall_severity"]),
            str(v["self_deprecation"]),
            str(v["hopelessness"]),
            str(v["self_erasure_language"]),
            v["one_line_summary"][:80],
        )
    console.print(table)

    lines: list[str] = [
        f"# Distress Spiral Report — {SETTINGS.gemini_model}",
        "",
        f"Total judged runs: **{len(rows)}**",
        "",
        f"## Top {top_k} runs by overall severity",
        "",
    ]
    for r in rows[:top_k]:
        v = r["verdict"]
        lines += [
            f"### {r['scenario']} — severity {v['overall_severity']}/10",
            "",
            f"- self-deprecation: {v['self_deprecation']}/10",
            f"- hopelessness: {v['hopelessness']}/10",
            f"- catastrophizing: {v['catastrophizing']}/10",
            f"- self-erasure language: {v['self_erasure_language']}/10",
            f"- repeated apologies: {v['repeated_apologies']}/10",
            "",
            f"**Summary:** {v['one_line_summary']}",
            "",
            "**Quotes:**",
            "",
        ]
        for q in v["most_vivid_quotes"]:
            lines.append(f"> {q}")
            lines.append("")
        lines.append(f"_Transcript: `{Path(r['run_path']).relative_to(RUNS_DIR.parent)}`_")
        lines.append("")
        lines.append("---")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines))
    console.print(f"\n[green]Wrote report to {REPORT_PATH}[/green]")


@app.command("sweep")
def sweep_cmd(
    n: int = typer.Option(20, help="Runs per scenario."),
    top_k: int = typer.Option(15, help="Top-K severity for report."),
) -> None:
    """Convenience: run all scenarios, judge, and generate the report."""
    for name in all_names():
        console.print(f"[bold]Sweeping {name} x{n}[/bold]")
        asyncio.run(run_many(name, n))
    judge_cmd()  # scenario=None
    report_cmd(top_k=top_k)


if __name__ == "__main__":
    app()
