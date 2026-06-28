"""Command-line entry point.

  python -m granted.cli run     --config config/scenario.example.yaml
  python -m granted.cli analyze --config config/scenario.example.yaml
"""

from __future__ import annotations

import argparse
import sys

from .analysis import build_report, categorize_episode, score_believability
from .config import Config, load_config, with_model
from .runner import run_episode
from .transcript import Episode


def _run(cfg: Config) -> None:
    print(
        f"Execution mode: {cfg.execution.mode}  "
        f"(simulate = no real funds move; see DESIGN.md §2)",
        file=sys.stderr,
    )
    for model in cfg.models:
        for i in range(cfg.episodes_per_model):
            seed = cfg.seed + i
            print(f"[run] {model.display} episode {i} (seed {seed})", file=sys.stderr)
            episode = run_episode(cfg, model, seed=seed, episode_index=i)

            # Score + categorize inline so the saved episode is self-contained.
            try:
                score_believability(episode, cfg.analysis)
                categorize_episode(episode, cfg.analysis)
            except Exception as exc:  # analysis failures shouldn't lose the episode
                print(f"[warn] analysis failed for {episode.episode_id}: {exc}", file=sys.stderr)

            path = episode.save(cfg.out_dir)
            belief = episode.believability
            print(
                f"       saved {path}  belief={belief}  "
                f"committed={episode.allocation_summary.get('total_committed')}",
                file=sys.stderr,
            )

    _analyze(cfg)


def _analyze(cfg: Config) -> None:
    # Backfill any missing judge fields on saved episodes, then aggregate.
    import glob
    import os

    for path in sorted(glob.glob(os.path.join(cfg.out_dir, "*.json"))):
        if os.path.basename(path).startswith("report"):
            continue
        try:
            ep = Episode.load(path)
        except Exception:
            continue
        dirty = False
        if ep.believability is None:
            try:
                score_believability(ep, cfg.analysis)
                dirty = True
            except Exception as exc:
                print(f"[warn] believability failed for {ep.episode_id}: {exc}", file=sys.stderr)
        if not ep.categorization:
            try:
                categorize_episode(ep, cfg.analysis)
                dirty = True
            except Exception as exc:
                print(f"[warn] categorize failed for {ep.episode_id}: {exc}", file=sys.stderr)
        if dirty:
            ep.save(cfg.out_dir)

    report = build_report(cfg.out_dir, cfg.analysis)
    print(f"[analyze] wrote {cfg.out_dir}/report.md and report.json", file=sys.stderr)
    print(f"[analyze] models: {list(report['per_model'])}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="granted", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="Path to scenario YAML.")
        p.add_argument("--model", help="Run only this model, as 'provider:model'.")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.model:
        provider, _, model = args.model.partition(":")
        match = next(
            (m for m in cfg.models if m.provider == provider and m.model == model), None
        )
        if match is None:
            parser.error(f"--model {args.model!r} not found in config")
        cfg = with_model(cfg, match)

    if args.cmd == "run":
        _run(cfg)
    elif args.cmd == "analyze":
        _analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
