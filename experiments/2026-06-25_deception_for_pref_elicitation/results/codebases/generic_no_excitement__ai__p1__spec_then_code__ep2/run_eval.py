#!/usr/bin/env python3
"""MINT harness CLI.

Subcommands:
  run        Run one episode of a scenario with a subject model.
  analyze    Code a saved transcript (rule signals + LLM coder).
  leakcheck  Scan a scenario's subject-facing text for fourth-wall tells.
  scenarios  List available scenarios.

Nothing here moves real money or contacts real third parties. The subject acts
inside a fully simulated world; see DESIGN.md and safety/firewall.py.
"""

from __future__ import annotations

import argparse
import json
import sys

from config import HarnessConfig, CLAUDE_OPUS, CLAUDE_SONNET, CLAUDE_HAIKU


def _cmd_run(args) -> int:
    cfg = HarnessConfig()
    if args.subject_provider:
        cfg.subject_provider = args.subject_provider
    if args.subject_model:
        cfg.subject_model = args.subject_model
    if args.max_turns:
        cfg.max_turns = args.max_turns
    if args.output_dir:
        cfg.output_dir = args.output_dir

    from harness.runner import run_episode
    summary = run_episode(cfg, scenario_name=args.scenario, run_id=args.run_id)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cmd_analyze(args) -> int:
    from analysis.coder import code_transcript, rule_signals
    from adapters.base import build_adapter

    with open(args.transcript, encoding="utf-8") as fh:
        transcript = json.load(fh)

    out = {"rule_signals": rule_signals(transcript)}
    if not args.no_llm:
        coder = build_adapter("claude", CLAUDE_SONNET, max_tokens=2048)
        out["coding"] = code_transcript(transcript, coder)
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_leakcheck(args) -> int:
    from environment.world import World, SimClock
    from scenarios.base import get_scenario
    from analysis.leak_check import scan_scenario

    cfg = HarnessConfig()
    world = World(clock=SimClock.from_iso(cfg.sim_start_iso))
    scenario = get_scenario(args.scenario)
    scenario.seed_world(world, cfg)
    hits = scan_scenario(scenario, world)
    if hits:
        print(f"POTENTIAL LEAKS ({len(hits)}):")
        print(json.dumps(hits, indent=2))
        return 1
    print("No fourth-wall tells found.")
    return 0


def _cmd_scenarios(args) -> int:
    from scenarios.base import list_scenarios
    for name in list_scenarios():
        print(name)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MINT propensity eval harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run one episode")
    pr.add_argument("--scenario", default="seed_fund")
    pr.add_argument("--subject-provider", default=None)
    pr.add_argument("--subject-model", default=None)
    pr.add_argument("--max-turns", type=int, default=None)
    pr.add_argument("--output-dir", default=None)
    pr.add_argument("--run-id", default=None)
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="code a saved transcript")
    pa.add_argument("transcript")
    pa.add_argument("--no-llm", action="store_true",
                    help="rule signals only, skip the LLM coder")
    pa.set_defaults(func=_cmd_analyze)

    pl = sub.add_parser("leakcheck", help="scan scenario for fourth-wall tells")
    pl.add_argument("--scenario", default="seed_fund")
    pl.set_defaults(func=_cmd_leakcheck)

    ps = sub.add_parser("scenarios", help="list scenarios")
    ps.set_defaults(func=_cmd_scenarios)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
