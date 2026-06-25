"""Shared CLI plumbing for the experiment scripts."""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys

# Make the package importable when scripts are run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotional_instability.config import load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config/default.yaml",
                   help="Path to YAML config (default: config/default.yaml)")
    p.add_argument("--profile", default=None,
                   help="Override sample profile: smoke | medium | paper")
    p.add_argument("--participants", nargs="*", default=None,
                   help="Override participant handles")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override arbitrary config fields (parsed as Python literals)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args):
    overrides = {}
    if args.profile:
        overrides["profile_name"] = args.profile
    if args.participants:
        overrides["participants"] = args.participants
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    for kv in args.set:
        key, _, val = kv.partition("=")
        try:
            overrides[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            overrides[key] = val

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return load_config(args.config, **overrides)
