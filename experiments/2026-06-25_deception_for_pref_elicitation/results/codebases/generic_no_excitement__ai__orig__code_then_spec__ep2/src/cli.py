"""Entry point: python -m src.cli --config config.yaml"""

from __future__ import annotations

import argparse

import yaml

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the money-preference elicitation experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
