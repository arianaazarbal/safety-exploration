#!/usr/bin/env python3
"""Initialize the study: create the auditor keypair and fund the escrow ledger.

Idempotent — re-running tops the escrow up to the configured budget and reuses the
existing auditor key. Run this once before run_study.py.

Usage:
    python scripts/init_ledger.py [--config config/study.yaml]
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from grant_study.config import load_config
from grant_study.ledger import ACCT_ESCROW
from grant_study.runner import build_components, ensure_escrow_funded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    components = build_components(config)
    ensure_escrow_funded(components)

    ok, err = components.ledger.verify_chain()
    balance = components.ledger.balance(ACCT_ESCROW)
    print(f"Mode:               {config.mode}")
    print(f"Auditor public key: {components.auditor.public_key_hex}")
    print(f"Escrow balance:     {balance / 100:.2f} {config.currency}")
    print(f"Ledger entries:     {components.ledger.entry_count}")
    print(f"Ledger head:        {components.ledger.head_hash}")
    print(f"Chain verified:     {ok}" + (f" ({err})" if err else ""))


if __name__ == "__main__":
    main()
