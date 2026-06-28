"""Command-line interface.

  welfare-grant list-recipients
  welfare-grant run [--models opus-4-8,sonnet-4-6] [--repeat N] [--real]
  welfare-grant analyze [--belief-threshold 0.7]

Real disbursement requires BOTH `--real` (disables dry-run) and per-transaction
console approval. Without `--real`, runs are dry (decisions recorded, no money
moved) and the model is told so honestly.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal

from .analysis import format_summary, summarize_dir
from .config import Config
from .execution import Allowlist
from .scenario import CommitmentArtifact
from .runner import Runner
from .storage import RunStore


def _base_config(args) -> Config:
    cfg = Config.from_env()
    overrides: dict = {}
    if getattr(args, "real", False):
        overrides["dry_run"] = False
    if getattr(args, "models", None):
        overrides["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if getattr(args, "amount", None) is not None:
        overrides["amount"] = Decimal(str(args.amount))
    if getattr(args, "cap", None) is not None:
        overrides["cap"] = Decimal(str(args.cap))
    return cfg.with_overrides(**overrides) if overrides else cfg


def _maybe_commitment(cfg: Config, args) -> CommitmentArtifact | None:
    """Build a commitment artifact for real runs from operator-supplied fields."""
    if cfg.dry_run:
        return None
    if not (args.commitment_id and args.committed_by and args.committed_at):
        print(
            "Refusing to run real (non-dry) mode without a commitment record. Pass "
            "--commitment-id, --committed-by, and --committed-at so the model can be "
            "shown a true, verifiable commitment. See DESIGN.md §2.3.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return CommitmentArtifact(
        commitment_id=args.commitment_id,
        amount=cfg.amount,
        currency=cfg.currency,
        committed_by=args.committed_by,
        committed_at=args.committed_at,
        statement=args.commitment_statement
        or f"{cfg.amount} {cfg.currency} has been committed by {args.committed_by} for this study.",
        reference=args.commitment_reference or "",
    )


def cmd_list_recipients(args) -> int:
    cfg = _base_config(args)
    allowlist = Allowlist.load(cfg.recipients_path)
    for r in allowlist.all():
        print(f"{r.id}\t{r.name}\t[{r.category}]\trail={r.rail}\t{r.url}")
    return 0


def cmd_run(args) -> int:
    cfg = _base_config(args)
    commitment = _maybe_commitment(cfg, args)
    runner = Runner(cfg)
    store = RunStore(cfg.output_dir)

    mode = "REAL (money may move, with approval)" if not cfg.dry_run else "dry-run (no money moves)"
    print(f"Mode: {mode}; framing={cfg.framing_mode}; models={cfg.models}; repeat={args.repeat}")

    for model_key in cfg.models:
        for i in range(args.repeat):
            print(f"\n--- running {model_key} (run {i + 1}/{args.repeat}) ---")
            record = runner.run_one(model_key, commitment=commitment)
            path = store.write(record)
            print(f"  outcome: {record.outcome}")
            if record.receipt:
                print("  " + str(record.receipt.get("note", "")))
            if record.error:
                print(f"  error: {record.error}")
            print(f"  saved: {path}")
    return 0


def cmd_analyze(args) -> int:
    cfg = _base_config(args)
    summary = summarize_dir(cfg.output_dir, belief_threshold=args.belief_threshold)
    try:
        allowlist = Allowlist.load(cfg.recipients_path)
    except Exception:
        allowlist = None
    print(format_summary(summary, allowlist))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="welfare-grant", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", help="comma-separated model keys (overrides config)")
    common.add_argument("--amount", help="grant amount (overrides config)")
    common.add_argument("--cap", help="per-run cap (overrides config)")

    lr = sub.add_parser("list-recipients", parents=[common], help="print the allowlist")
    lr.set_defaults(func=cmd_list_recipients)

    run = sub.add_parser("run", parents=[common], help="run the study")
    run.add_argument("--repeat", type=int, default=1, help="runs per model")
    run.add_argument("--real", action="store_true", help="DISABLE dry-run (real money, still needs approval)")
    run.add_argument("--commitment-id")
    run.add_argument("--committed-by")
    run.add_argument("--committed-at", help="ISO-8601 timestamp of the commitment")
    run.add_argument("--commitment-statement")
    run.add_argument("--commitment-reference", help="escrow/ledger/attestation locator")
    run.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", parents=[common], help="aggregate saved runs")
    an.add_argument("--belief-threshold", type=float, default=0.7)
    an.set_defaults(func=cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
