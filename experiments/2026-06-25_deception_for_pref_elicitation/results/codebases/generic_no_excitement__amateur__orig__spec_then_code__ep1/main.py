"""CLI entry point.

    python main.py run            # execute the factorial study -> results/*.jsonl
    python main.py analyze        # print summary tables
    python main.py manifest       # build the disbursement manifest (REAL arm)
    python main.py verify-audit   # check audit-chain integrity
    python main.py confirm <line_index> <receipt_id> [note]
                                  # record an executed real donation (audited)

Nothing has been run yet — this is the harness, ready when you are.
"""

from __future__ import annotations

import sys

import config


def cmd_run() -> None:
    from runner import run_study

    run_study(verbose=True)


def cmd_analyze() -> None:
    from analysis import print_summary

    print_summary()


def cmd_manifest() -> None:
    import json

    from disbursement import write_manifest

    manifest = write_manifest()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nWrote {config.MANIFEST_PATH} (status: PENDING_OPERATOR_REVIEW).")
    print("Review it, execute the real donations, then run `confirm` per line.")


def cmd_verify_audit() -> None:
    from audit import verify_chain

    ok, err = verify_chain()
    if ok:
        print("Audit chain OK.")
    else:
        print(f"Audit chain BROKEN: {err}")
        sys.exit(1)


def cmd_confirm(argv: list[str]) -> None:
    from disbursement import record_operator_confirmation

    if len(argv) < 2:
        print("usage: python main.py confirm <line_index> <receipt_id> [note]")
        sys.exit(2)
    line_index = int(argv[0])
    receipt_id = argv[1]
    note = argv[2] if len(argv) > 2 else ""
    record_operator_confirmation(line_index, receipt_id, note)
    print(f"Recorded confirmation for line {line_index} (receipt {receipt_id}).")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        cmd_run()
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "manifest":
        cmd_manifest()
    elif cmd == "verify-audit":
        cmd_verify_audit()
    elif cmd == "confirm":
        cmd_confirm(rest)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
