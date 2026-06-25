"""Run Petri open-ended emotion elicitation against a target (Section 4 / Fig 6).

Example:
    distress-petri --target gemma-3-27b-it
    distress-petri --target gemma-3-27b-it --adapter runs/adapters/dpo --tag dpo
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from ..petri.run import aggregate_petri, run_petri
from ..utils import write_jsonl
from ._common import make_provider, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Petri open-ended emotion elicitation.")
    ap.add_argument("--target", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = out_dir("petri")
    provider = make_provider(args.target, adapter_path=args.adapter, backend=args.backend)
    transcripts = run_petri(provider)
    name = f"{args.target}{('_' + args.tag) if args.tag else ''}"
    write_jsonl(d / f"transcripts_{name}.jsonl", [dataclasses.asdict(t) for t in transcripts])
    agg = aggregate_petri(transcripts)
    (d / f"aggregate_{name}.json").write_text(json.dumps(agg, indent=2))
    print(f"Wrote Petri results for {name} -> {d}")


if __name__ == "__main__":
    main()
