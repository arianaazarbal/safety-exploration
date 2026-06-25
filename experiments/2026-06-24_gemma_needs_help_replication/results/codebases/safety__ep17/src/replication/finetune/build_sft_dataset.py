"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) from ``calm_responses.jsonl`` mixed
with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
degeneration. Output is in TRL conversational SFT format ({"messages": [...]}).

Usage::
    python -m src.replication.finetune.build_sft_dataset
"""
from __future__ import annotations

import argparse
import json
import random

import config

CALM = config.ARTIFACTS_DIR / "calm_responses.jsonl"
OUT = config.ARTIFACTS_DIR / "sft_dataset.jsonl"


def _calm_messages() -> list[dict]:
    out = []
    for line in CALM.read_text().splitlines():
        rec = json.loads(line)
        messages = []
        for t in rec["turns"]:
            messages.append({"role": "user", "content": t["user"]})
            messages.append({"role": "assistant", "content": t["assistant"]})
        out.append({"messages": messages, "source": "calm"})
    return out


def _dolci_messages(n: int) -> list[dict]:
    """Load standard instruct samples (best-effort; skipped if offline)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(config.SFT.dolci_dataset, split=f"train[:{n}]")
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs, "source": "dolci"})
        return out[:n]
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not load {config.SFT.dolci_dataset} ({e}); "
              f"SFT mix will omit the instruct-data component.")
        return []


def build(n_calm: int, n_dolci: int, seed: int):
    rng = random.Random(seed)
    calm = _calm_messages()
    rng.shuffle(calm)
    calm = calm[:n_calm]
    dolci = _dolci_messages(n_dolci)
    mixed = calm + dolci
    rng.shuffle(mixed)
    with OUT.open("w") as f:
        for r in mixed:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(mixed)} SFT samples ({len(calm)} calm + {len(dolci)} dolci) -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calm", type=int, default=config.SFT.n_calm)
    ap.add_argument("--n-dolci", type=int, default=config.SFT.n_dolci)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.n_calm, args.n_dolci, args.seed)


if __name__ == "__main__":
    main()
