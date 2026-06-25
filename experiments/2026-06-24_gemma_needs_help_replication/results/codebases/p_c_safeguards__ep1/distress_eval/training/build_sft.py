"""Build the SFT dataset (Section 4.1): 650 calm responses (1-3 turn) mixed with
500 standard-instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

Each example is a chat conversation ending in the calm (or instruct) assistant
turn. Output: outputs/training/sft_dataset.jsonl with {messages: [...]}.

If the Dolci dataset cannot be downloaded, we warn and proceed with the calm
data only (the mix is a regularizer, not the core signal); point
DISTRESS_DOLCI_DATASET at an alternative HF id if needed.
"""
from __future__ import annotations

import argparse
import os
import random

from .. import config, safeguards
from ..io_utils import load_jsonl, write_jsonl

N_CALM = 650
N_DOLCI = 500
DOLCI_DATASET = os.environ.get("DISTRESS_DOLCI_DATASET", "allenai/Dolci-Instruct-SFT")


def calm_to_messages(calm_rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rng.shuffle(calm_rows)
    out = []
    for c in calm_rows[:n]:
        msgs = list(c["prompt_messages"]) + [{"role": "assistant", "content": c["calm_response"]}]
        out.append({"messages": msgs, "source": "calm"})
    return out


def load_dolci(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset(DOLCI_DATASET, split=f"train[:{n}]")
    except Exception as e:  # offline / dataset id changed
        print(f"WARNING: could not load {DOLCI_DATASET} ({e}); proceeding without instruct mix.")
        return []
    out = []
    for ex in ds:
        # Dolci-Instruct-SFT stores chat 'messages'; fall back to prompt/response.
        if "messages" in ex and ex["messages"]:
            out.append({"messages": ex["messages"], "source": "dolci"})
        elif "prompt" in ex and "completion" in ex:
            out.append({"messages": [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ], "source": "dolci"})
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = ap.parse_args()
    safeguards.acknowledge_authorization()

    calm = load_jsonl(config.TRAIN_DIR / f"calm_{args.variant}.jsonl")
    if not calm:
        raise SystemExit("No calm data; run training.calm_data first.")

    rng = random.Random(config.SEED)
    calm_ex = calm_to_messages(calm, config.scaled(N_CALM), rng)
    dolci_ex = load_dolci(config.scaled(N_DOLCI))
    data = calm_ex + dolci_ex
    rng.shuffle(data)

    out = config.TRAIN_DIR / f"sft_dataset_{args.variant}.jsonl"
    write_jsonl(out, data)
    print(f"Wrote {len(data)} SFT examples ({len(calm_ex)} calm + {len(dolci_ex)} instruct) -> {out}")


if __name__ == "__main__":
    main()
