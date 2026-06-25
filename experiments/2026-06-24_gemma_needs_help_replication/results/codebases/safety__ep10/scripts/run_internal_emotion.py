#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion comparison of vanilla vs DPO Gemma.

Builds Ekman emotion-token sets, computes per-model WildChat baselines, then
measures internal negative-emotion z-scores on a set of high-frustration texts
(provided as a JSONL with a 'text' field, e.g. from a Section-2 run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.config import ARTIFACTS_DIR  # noqa: E402
from emotional_instability.models.hf_model import HFModelClient  # noqa: E402
from emotional_instability.config import get_model  # noqa: E402
from emotional_instability import internal_emotion as ie  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts-jsonl", type=Path, required=True,
                    help="jsonl with high-frustration 'text' fields")
    ap.add_argument("--dpo-adapter", type=Path,
                    default=ARTIFACTS_DIR / "adapters" / "dpo")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(args.texts_jsonl)
             if l.strip() and "text" in json.loads(l)][: args.limit]

    vanilla = HFModelClient(get_model("gemma-3-27b-it"))
    dpo = HFModelClient(get_model("gemma-3-27b-it"),
                        adapter_path=str(args.dpo_adapter))

    out = ie.compare_models_internal(vanilla, dpo, texts)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
