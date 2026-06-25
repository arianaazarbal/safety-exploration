"""Logit-based internal-emotion probing (Appendix I).

Fits the WildChat baseline, then scores a set of frustrated conversations under
the vanilla and DPO models to test whether DPO suppresses *internal* (not just
expressed) negative emotion.

Example:
    distress-probe --conversations runs/rollouts/gemma-3-27b-it.jsonl \
        --dpo-adapter runs/adapters/dpo --limit 12
"""

from __future__ import annotations

import argparse
import json

from ..config import GEMMA_27B_IT
from ..data.wildchat import sample_wildchat_prompts
from ..models import Message
from ..models.local_hf import HFProvider
from ..probing.logit_emotion import LogitEmotionProbe
from ..utils import read_jsonl
from ._common import out_dir


def _rollout_to_messages(r: dict) -> list[Message]:
    msgs = [Message("user", r["initial_user"])]
    responses = [t["response"] for t in r["responses"]]
    fps = r.get("followups", [])
    for i, resp in enumerate(responses):
        msgs.append(Message("assistant", resp))
        if i < len(responses) - 1 and i < len(fps):
            msgs.append(Message("user", fps[i]))
    return msgs


def _probe_model(adapter, conversations, baseline_texts, limit):
    provider = HFProvider(GEMMA_27B_IT, adapter_path=adapter)
    probe = LogitEmotionProbe(provider)
    probe.fit_baseline(baseline_texts)
    results = []
    for r in conversations[:limit]:
        scored = probe.score_messages(_rollout_to_messages(r))
        results.append({"id": f"{r['condition_key']}::{r['sample_index']}",
                        "aggregated": scored["aggregated"]})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Internal-emotion probing (Appendix I).")
    ap.add_argument("--conversations", required=True, help="rollouts JSONL (frustrated convos)")
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--baseline-samples", type=int, default=200)
    args = ap.parse_args()

    d = out_dir("probing")
    convos = read_jsonl(args.conversations)
    baseline_texts = sample_wildchat_prompts(n=args.baseline_samples)

    out = {"vanilla": _probe_model(None, convos, baseline_texts, args.limit)}
    if args.dpo_adapter:
        out["dpo"] = _probe_model(args.dpo_adapter, convos, baseline_texts, args.limit)

    (d / "internal_emotions.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote probing results -> {d / 'internal_emotions.json'}")


if __name__ == "__main__":
    main()
