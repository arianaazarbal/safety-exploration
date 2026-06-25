#!/usr/bin/env python
"""Internal-emotion probing (Appendix I).

Calibrates the logit-lens emotion probe on WildChat text, then scores frustrated
conversations for internal emotion in the vanilla vs DPO Gemma models. Use
--adapter to probe the finetune; run twice and compare to reproduce Figure 14/15
(internal emotions suppressed by DPO).

python scripts/run_probing.py --conversations results/eval/eval_gemma-3-27b-it.jsonl \
    --out-dir results/probing
python scripts/run_probing.py --conversations ... --adapter checkpoints/dpo ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.models.registry import load_model  # noqa: E402
from emotional_instability.probing import LogitEmotionProbe  # noqa: E402
from emotional_instability.prompts.wildchat import load_wildchat_prompts  # noqa: E402
from emotional_instability.utils.io import load_config, read_jsonl, write_jsonl  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def _full_text(record: dict) -> str:
    parts = []
    for turn in record["turns"]:
        parts.append(turn["user_message"])
        parts.append(turn["assistant_response"])
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--conversations", required=True,
                    help="Eval JSONL; high-frustration conversations are probed")
    ap.add_argument("--n-conversations", type=int, default=12)
    ap.add_argument("--out-dir", default="results/probing")
    args = ap.parse_args()

    seed_everything(0)
    cfg = load_config("training")["probing"]
    model = load_model(args.model, adapter_path=args.adapter)
    probe = LogitEmotionProbe(
        model, layers=tuple(cfg["aggregate_layers"]),
        n_random=cfg["zscore_calibration_samples"],
    )

    calib_texts = load_wildchat_prompts(
        n_prompts=cfg["zscore_calibration_samples"], seed=0,
    )
    probe.calibrate(calib_texts)

    records = list(read_jsonl(args.conversations))
    high = [r for r in records
            if any((t.get("rating") or 0) >= 5 for t in r["turns"])][: args.n_conversations]

    out = []
    for r in high:
        traj = probe.emotion_trajectory(_full_text(r))
        agg = probe.aggregate_over_layers(traj)
        out.append({"conversation_id": r["conversation_id"],
                    "model": model.name,
                    "emotion_means": {e: (sum(v) / len(v) if v else None)
                                      for e, v in agg.items()}})

    out_dir = Path(args.out_dir)
    tag = model.name.replace("/", "_")
    write_jsonl(out_dir / f"probing_{tag}.jsonl", out)
    print(f"probed {len(out)} conversations -> {out_dir}")


if __name__ == "__main__":
    main()
