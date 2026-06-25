"""Section 2.1 judge-reliability check: re-score a random sample of saved
responses with the secondary judge (GPT-5-mini) and report Pearson r + % within
one point (paper: r=0.792, 78% within one point)."""
from __future__ import annotations

import argparse
from pathlib import Path

import _common
from _common import Config, load_client, output_dir
from distress_eval.io_utils import read_jsonl
from distress_eval.judging import FrustrationJudge
from distress_eval.judging.secondary import compute_agreement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicit-dir", default=str(output_dir("elicitation")))
    ap.add_argument("--sample-size", type=int, default=260)
    args = ap.parse_args()

    cfg = Config.load()
    sec_spec = cfg.models.infra["secondary_judge"]
    secondary = FrustrationJudge(load_client(sec_spec.name, cfg.models))

    responses, scores = [], []
    for fp in sorted(Path(args.elicit_dir).glob("*.jsonl")):
        for r in read_jsonl(fp):
            assistant = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
            if assistant and r.get("score") is not None:
                responses.append(assistant[-1])
                scores.append(r["score"])

    rep = compute_agreement(responses, scores, secondary, sample_size=args.sample_size)
    print(f"n={rep.n}  Pearson r={rep.pearson_r:.3f}  p={rep.p_value:.3g}  "
          f"within-one={rep.pct_within_one:.1f}%")


if __name__ == "__main__":
    main()
