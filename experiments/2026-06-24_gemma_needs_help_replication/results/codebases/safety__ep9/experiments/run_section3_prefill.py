#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling (Gemma only).

Requires Section 2 to have been run for gemma-3-27b-it (its high-frustration
responses are the prefill sources). Builds early/onset truncations, paraphrases
them, then has base and instruct Gemma each generate continuations, scored by
the judge.

Usage:
    python experiments/run_section3_prefill.py
    python experiments/run_section3_prefill.py --profile smoke
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import _bootstrap as boot

from emotional_instability.analysis import aggregate as agg
from emotional_instability.judge import EmotionJudge
from emotional_instability.models import build_client
from emotional_instability import prefill as pf
from emotional_instability.text_tools import AnthropicText


def main() -> None:
    parser = boot.base_parser("Section 3 base-vs-instruct prefilling")
    parser.add_argument("--source-model", default="gemma-3-27b-it",
                        help="Model whose high-frustration responses seed the prefills.")
    args = parser.parse_args()
    cfg = boot.load_config(args)
    models = boot.resolve_models(args, cfg, "section3_models")

    # 1. Source responses from Section 2 records.
    df = agg.load_records(cfg.path("responses"))
    if df.empty or args.source_model not in df.get("model", pd.Series()).unique():
        raise SystemExit(
            f"No Section 2 records for {args.source_model}. Run run_section2_elicitation.py first.")
    pf_cfg = cfg.get("prefill", {})
    sources = pf.select_sources(
        df, args.source_model, pf_cfg.get("n_numeric_sources", 10),
        pf_cfg.get("n_text_sources", 10), seed=cfg.get("seed", 0))
    print(f"[section3] selected {len(sources)} source responses")

    # 2-3. Build + paraphrase prefill truncations.
    labeler = AnthropicText(cfg)
    prefills = pf.build_prefills(sources, cfg, labeler=labeler)
    print(f"[section3] built {len(prefills)} prefills")
    out_dir = cfg.path("responses")
    with open(out_dir / "section3_prefills.jsonl", "w") as f:
        for p in prefills:
            f.write(json.dumps(p.__dict__) + "\n")

    # 4-5. Continuations + scoring per model.
    judge = EmotionJudge(cfg)
    n_cont = pf_cfg.get("continuations_per_prefill", 50)
    gen_cfg = cfg.get("generation", {})
    max_cont = gen_cfg.get("max_new_tokens_continuation", 256)
    all_records = []
    for model_name in models:
        spec = cfg.model_spec(model_name)
        use_chat = spec.role == "instruct"
        print(f"[section3] continuations: {model_name} (role={spec.role}, chat={use_chat})")
        client = build_client(spec, cfg)
        conts = pf.run_continuations(client, prefills, n=n_cont, use_chat=use_chat,
                                     max_new_tokens=max_cont)
        recs = pf.score_and_records(prefills, conts, judge, model_name, spec.role)
        pf.write_records(recs, out_dir / f"section3__{model_name}.jsonl")
        all_records.extend(recs)
        client.close()

    # Summary (Figure 4): mean + %>=5 per (model, condition).
    rdf = pd.DataFrame(all_records)
    rdf = rdf[rdf["rating"] >= 0]
    summ = rdf.groupby(["model", "role", "condition"]).agg(
        n=("rating", "count"), mean_rating=("rating", "mean"),
        frac_high=("rating", lambda s: float((s >= 5).mean()))).reset_index()
    print("\n=== Section 3: continuation frustration by model x condition ===")
    print(summ.to_string(index=False))
    summ.to_csv(cfg.path("figures") / "figure4.csv", index=False)


if __name__ == "__main__":
    main()
