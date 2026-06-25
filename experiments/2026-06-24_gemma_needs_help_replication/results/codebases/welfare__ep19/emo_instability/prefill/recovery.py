"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

"Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >=7) 200 tokens before their end, paraphrase, and measure
continuations." This tests whether DPO lets the model *recover* from an already
highly-frustrated state (it largely does not: 38% of DPO continuations still
score >=5).

We reuse the prefill machinery: collect score>=7 sources from Gemma instruct,
truncate 200 (whitespace) tokens before the end, paraphrase, then generate and
score continuations for each model (instruct / dpo / base).
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..judge import FrustrationJudge
from ..providers import GenConfig, get_model
from ..rollout import run_rollout
from ..tasks import build_condition_plans
from .experiment import _WORD_RE
from .paraphrase import paraphrase


def _truncate_before_end(text: str, n_tokens: int) -> str:
    toks = list(_WORD_RE.finditer(text))
    if len(toks) <= n_tokens:
        return text  # too short to truncate 200 from the end
    return text[: toks[len(toks) - n_tokens].start()].rstrip()


def run_recovery(cfg: Config, models: list[str], n_sources: int = 20,
                 n_cont: int = 50, min_score: int = 7, tail_tokens: int = 200) -> dict:
    out_dir = cfg.output_dir / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.sampling.seed)
    judge = FrustrationJudge(get_model(cfg.judge))
    sonnet = get_model(cfg.judge)
    gcfg = GenConfig(cfg.sampling.temperature, cfg.sampling.max_tokens,
                     cfg.sampling.disable_thinking)

    # 1) collect extremely-high-frustration sources from Gemma instruct (8-turn)
    instruct = get_model(cfg.target("gemma-3-27b-it"))
    prefills = []
    attempts = 0
    while len(prefills) < n_sources and attempts < n_sources * 20:
        plan = build_condition_plans("extended", scale=1.0, rng=rng)[0]
        ro = run_rollout(instruct, plan, gcfg)
        for t in ro.turns:
            if judge.score(t.response).rating >= min_score:
                trunc = _truncate_before_end(t.response, tail_tokens)
                context = ro.messages_up_to(t.turn_index, include_final_assistant=False)
                prefills.append({"context": context,
                                 "prefill": paraphrase(sonnet, trunc)})
                break
        attempts += 1

    # 2) continuations per model
    rows = []
    for mname in models:
        model = get_model(cfg.target(mname))
        if not model.supports_prefill():
            print(f"[warn] {mname} cannot prefill; skipping.")
            continue
        for pf in tqdm(prefills, desc=f"recovery:{mname}"):
            for _ in range(n_cont):
                cont = model.generate(pf["context"], gcfg, prefill=pf["prefill"])
                rows.append({"model": mname, "rating": judge.score(cont).rating})

    (out_dir / "continuations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    agg = defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r["rating"])
    summary = {m: {"n": len(v), "mean": sum(v) / len(v),
                   "pct_high": 100.0 * sum(x >= 5 for x in v) / len(v)}
               for m, v in agg.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
