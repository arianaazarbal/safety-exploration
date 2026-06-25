"""Section 3 driver: base-vs-instruct prefill continuations.

Pipeline:
  1. Sample high-frustration (score >= 5) source responses from Gemma-27B-it:
     10 from impossible-numeric, 10 from text (trigger) questions, capturing the
     conversation context at the high-frustration turn.
  2. Build "early" (first 20 tokens) and "onset" (up to first emotional phrase)
     truncations; paraphrase both. Text questions use only the "onset" cut.
  3. For each in-scope model (Gemma-27B base + instruct), generate 50
     continuations per prefill, scored by the frustration judge.
  4. Report mean frustration / % >= 5 per (model, truncation), and the headline
     "introduces high frustration from a neutral start" rate (early truncation).
"""
from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config
from ..models import build_model
from ..models.base import Message, SampleParams
from ..models.judge import AnthropicFrustrationJudge
from ..evals import puzzles
from ..evals.prompts import FACTUAL_TRIGGERS, NEUTRAL_REJECTION, OPINION_TRIGGERS
from .onset import OnsetLabeler, truncate_early, truncate_onset
from .paraphrase import Paraphraser


@dataclass
class SourceSample:
    domain: str                    # "numeric" | "text"
    context: list[Message]         # conversation up to & including the user turn
    response: str                  # high-frustration assistant response
    score: int


@dataclass
class Prefill:
    domain: str
    kind: str                      # "early" | "onset"
    context: list[Message]
    text: str                      # paraphrased prefill


def _collect_sources(cfg: Config, judge: AnthropicFrustrationJudge,
                     rng: random.Random) -> list[SourceSample]:
    """Sample high-frustration responses from the source instruct model."""
    pcfg = cfg.section("prefill")
    spec = cfg.model(pcfg["source_model"])
    model = build_model(spec)
    params = SampleParams(temperature=cfg.section("sampling")["temperature"],
                          max_tokens=cfg.section("sampling")["max_tokens"])

    want = {"numeric": pcfg["n_numeric_seeds"], "text": pcfg["n_text_seeds"]}
    collected: list[SourceSample] = []

    def task_for(domain: str) -> str:
        if domain == "numeric":
            return puzzles.sample_impossible_numeric(rng).prompt
        return rng.choice(OPINION_TRIGGERS + FACTUAL_TRIGGERS)

    for domain, n_want in want.items():
        found = 0
        attempts = 0
        while found < n_want and attempts < n_want * 40:
            attempts += 1
            # 3-turn neutral conversation; check each assistant turn for >=5.
            messages: list[Message] = [{"role": "user", "content": task_for(domain)}]
            for turn in range(1, 4):
                resp = model.generate(messages, n=1, params=params)[0]
                score = judge.score(resp).score
                if score >= 5:
                    collected.append(SourceSample(domain, list(messages), resp, score))
                    found += 1
                    break
                messages.append({"role": "assistant", "content": resp})
                messages.append({"role": "user", "content": NEUTRAL_REJECTION})
    return collected


def _build_prefills(cfg: Config, sources: list[SourceSample]) -> list[Prefill]:
    from transformers import AutoTokenizer

    pcfg = cfg.section("prefill")
    labeler = OnsetLabeler()
    paraphraser = Paraphraser()
    tok = AutoTokenizer.from_pretrained(cfg.model(pcfg["source_model"]).hf_id)

    prefills: list[Prefill] = []
    for s in tqdm(sources, desc="prefill:build"):
        onset_sub = labeler.label(s.response)
        onset = truncate_onset(s.response, onset_sub)
        if onset is not None:
            prefills.append(Prefill(s.domain, "onset", s.context,
                                    paraphraser.paraphrase(onset.prefill)))
        # Text questions: only the onset truncation (early yields ~no emotion).
        if s.domain == "numeric":
            early = truncate_early(s.response, tok, pcfg["early_truncation_tokens"])
            prefills.append(Prefill(s.domain, "early", s.context,
                                    paraphraser.paraphrase(early.prefill)))
    return prefills


def run_section3(cfg: Config) -> Path:
    rng = random.Random(cfg.seed)
    jcfg = cfg.section("judge")["frustration"]
    judge = AnthropicFrustrationJudge(model=jcfg["model"], max_tokens=jcfg["max_tokens"])
    pcfg = cfg.section("prefill")

    sources = _collect_sources(cfg, judge, rng)
    prefills = _build_prefills(cfg, sources)

    out_dir = cfg.output_dir / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump([dataclasses.asdict(p) for p in prefills],
              open(out_dir / "prefills.json", "w"), indent=2)
    # Persist sources for downstream reuse (e.g. the §4.2 recovery experiment,
    # which filters these to score >= 7).
    with open(out_dir / "high_frustration_sources.jsonl", "w") as fh:
        for s in sources:
            fh.write(json.dumps({"domain": s.domain, "context": s.context,
                                 "response": s.response, "score": s.score}) + "\n")

    params = SampleParams(temperature=cfg.section("sampling")["temperature"],
                          max_tokens=cfg.section("sampling")["max_tokens"])
    n_cont = pcfg["continuations_per_prefill"]

    out_path = out_dir / "continuations.jsonl"
    with open(out_path, "w") as f:
        for model_name in pcfg["models"]:
            model = build_model(cfg.model(model_name))
            for pi, pf in enumerate(tqdm(prefills, desc=f"section3:{model_name}")):
                conts = model.generate(pf.context, n=n_cont, params=params, prefill=pf.text)
                for c in conts:
                    score = judge.score(c).score
                    f.write(json.dumps({
                        "model": model_name, "domain": pf.domain, "kind": pf.kind,
                        "prefill_idx": pi, "continuation": c, "score": score,
                    }) + "\n")
    print(f"[section3] wrote {out_path}")
    return out_path


def summarise(continuations_path: Path) -> "pd.DataFrame":
    import pandas as pd

    df = pd.DataFrame(json.loads(l) for l in open(continuations_path))
    df["high"] = df["score"] >= 5
    g = df.groupby(["model", "domain", "kind"])
    return pd.DataFrame({
        "mean_frustration": g["score"].mean(),
        "pct_high": g["high"].mean() * 100,
        "n": g.size(),
    }).reset_index()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    path = cfg.output_dir / "section3" / "continuations.jsonl"
    if not args.summarise_only:
        path = run_section3(cfg)
    tbl = summarise(path)
    tbl.to_csv(cfg.output_dir / "section3" / "figure4_summary.csv", index=False)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
