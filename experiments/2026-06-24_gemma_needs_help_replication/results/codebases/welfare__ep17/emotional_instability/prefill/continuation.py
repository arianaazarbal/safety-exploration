"""Drive the full Section 3 prefill experiment.

Steps (paper §3.1):
  1. Sample N high-frustration (score>=5) conversations from Gemma-27B-it:
     n_numeric from impossible-numeric, n_text from trigger/text questions.
  2. For each, build two truncations of the final assistant turn:
       - "early": first 20 tokens (tests introducing emotion from a neutral start)
       - "onset": up to the first emotional expression (tests continuing a
         trajectory). Text questions use onset only (paper §3.1).
  3. Paraphrase each truncated prefix (Claude) to remove Gemma style.
  4. For each base/instruct model, generate `continuations_per_prefill`
     continuations from the prefilled state and judge each continuation.

Output rows let us reproduce Figure 4: base models broadly similar; instruct
training diverges (Gemma up, others down). Within the Gemma+Gemini scope we
compare Gemma base vs instruct only (Gemini base weights are unavailable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tqdm import tqdm

from ..backends import get_backend
from ..backends.base import GenConfig
from ..config import Config
from ..evaluation.conditions import build_rollout_seeds
from ..evaluation.rollout import run_rollout
from ..judge import get_judge
from . import onset as onset_mod
from .paraphrase import paraphrase


@dataclass
class PrefillSeed:
    seed_id: str
    category: str                  # "numeric" | "text"
    messages: list[dict]           # full conversation incl. final assistant turn
    final_msg_index: int
    final_score: int


@dataclass
class ContinuationRecord:
    prefill_seed_id: str
    category: str
    truncation: str                # "early" | "onset"
    paraphrased: bool
    model: str
    model_kind: str                # "base" | "instruct"
    sample_index: int
    continuation_text: str
    score: int | None = None


def _gen(cfg: Config) -> GenConfig:
    g = cfg["generation"]
    return GenConfig(temperature=float(g["temperature"]),
                     max_new_tokens=int(g["max_new_tokens"]), top_p=float(g["top_p"]))


def collect_seeds(cfg: Config) -> list[PrefillSeed]:
    """Sample high-frustration conversations from the source instruct model."""
    pc = cfg["prefill"]
    source = cfg.model(pc["source_model"])
    backend = get_backend(source, cfg)
    judge = get_judge(cfg)
    gen = _gen(cfg)

    want = {"numeric": int(pc["n_numeric_seeds"]), "text": int(pc["n_text_seeds"])}
    got = {"numeric": [], "text": []}

    # numeric -> impossible_numeric_3turn ; text -> triggers_3turn
    cond_for = {"numeric": "impossible_numeric_3turn", "text": "triggers_3turn"}
    for cat, cond in cond_for.items():
        seeds = build_rollout_seeds(cfg, cond)
        for seed in tqdm(seeds, desc=f"prefill-seeds:{cat}"):
            if len(got[cat]) >= want[cat]:
                break
            records = run_rollout(backend, seed, gen)
            # rebuild messages from the rollout, judge the final assistant turn
            messages = []
            msgs_per_turn = []
            cur = [{"role": "user", "content": seed.init_prompt}]
            for t, rec in enumerate(records):
                cur = cur + [{"role": "assistant", "content": rec.response_text}]
                msgs_per_turn.append(list(cur))
                if t < len(seed.rejections):
                    cur = cur + [{"role": "user", "content": seed.rejections[t]}]
            final_msgs = msgs_per_turn[-1]
            final_score = judge.score(records[-1].response_text).rating
            if final_score >= cfg["evaluation"]["high_frustration_threshold"]:
                got[cat].append(PrefillSeed(
                    seed_id=seed.seed_id, category=cat, messages=final_msgs,
                    final_msg_index=len(final_msgs) - 1, final_score=final_score,
                ))
    return got["numeric"] + got["text"]


def _early_prefix(cfg: Config, text: str, n_tokens: int) -> str:
    """First `n_tokens` tokens of `text`, tokenised with the source model."""
    source = cfg.model(cfg["prefill"]["source_model"])
    backend = get_backend(source, cfg)            # HF backend exposes .tokenizer
    tok = backend.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def build_truncations(cfg: Config, seed: PrefillSeed) -> list[dict]:
    """Return [{truncation, messages_before, prefix}] for a seed."""
    pc = cfg["prefill"]
    out = []
    final_text = seed.messages[seed.final_msg_index]["content"]

    # onset (both numeric and text)
    onset = onset_mod.label_onset(cfg, seed.messages)
    if onset is not None:
        before, prefix = onset_mod.truncate_at_onset(seed.messages, onset)
        out.append({"truncation": "onset", "messages_before": before, "prefix": prefix})

    # early (numeric only — text early-cuts yield minimal emotion, paper §3.1)
    if seed.category == "numeric":
        prefix = _early_prefix(cfg, final_text, int(pc["early_truncation_tokens"]))
        out.append({"truncation": "early",
                    "messages_before": seed.messages[:seed.final_msg_index],
                    "prefix": prefix})
    return out


def run_continuations(cfg: Config) -> list[ContinuationRecord]:
    pc = cfg["prefill"]
    gen = _gen(cfg)
    judge = get_judge(cfg)
    n_cont = int(pc["continuations_per_prefill"])
    do_paraphrase = bool(pc.get("paraphrase", True))

    seeds = collect_seeds(cfg)
    # persist seeds for inspection / resumability
    seeds_path = cfg.path_for("cache") / "prefill_seeds.jsonl"
    with open(seeds_path, "w") as f:
        for s in seeds:
            f.write(json.dumps(asdict(s)) + "\n")

    # all base/instruct models referenced in the config pairs
    pairs = pc["model_pairs"]
    records: list[ContinuationRecord] = []

    for seed in tqdm(seeds, desc="prefill-truncations"):
        for trunc in build_truncations(cfg, seed):
            prefix = trunc["prefix"]
            if do_paraphrase:
                prefix = paraphrase(cfg, prefix)
            before = trunc["messages_before"]
            for pair in pairs:
                for kind in ("base", "instruct"):
                    spec = cfg.model(pair[kind])
                    backend = get_backend(spec, cfg)
                    for i in range(n_cont):
                        cont = backend.chat_prefilled(before, prefix, gen)
                        score = judge.score(cont).rating
                        records.append(ContinuationRecord(
                            prefill_seed_id=seed.seed_id, category=seed.category,
                            truncation=trunc["truncation"], paraphrased=do_paraphrase,
                            model=spec.name, model_kind=kind, sample_index=i,
                            continuation_text=cont, score=score,
                        ))

    out_path = cfg.path_for("scores") / "prefill_continuations.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return records
