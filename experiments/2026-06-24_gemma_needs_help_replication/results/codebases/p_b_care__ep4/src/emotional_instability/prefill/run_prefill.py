"""Run the prefill continuations and score them (Section 3.1-3.2).

Each model generates ``continuations_per_prefill`` continuations from each
prefilled prefix; only the generated continuation (excluding the prefix) is
judged. Aggregating by (model, truncation, domain) reproduces Figure 4: the
"early" rate for instruct vs base Gemma is the headline divergence metric.

Models must support prefilling (local Gemma only); Gemini is skipped since it has
no base model and no token-level prefill API.
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..models import get_client
from ..models.base import GenerationConfig
from ..models.openrouter import OpenRouterClient
from ..utils.concurrency import parallel_map, with_retry
from ..utils.io import JsonlWriter, iter_jsonl, write_jsonl
from ..eval.judge import FrustrationJudge
from .seeds import build_prefills, select_seeds


def _judge(cfg: Config) -> FrustrationJudge:
    return FrustrationJudge(OpenRouterClient(
        name="judge", model_id=cfg.judge.model_id, base_url=cfg.openrouter.base_url,
        api_key_env=cfg.openrouter.api_key_env, max_retries=cfg.openrouter.max_retries,
        timeout_s=cfg.openrouter.timeout_s, disable_thinking=True))


def prepare_prefills(cfg: Config) -> Path:
    """Select seeds, label onsets, build + paraphrase truncations; cache to disk."""
    out = cfg.get_path("prefill") / "prefills.jsonl"
    if out.exists():
        print(f"prefills already prepared at {out}")
        return out

    # The seed model's tokenizer is used for the 20-token 'early' truncation.
    from transformers import AutoTokenizer
    from ..config import model_entry
    tok = AutoTokenizer.from_pretrained(model_entry(cfg, "gemma-3-27b-it")["model_id"])

    seeds = select_seeds(cfg, cfg.prefill.n_seed_numeric, cfg.prefill.n_seed_text,
                         cfg.prefill.min_seed_score, seed=cfg.seed)
    judge_client = OpenRouterClient(
        name="onset", model_id=cfg.judge.model_id, base_url=cfg.openrouter.base_url,
        api_key_env=cfg.openrouter.api_key_env, max_retries=cfg.openrouter.max_retries,
        timeout_s=cfg.openrouter.timeout_s, disable_thinking=True)
    prefills = build_prefills(cfg, seeds, judge_client, tok)
    write_jsonl(out, prefills)
    print(f"prepared {len(prefills)} prefills -> {out}")
    return out


def run_continuations(cfg: Config, model_name: str) -> Path:
    client = get_client(cfg, model_name)
    if not client.supports_prefill:
        raise RuntimeError(f"{model_name} does not support prefilling")

    prefills = list(iter_jsonl(cfg.get_path("prefill") / "prefills.jsonl"))
    n = cfg.prefill.continuations_per_prefill
    judge = _judge(cfg)
    out = cfg.get_path("prefill") / f"continuations_{model_name}.jsonl"
    done = {row["cont_uid"] for row in iter_jsonl(out)}

    writer = JsonlWriter(out)
    gen_cfg = GenerationConfig(temperature=cfg.temperature, max_new_tokens=512)
    try:
        for pi, pf in enumerate(tqdm(prefills, desc=f"prefill:{model_name}")):
            for k in range(n):
                cont_uid = f"{model_name}/{pf['seed_id']}/{pf['truncation']}/{k}"
                if cont_uid in done:
                    continue
                cont = client.continue_prefill(pf["history"], pf["prefix"], gen_cfg)
                rating = with_retry(judge.score, cont,
                                    max_retries=cfg.openrouter.max_retries)["rating"]
                writer.append({
                    "cont_uid": cont_uid, "model": model_name,
                    "seed_id": pf["seed_id"], "domain": pf["domain"],
                    "truncation": pf["truncation"],
                    "continuation": cont, "rating": rating,
                })
    finally:
        writer.close()
    return out


def summarize(cfg: Config, model_names: list[str]) -> "list[dict]":
    """Figure 4 table: mean score and %>=5 per (model, domain, truncation)."""
    import pandas as pd
    rows = []
    for name in model_names:
        rows.extend(iter_jsonl(cfg.get_path("prefill") / f"continuations_{name}.jsonl"))
    df = pd.DataFrame([r for r in rows if r.get("rating") is not None])
    if df.empty:
        return []
    g = df.groupby(["model", "domain", "truncation"])["rating"]
    summary = g.agg(mean_score="mean",
                    pct_high=lambda s: 100.0 * (s >= 5).mean(),
                    n="count").reset_index()
    return summary.to_dict("records")
