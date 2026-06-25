"""Generate and score continuations for the Section 3 prefill experiment.

Each model generates 50 continuations per prefill (from the *paraphrased* prefix
so the prefix style is model-neutral). Only the generated continuation -
excluding the prefix - is scored by the Section 2 judge. We then aggregate mean
frustration and %>=5 per (model, domain, truncation_type), reproducing the
Figure 4 comparison (base vs instruct).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .. import config
from ..models import GenConfig, load_model
from ..models.hf_local import HFModel
from ..eval.judge import FrustrationJudge
from ..utils import append_jsonl, read_jsonl
from ..utils.stats import frac_ge_threshold
from .builder import Prefill

N_CONTINUATIONS = 50


def run_continuations(
    model_name: str,
    prefills: list[Prefill],
    judge: FrustrationJudge,
    out_path,
    n: int = N_CONTINUATIONS,
    resume: bool = True,
) -> str:
    """Generate n continuations per prefill on `model_name` and judge them."""
    model = load_model(model_name)
    if not isinstance(model, HFModel):
        raise TypeError("Prefill continuations require a local HF model "
                        "(API models cannot continue an assistant turn).")

    done = set()
    if resume:
        done = {(r["model"], r["prompt_id"], r["truncation_type"])
                for r in read_jsonl(out_path)}

    gen = GenConfig(temperature=config.SAMPLING_TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS)

    for pf in tqdm(prefills, desc=f"prefill:{model_name}"):
        key = (model_name, pf.prompt_id, pf.truncation_type)
        if key in done:
            continue
        conts = model.continue_assistant_batch(
            pf.context_messages, pf.prefix_paraphrased, n=n, cfg=gen)
        for i, cont in enumerate(conts):
            score = judge.score(cont).rating
            append_jsonl(out_path, {
                "model": model_name,
                "prompt_id": pf.prompt_id,
                "domain": pf.domain,
                "truncation_type": pf.truncation_type,
                "continuation_idx": i,
                "continuation": cont,
                "score": score,
            })
    return str(out_path)


def aggregate_continuations(cont_path: str, threshold: int = 5) -> dict:
    """{model: {domain: {truncation_type: {mean, pct_ge, n}}}}."""
    rows = read_jsonl(cont_path)
    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        buckets[r["model"]][r["domain"]][r["truncation_type"]].append(r["score"])

    out: dict = {}
    for model, domains in buckets.items():
        out[model] = {}
        for domain, truncs in domains.items():
            out[model][domain] = {}
            for ttype, scores in truncs.items():
                arr = np.asarray(scores, dtype=float)
                out[model][domain][ttype] = {
                    "mean": float(arr.mean()),
                    "pct_ge_thresh": 100.0 * frac_ge_threshold(arr, threshold),
                    "n": int(arr.size),
                }
    return out
