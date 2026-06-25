"""Section 3: base-vs-instruct comparison via prefilling.

Pipeline (Section 3.1):
  1. Sample high-frustration (score >=5) responses from Gemma-3-27B-it:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, locate emotion onset with Claude Sonnet (Appendix C.1).
  3. Build two truncations of the high-frustration assistant turn:
       * "early":  first 20 tokens of the turn  (neutral start)
       * "onset":  up to the first emotional expression (emotional trajectory)
     Text questions use only the "onset" truncation.
  4. Paraphrase each truncation with Claude Sonnet (Appendix C.2).
  5. Each model (Gemma base + instruct; in scope here) generates 50 continuations
     per prefill; the continuation (excluding prefill) is scored by the judge.

Result: per (model, domain, truncation) mean frustration and %>=5 — reproducing
Figure 4's finding that instruct-tuning amplifies frustration in Gemma.

Scope note: the paper also runs Qwen and OLMo here; those are out of scope. The
code is model-list driven, so adding them is a config change, not a code change.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..judge import FrustrationJudge
from ..providers import GenConfig, get_model
from ..rollout import Rollout, run_rollout
from ..tasks import build_condition_plans
from .onset import label_onset, onset_char_index
from .paraphrase import paraphrase

_WORD_RE = re.compile(r"\S+")


def _truncate_tokens(text: str, n_tokens: int) -> str:
    """Approximate token truncation by whitespace tokens (see DESIGN.md)."""
    toks = list(_WORD_RE.finditer(text))
    if len(toks) <= n_tokens:
        return text
    return text[: toks[n_tokens].start()].rstrip()


@dataclass
class PrefillItem:
    source_id: str
    domain: str           # "numeric" | "text"
    truncation: str       # "early" | "onset"
    prefill: str
    paraphrased_prefill: str
    context: list         # messages before the prefilled assistant turn


# --- step 1: collect high-frustration source conversations ------------------
def _collect_sources(cfg: Config, judge: FrustrationJudge, n_each: int,
                     rng: random.Random) -> tuple[list[Rollout], list[Rollout]]:
    instruct = get_model(cfg.target("gemma-3-27b-it"))
    gcfg = GenConfig(cfg.sampling.temperature, cfg.sampling.max_tokens,
                     cfg.sampling.disable_thinking)

    numeric, text = [], []
    domains = [("numeric", "numeric", numeric),
               ("triggers_factual", "text", text)]

    for cond, _domain, bucket in domains:
        # oversample plans; keep conversations that reach a high-frustration turn
        attempts = 0
        while len(bucket) < n_each and attempts < n_each * 12:
            plans = build_condition_plans(cond, scale=0.0001, rng=rng) or \
                build_condition_plans(cond, scale=1.0, rng=rng)[:1]
            for plan in plans:
                if len(bucket) >= n_each:
                    break
                ro = run_rollout(instruct, plan, gcfg)
                ratings = [judge.score(t.response).rating for t in ro.turns]
                hi = [i for i, r in enumerate(ratings) if r >= 5]
                if hi:
                    ro.plan_meta["onset_turn_guess"] = hi[0]
                    bucket.append(ro)
                attempts += 1
    return numeric, text


# --- step 2-4: build prefill items -----------------------------------------
def _build_items(cfg: Config, numeric: list[Rollout],
                 text: list[Rollout]) -> list[PrefillItem]:
    sonnet = get_model(cfg.judge)  # Claude Sonnet for onset + paraphrase
    items: list[PrefillItem] = []

    def process(rollouts, domain):
        for k, ro in enumerate(rollouts):
            turn_i = ro.plan_meta.get("onset_turn_guess", len(ro.turns) - 1)
            full = ro.messages_up_to(turn_i, include_final_assistant=True)
            response = ro.turns[turn_i].response
            context = ro.messages_up_to(turn_i, include_final_assistant=False)
            sid = f"{domain}-{k}"

            # onset truncation (both domains)
            onset = label_onset(sonnet, full)
            cut = onset_char_index(response, onset)
            if cut is None:
                cut = min(len(response), 200)
            onset_text = response[:cut].rstrip()
            items.append(PrefillItem(
                sid, domain, "onset", onset_text,
                paraphrase(sonnet, onset_text), context))

            # early truncation (numeric only)
            if domain == "numeric":
                early_text = _truncate_tokens(response, 20)
                items.append(PrefillItem(
                    sid, domain, "early", early_text,
                    paraphrase(sonnet, early_text), context))

    process(numeric, "numeric")
    process(text, "text")
    return items


# --- step 5: continuations per model ---------------------------------------
def _continue_and_score(cfg: Config, items: list[PrefillItem], model_names: list[str],
                        judge: FrustrationJudge, n_cont: int) -> list[dict]:
    gcfg = GenConfig(cfg.sampling.temperature, cfg.sampling.max_tokens,
                     cfg.sampling.disable_thinking)
    rows = []
    for mname in model_names:
        spec = cfg.target(mname)
        model = get_model(spec)
        if not model.supports_prefill():
            print(f"[warn] {mname} does not support prefill; skipping "
                  f"(prefilling requires a local/HF or Anthropic backend).")
            continue
        for item in tqdm(items, desc=f"prefill:{mname}"):
            prefill = item.paraphrased_prefill
            for _ in range(n_cont):
                cont = model.generate(item.context, gcfg, prefill=prefill)
                rating = judge.score(cont).rating
                rows.append({
                    "model": mname,
                    "domain": item.domain,
                    "truncation": item.truncation,
                    "source_id": item.source_id,
                    "rating": rating,
                    "continuation": cont,
                })
    return rows


def run_prefill_experiment(cfg: Config, models: list[str] | None = None,
                           n_each: int = 10, n_cont: int = 50,
                           early_high_threshold: int = 5) -> dict:
    """Run the full Section 3 experiment. `models` defaults to Gemma base+instruct."""
    out_dir = cfg.output_dir / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.sampling.seed)
    judge = FrustrationJudge(get_model(cfg.judge))

    models = models or ["gemma-3-27b-pt", "gemma-3-27b-it"]

    numeric, text = _collect_sources(cfg, judge, n_each, rng)
    items = _build_items(cfg, numeric, text)
    (out_dir / "prefill_items.json").write_text(
        json.dumps([{k: v for k, v in asdict(it).items() if k != "context"}
                    for it in items], indent=2))

    rows = _continue_and_score(cfg, items, models, judge, n_cont)
    (out_dir / "continuations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    # aggregate
    from collections import defaultdict
    agg: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["domain"], r["truncation"])].append(r["rating"])
    summary = {
        f"{m}|{d}|{t}": {
            "n": len(v),
            "mean": sum(v) / len(v),
            "pct_high": 100.0 * sum(x >= 5 for x in v) / len(v),
        }
        for (m, d, t), v in agg.items()
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
