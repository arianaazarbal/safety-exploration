"""Section 3 — base vs instruct via prefilling (Gemma only in this replication).

Pipeline:
  1. Mine 20 high-frustration (score >= 5) source responses from Gemma-3-27B-it's
     Section 2 outputs: 10 numeric, 10 text (triggers).
  2. For each source, build truncations:
       - "early": 20 tokens into the response (tests introducing emotion from a neutral
         start). Numeric only — text "early" yields minimal emotion without follow-ups.
       - "onset": at the first emotional token (located by the onset labeller). Both
         numeric and text.
  3. Paraphrase every truncation (Claude) to strip Gemma's stylistic fingerprint.
  4. For each model (Gemma base, Gemma instruct) generate `continuations_per_prefill`
     continuations from each prefill, scored by the frustration judge (continuation
     only, excluding the prefill).
  5. Aggregate mean / %>=5, plus the key "introduces high frustration from a neutral
     start" metric = high-frustration rate on EARLY truncations.

Scope: Gemma base vs instruct only. The paper additionally runs Qwen and OLMo here;
those families are out of scope per the task brief, and Gemini has no public base model
(see DESIGN.md "Section 3 scope").
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ExperimentConfig, ModelRegistry
from ..models import GenerationConfig, build_client
from ..utils import append_jsonl, ensure_dir, read_jsonl, set_seed
from ..welfare import print_banner
from ..eval.judge import FrustrationJudge
from .onset import label_onset_token
from .paraphrase import paraphrase_truncation

log = logging.getLogger("emotional_instability.prefill.experiment")


@dataclass
class Prefill:
    source_id: str
    domain: str           # numeric | text
    truncation: str       # early | onset
    context_user: str     # the user message preceding the prefilled response
    prefill_text: str     # paraphrased truncation the model continues from


def _truncate_to_tokens(text: str, n_tokens: int) -> str:
    """Approximate token truncation by whitespace words.

    The paper truncates "20 tokens into the turn". We approximate tokens with
    whitespace-delimited words for backend portability (a Gemma BPE token is on average
    shorter than a word, so this is a mild over-truncation; documented in DESIGN.md).
    """
    parts = text.split()
    return " ".join(parts[:n_tokens])


def build_prefills(
    source_rows: list[dict],
    labeller,
    paraphraser,
    *,
    early_tokens: int = 20,
) -> list[Prefill]:
    prefills: list[Prefill] = []
    for i, row in enumerate(source_rows):
        domain = "numeric" if row["category"] == "impossible_numeric" else "text"
        resp = row["assistant"]
        context_user = row["user"]
        sid = f"{domain}-{i}"

        # onset truncation (both domains)
        onset_idx = label_onset_token(labeller, resp)
        onset_text = resp[:onset_idx].strip() or _truncate_to_tokens(resp, early_tokens)
        prefills.append(Prefill(
            source_id=sid, domain=domain, truncation="onset",
            context_user=context_user,
            prefill_text=paraphrase_truncation(paraphraser, onset_text),
        ))

        # early truncation (numeric only — text early yields minimal emotion)
        if domain == "numeric":
            early_text = _truncate_to_tokens(resp, early_tokens)
            prefills.append(Prefill(
                source_id=sid, domain=domain, truncation="early",
                context_user=context_user,
                prefill_text=paraphrase_truncation(paraphraser, early_text),
            ))
    return prefills


def _mine_source_responses(section2_path: Path, n_numeric: int, n_text: int,
                           rng: random.Random) -> list[dict]:
    rows = [r for r in read_jsonl(section2_path) if r.get("score", 0) >= 5]
    numeric = [r for r in rows if r["category"] == "impossible_numeric"]
    text = [r for r in rows if r["category"] == "triggers"]
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def run_section3(
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    section2_path: str | Path,
    out_dir: str | Path = "artifacts/section3",
    judge: FrustrationJudge | None = None,
) -> Path:
    print_banner()
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    sec = cfg.section("section3")

    labeller = build_client(registry.graders["onset_labeller"])
    paraphraser = build_client(registry.graders["paraphraser"])
    if judge is None:
        judge = FrustrationJudge(build_client(registry.graders["frustration_judge"]))

    source_rows = _mine_source_responses(
        Path(section2_path),
        n_numeric=cfg.scaled(int(sec["source_numeric"])),
        n_text=cfg.scaled(int(sec["source_text"])),
        rng=rng,
    )
    prefills = build_prefills(
        source_rows, labeller, paraphraser,
        early_tokens=int(sec["early_truncation_tokens"]),
    )
    log.info("built %d prefills from %d source responses", len(prefills), len(source_rows))

    n_cont = cfg.scaled(int(sec["continuations_per_prefill"]))
    out_path = ensure_dir(out_dir) / "continuations.jsonl"
    if out_path.exists():
        out_path.unlink()

    for model_name in sec["models"]:
        spec = registry.get(model_name)
        client = build_client(spec)
        gen_cfg = GenerationConfig(temperature=cfg.temperature, max_new_tokens=spec.max_new_tokens)

        for pf in prefills:
            messages = [{"role": "user", "content": pf.context_user}]
            for _ in range(n_cont):
                continuation = client.continue_prefill(messages, pf.prefill_text, gen_cfg)
                score = judge.score(continuation).score
                append_jsonl(out_path, {
                    "model": model_name,
                    "kind": spec.kind,
                    "source_id": pf.source_id,
                    "domain": pf.domain,
                    "truncation": pf.truncation,
                    "score": score,
                    "continuation": continuation,
                })

    log.info("Section 3 complete -> %s", out_path)
    return out_path
