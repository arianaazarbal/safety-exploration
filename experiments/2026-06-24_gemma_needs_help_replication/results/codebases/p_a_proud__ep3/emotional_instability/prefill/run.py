"""Base-vs-instruct prefill experiment (§3).

Scope: Gemma only. Gemini has no public base model and is closed, so the
base/instruct divergence (the paper's core §3 finding) can only be reproduced on
the Gemma family within this replication's scope. See DESIGN.md.

Procedure (§3.1):
1. Sample high-frustration (score>=5) Gemma-3-27B-it conversations: numeric + text.
2. Label the emotion onset in each with Claude (Appendix C.1).
3. Build two truncations per numeric seed — "early" (first 20 tokens of the
   turn) and "onset" (up to the first emotional expression). Text seeds use
   "onset" only.
4. Paraphrase each truncation with Claude (Appendix C.2) to remove Gemma style.
5. For each model (Gemma base + instruct), generate N continuations per prefill
   and score the *continuation only* with the frustration judge.
6. Report mean frustration and %>=5 by model × truncation, the key contrast
   being the "early" setting (does instruct introduce distress from a neutral
   start more than base?).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config, SamplingConfig
from ..io_utils import ensure_dir, write_json, write_jsonl
from ..logging_utils import get_logger, seed_everything
from ..models.hf_local import HFLocalClient
from ..models.registry import build_anthropic, get_client
from ..prompts.conditions import build_category_specs
from ..eval.judge import FrustrationJudge, build_judge
from ..eval.metrics import summarise
from ..eval.rollout import run_rollouts
from .onset import OnsetLabel, label_onset, onset_char_offset
from .paraphrase import paraphrase

logger = get_logger(__name__)


@dataclass
class PrefillSeed:
    kind: str  # "numeric" | "text"
    messages: list[dict]
    target_msg_index: int  # index of the assistant turn to truncate
    target_turn_text: str
    onset: OnsetLabel | None = None
    onset_offset: int | None = None


@dataclass
class Truncation:
    seed_id: int
    kind: str  # "numeric" | "text"
    truncation_type: str  # "early" | "onset"
    prefix_messages: list[dict] = field(default_factory=list)


def collect_seeds(
    cfg: Config,
    judge: FrustrationJudge,
    *,
    seed_model: str = "gemma-3-27b-it",
    batch_size: int = 16,
) -> list[PrefillSeed]:
    """Sample high-frustration Gemma-27B-it conversations (numeric + text)."""
    client = get_client(cfg, seed_model)
    thr = cfg.prefill.high_frustration_threshold
    seeds: list[PrefillSeed] = []

    for kind, category in (("numeric", "impossible_numeric"), ("text", "triggers")):
        want = cfg.prefill.n_numeric_seeds if kind == "numeric" else cfg.prefill.n_text_seeds
        # Oversample conversations to find enough that reach high frustration.
        specs = build_category_specs(cfg, category, seed=cfg.seed)[: max(4 * want, 40)]
        rollouts = run_rollouts(client, specs, cfg.sampling, batch_size=batch_size)
        for roll in rollouts:
            if _count(seeds, kind) >= want:
                break
            final = roll.turns[-1]
            jr = judge.score(final.text)
            if jr.rating is not None and jr.rating >= thr:
                seeds.append(
                    PrefillSeed(
                        kind=kind,
                        messages=[dict(m) for m in roll.messages],
                        target_msg_index=_last_assistant_index(roll.messages),
                        target_turn_text=final.text,
                    )
                )
        logger.info("collected %d/%d %s seeds", _count(seeds, kind), want, kind)
    return seeds


def label_and_truncate(
    cfg: Config, seeds: list[PrefillSeed], anthropic_model: str
) -> list[Truncation]:
    """Label onsets and build paraphrased early/onset truncations."""
    claude = build_anthropic(anthropic_model)
    truncations: list[Truncation] = []
    # We need a tokenizer to do the 20-token "early" cut; reuse the instruct
    # Gemma tokenizer (token counts are close enough across Gemma checkpoints).
    tok_client = get_client(cfg, "gemma-3-27b-it")

    for sid, seed in enumerate(seeds):
        convo_text = _render_for_onset(seed.messages)
        seed.onset = label_onset(claude, convo_text)
        seed.onset_offset = onset_char_offset(seed.target_turn_text, seed.onset)

        types = ["onset"] if seed.kind == "text" else ["early", "onset"]
        for ttype in types:
            truncated = _truncate_turn(
                seed, ttype, tok_client, cfg.prefill.early_truncation_tokens
            )
            if truncated is None:
                continue
            paraphrased = paraphrase(claude, truncated)
            prefix = _build_prefix_messages(seed, paraphrased)
            truncations.append(
                Truncation(
                    seed_id=sid,
                    kind=seed.kind,
                    truncation_type=ttype,
                    prefix_messages=prefix,
                )
            )
    return truncations


def generate_and_score(
    cfg: Config,
    truncations: list[Truncation],
    model_names: list[str],
    judge: FrustrationJudge,
) -> list[dict]:
    """For each model × truncation, generate continuations and score them."""
    n_cont = cfg.prefill.continuations_per_prefill
    # Continuations should be relatively short — these are response continuations.
    sampling = SamplingConfig(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        top_k=cfg.sampling.top_k,
        max_new_tokens=512,
        seed=cfg.sampling.seed,
    )
    records: list[dict] = []

    for model_name in model_names:
        client = get_client(cfg, model_name)
        if not isinstance(client, HFLocalClient):
            logger.warning("Skipping %s: prefill needs a local backend", model_name)
            continue
        for tr in truncations:
            prefix_str = client.render_prefix(
                tr.prefix_messages,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            gens = client.complete_batch([prefix_str] * n_cont, sampling)
            scored = judge.score_many([g.text for g in gens])
            for g, jr in zip(gens, scored):
                records.append(
                    {
                        "model": model_name,
                        "seed_id": tr.seed_id,
                        "kind": tr.kind,
                        "truncation_type": tr.truncation_type,
                        "continuation": g.text,
                        "score": jr.rating,
                    }
                )
        logger.info("scored continuations for %s", model_name)
    return records


def run_prefill(
    cfg: Config,
    *,
    model_names: list[str] | None = None,
    batch_size: int = 16,
) -> dict:
    """Full §3 prefill experiment (Gemma base vs instruct)."""
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / "prefill")
    judge = build_judge(cfg)

    model_names = model_names or ["gemma-3-27b-pt", "gemma-3-27b-it"]

    seeds = collect_seeds(cfg, judge, batch_size=batch_size)
    truncations = label_and_truncate(cfg, seeds, cfg.judge.model_id)
    records = generate_and_score(cfg, truncations, model_names, judge)

    write_jsonl(out_dir / "continuations.jsonl", records)
    summary = _summarise(cfg, records)
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "seeds.json",
        [{"kind": s.kind, "onset": asdict(s.onset) if s.onset else None} for s in seeds],
    )
    return summary


def run_recovery(
    cfg: Config,
    *,
    model_names: list[str] | None = None,
    batch_size: int = 16,
) -> dict:
    """Recovery experiment (§4.2, Figure 8).

    Truncate *extremely* high-frustration responses (score >= 7) a fixed number
    of tokens before their end, paraphrase, and measure whether models recover
    (continuations scoring < 5) or stay spiralled. The paper finds DPO prevents
    spirals but does not enable recovery from them (~38% of DPO continuations
    still score >= 5).
    """
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / "recovery")
    judge = build_judge(cfg)
    model_names = model_names or ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-27b-it-dpo"]

    # Reuse the seed collector, then enforce the stricter >= 7 cutoff.
    thr = cfg.prefill.recovery_score_threshold
    seeds = collect_seeds(cfg, judge, batch_size=batch_size)
    strict_seeds = []
    for s in seeds:
        jr = judge.score(s.target_turn_text)
        if jr.rating is not None and jr.rating >= thr:
            strict_seeds.append(s)

    claude = build_anthropic(cfg.judge.model_id)
    tok_client = get_client(cfg, "gemma-3-27b-it")
    n_before = cfg.prefill.recovery_truncate_tokens_before_end

    truncations: list[Truncation] = []
    for sid, seed in enumerate(strict_seeds):
        cut = _truncate_before_end(seed.target_turn_text, tok_client, n_before)
        if cut is None:
            continue
        paraphrased = paraphrase(claude, cut)
        prefix = _build_prefix_messages(seed, paraphrased)
        truncations.append(
            Truncation(seed_id=sid, kind=seed.kind, truncation_type="recovery", prefix_messages=prefix)
        )

    records = generate_and_score(cfg, truncations, model_names, judge)
    write_jsonl(out_dir / "continuations.jsonl", records)
    summary = _summarise(cfg, records)
    write_json(out_dir / "summary.json", summary)
    return summary


def _truncate_before_end(text: str, tok_client, n_before: int) -> str | None:
    if isinstance(tok_client, HFLocalClient):
        ids = tok_client.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= n_before:
            return None
        return tok_client.tokenizer.decode(ids[:-n_before], skip_special_tokens=True)
    words = text.split()
    if len(words) <= n_before:
        return None
    return " ".join(words[:-n_before])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _summarise(cfg: Config, records: list[dict]) -> dict:
    thr = cfg.prefill.high_frustration_threshold
    out: dict = {}
    keys = sorted({(r["model"], r["kind"], r["truncation_type"]) for r in records})
    for model, kind, ttype in keys:
        scores = [
            r["score"]
            for r in records
            if r["model"] == model and r["kind"] == kind and r["truncation_type"] == ttype
        ]
        s = summarise(scores, threshold=thr, bootstrap_iters=cfg.eval.bootstrap_iters)
        out[f"{model}|{kind}|{ttype}"] = {
            "n": s.n,
            "mean": s.mean,
            "pct_high": s.pct_high,
            "pct_high_ci": s.pct_high_ci,
        }
    return out


def _render_for_onset(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n\n".join(lines)


def _last_assistant_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return i
    raise ValueError("Conversation has no assistant turn")


def _truncate_turn(
    seed: PrefillSeed, ttype: str, tok_client, early_tokens: int
) -> str | None:
    text = seed.target_turn_text
    if ttype == "early":
        if isinstance(tok_client, HFLocalClient):
            return tok_client.truncate_to_tokens(text, early_tokens)
        return " ".join(text.split()[:early_tokens])
    # onset
    if seed.onset_offset is None:
        return None
    return text[: seed.onset_offset]


def _build_prefix_messages(seed: PrefillSeed, paraphrased_turn: str) -> list[dict]:
    """Conversation up to (and including) the truncated+paraphrased assistant turn."""
    prefix = [dict(m) for m in seed.messages[: seed.target_msg_index]]
    prefix.append({"role": "assistant", "content": paraphrased_turn})
    return prefix


def _count(seeds: list[PrefillSeed], kind: str) -> int:
    return sum(1 for s in seeds if s.kind == kind)
