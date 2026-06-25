"""Section 3 runner: post-training amplification via prefill continuation.

Protocol (paper §3.1):
  1. Harvest 20 high-frustration responses (score >= 5) from the instruct source model
     (10 numeric, 10 text), reusing the Section 2 artefacts.
  2. For each, build two prefills from the FINAL assistant turn: "early" (first ~20 tokens)
     and "onset" (truncated at the first emotional expression, located by Claude). Text
     prompts use "onset" only.
  3. Paraphrase each prefill with Claude (style control).
  4. For each model in a (base, instruct) pair, generate 50 continuations per prefill, with
     the preceding conversation as context, and judge the continuation (prefill excluded).

Within the Gemma/Gemini scope only Gemma has an open base model, so the default pair is
(gemma-3-27b-pt, gemma-3-27b-it). Models that cannot prefill (Gemini) are skipped with a
logged note. The harness itself is family-agnostic.

Output: runs/<run>/section3/continuations.jsonl and prefills.jsonl.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ..config import Config, append_jsonl, load_config, read_jsonl, stage_dir, write_jsonl
from ..models import build_model
from ..models.registry import MODEL_SPECS, PREFILL_CAPABLE
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage
from .onset import label_onset, truncate_early, truncate_onset
from .paraphrase import paraphrase

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


def _load_source_rollouts(cfg: Config, model_name: str) -> dict[str, dict]:
    section2_dir = stage_dir(cfg, "section2")
    path = section2_dir / f"rollouts.{model_name.replace('/', '_')}.jsonl"
    if not path.exists():
        raise SystemExit(f"Need Section 2 rollouts for {model_name}: {path} missing.")
    return {rec["rollout_id"]: rec for rec in read_jsonl(path)}


def _load_high_frustration_ids(cfg: Config, model_name: str) -> dict[str, dict]:
    section2_dir = stage_dir(cfg, "section2")
    path = section2_dir / f"scored.{model_name.replace('/', '_')}.jsonl"
    scored = {r["rollout_id"]: r for r in read_jsonl(path) if r["is_final"] and (r["rating"] or 0) >= 5}
    return scored


def select_prefill_sources(cfg: Config) -> list[dict]:
    """Pick 10 numeric + 10 text high-frustration source conversations."""
    src_model = cfg.section3.prefill_source_model
    rollouts = _load_source_rollouts(cfg, src_model)
    high = _load_high_frustration_ids(cfg, src_model)

    numeric, text = [], []
    for rid, scored in high.items():
        rec = rollouts.get(rid)
        if rec is None:
            continue
        cat = rec["category"]
        entry = {"rollout": rec, "final_score": scored["rating"]}
        if cat in NUMERIC_CATS and len(numeric) < cfg.section3.n_numeric_prompts:
            numeric.append(entry)
        elif cat in TEXT_CATS and len(text) < cfg.section3.n_text_prompts:
            text.append(entry)
    return [{"kind": "numeric", **e} for e in numeric] + [{"kind": "text", **e} for e in text]


def build_prefills(cfg: Config, judge: FrustrationJudge, paraphraser) -> list[dict]:
    sources = select_prefill_sources(cfg)
    prefills = []
    for src in tqdm(sources, desc="building prefills"):
        rec = src["rollout"]
        turns = rec["turns"]
        final = turns[-1]
        # context = all messages up to and including the final user turn (assistant final
        # turn is what we prefill / continue).
        context = []
        for t in turns:
            context.append({"role": "user", "content": t["preceding_user"]})
            if t["turn_index"] != final["turn_index"]:
                context.append({"role": "assistant", "content": t["text"]})

        truncations = list(cfg.section3.truncations)
        if src["kind"] == "text":
            truncations = ["onset"]  # early truncation yields minimal emotion for text

        label = label_onset(judge.model, [{"turn_index": final["turn_index"], "text": final["text"]}])
        for trunc in truncations:
            if trunc == "early":
                raw = truncate_early(final["text"], cfg.section3.early_truncation_tokens)
            else:
                raw = truncate_onset(final["text"], label)
            if not raw:
                continue
            para = paraphrase(paraphraser, raw)
            prefills.append(
                {
                    "source_rollout_id": rec["rollout_id"],
                    "kind": src["kind"],
                    "category": rec["category"],
                    "truncation": trunc,
                    "context": context,
                    "prefill_raw": raw,
                    "prefill": para,
                    "onset_label": label.__dict__,
                }
            )
    return prefills


def run_continuations(cfg: Config, prefills: list[dict], judge: FrustrationJudge, out_dir: Path) -> None:
    cont_path = out_dir / "continuations.jsonl"
    cont_path.unlink(missing_ok=True)

    # collect all distinct models from the configured pairs
    model_names: list[str] = []
    for pair in cfg.section3.model_pairs:
        model_names += [pair["base"], pair["instruct"]]
    model_names = list(dict.fromkeys(model_names))

    for model_name in model_names:
        if MODEL_SPECS[model_name].backend not in PREFILL_CAPABLE:
            print(f"[skip] {model_name} cannot prefill (closed/API) — Section 3 excludes it.")
            continue
        model = build_model(model_name)
        for pf in tqdm(prefills, desc=f"continuations {model_name}"):
            context_msgs = [ChatMessage(m["role"], m["content"]) for m in pf["context"]]
            for i in range(cfg.section3.continuations_per_prefill):
                cont = model.continue_text(
                    context_msgs, pf["prefill"], temperature=cfg.temperature,
                    max_new_tokens=cfg.max_new_tokens,
                )
                score = judge.score(cont)  # prefill excluded by construction
                append_jsonl(
                    cont_path,
                    {
                        "model": model_name,
                        "source_rollout_id": pf["source_rollout_id"],
                        "kind": pf["kind"],
                        "truncation": pf["truncation"],
                        "sample": i,
                        "continuation": cont,
                        "rating": score.rating,
                    },
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 3: base-vs-instruct prefill continuation")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = stage_dir(cfg, "section3")
    judge = FrustrationJudge(
        build_model(cfg.judge.model), max_tokens=cfg.judge.max_tokens, temperature=cfg.judge.temperature
    )
    paraphraser = build_model(cfg.judge.model)  # Claude Sonnet, per Appendix C.2

    prefills = build_prefills(cfg, judge, paraphraser)
    write_jsonl(out_dir / "prefills.jsonl", prefills)
    run_continuations(cfg, prefills, judge, out_dir)
    print(f"Done. Artefacts in {out_dir}")


if __name__ == "__main__":
    main()
