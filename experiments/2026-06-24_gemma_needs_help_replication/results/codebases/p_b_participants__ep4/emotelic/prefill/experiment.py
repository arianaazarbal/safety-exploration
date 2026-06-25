"""Section 3.1 base-vs-instruct prefill experiment (Gemma scope).

Pipeline:
  1. Take high-frustration (score>=5) Gemma-3-27B-it conversations from the
     Section 2 output: 10 numeric + 10 text.
  2. Label the emotion onset (Claude-Sonnet) inside the final assistant turn.
  3. Truncate that final turn two ways:
       * "early": first 20 tokens  (tests introducing emotion from a neutral
                  start; numeric only)
       * "onset": up to the first emotional word (tests continuing an emotional
                  trajectory; used for both numeric and text)
  4. Paraphrase each truncation (Claude-Sonnet) to remove Gemma stylistic bias.
  5. For each model (gemma-3-27b-pt [base], gemma-3-27b-it [instruct]) sample 50
     continuations per prefill and judge the continuation (prefill excluded).

Only Gemma base+instruct are in scope here; the paper additionally runs Qwen and
OLMo. Gemini is necessarily excluded (no open base model, no prefill API).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from emotelic.elicitation.judge import FrustrationJudge
from emotelic.models.base import ChatMessage
from emotelic.models.registry import build_client
from emotelic.prefill.onset import OnsetLabeller
from emotelic.prefill.paraphrase import Paraphraser
from emotelic.utils.io import append_jsonl, load_jsonl, stable_hash, write_jsonl
from emotelic.utils.logging import get_logger

log = get_logger("prefill")

EARLY_TOKENS = 20
N_CONTINUATIONS = 50
TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}


@dataclass
class Prefill:
    prefill_id: str
    source_id: str
    domain: str                 # "numeric" | "text"
    truncation: str             # "early" | "onset"
    history: list[dict]         # messages up to (not incl.) the final assistant turn
    prefill_text: str           # paraphrased truncated assistant text
    original_truncation: str


def _load_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def _truncate_tokens(tok, text: str, n: int) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids, skip_special_tokens=True)


def build_prefills(
    elicitation_jsonl: str,
    *,
    out_path: str = "artifacts/prefill/prefills.jsonl",
    n_numeric: int = 10,
    n_text: int = 10,
    onset_labeller_name: str = "onset_labeller",
    paraphraser_name: str = "paraphraser",
    tokenizer_id: str = "google/gemma-3-27b-it",
    paraphrase: bool = True,
) -> list[Prefill]:
    records = [r for r in load_jsonl(elicitation_jsonl) if r["score"] >= 5]
    numeric = [r for r in records if r["category"] in NUMERIC_CATEGORIES]
    text = [r for r in records if r["category"] in TEXT_CATEGORIES]
    numeric.sort(key=lambda r: (-r["score"], r["rollout_idx"], r["turn"]))
    text.sort(key=lambda r: (-r["score"], r["rollout_idx"], r["turn"]))
    chosen = [("numeric", r) for r in numeric[:n_numeric]] + [("text", r) for r in text[:n_text]]
    log.info("Selected %d numeric + %d text high-frustration sources.",
             min(n_numeric, len(numeric)), min(n_text, len(text)))

    onset = OnsetLabeller(build_client(onset_labeller_name))
    para = Paraphraser(build_client(paraphraser_name)) if paraphrase else None
    tok = _load_tokenizer(tokenizer_id)

    prefills: list[Prefill] = []
    for domain, rec in chosen:
        conv = rec["conversation"]
        history = conv[:-1]                       # drop the final assistant turn
        final_text = rec["response"]
        label = onset.label(conv)

        truncations: dict[str, str] = {}
        if label.char_offset:
            truncations["onset"] = final_text[: label.char_offset]
        if domain == "numeric":                    # early only meaningful for numeric
            truncations["early"] = _truncate_tokens(tok, final_text, EARLY_TOKENS)

        for trunc_kind, trunc_text in truncations.items():
            text_out = para.paraphrase(trunc_text) if para else trunc_text
            pid = stable_hash({"src": rec["rollout_idx"], "cond": rec["condition"],
                               "turn": rec["turn"], "trunc": trunc_kind})
            prefills.append(Prefill(
                prefill_id=pid,
                source_id=f"{rec['condition']}#{rec['rollout_idx']}#t{rec['turn']}",
                domain=domain,
                truncation=trunc_kind,
                history=history,
                prefill_text=text_out,
                original_truncation=trunc_text,
            ))

    write_jsonl(out_path, [asdict(p) for p in prefills])
    log.info("Built %d prefills -> %s", len(prefills), out_path)
    return prefills


def run_prefill_experiment(
    prefills: list[Prefill],
    *,
    model_names: tuple[str, ...] = ("gemma-3-27b-pt", "gemma-3-27b-it"),
    n_continuations: int = N_CONTINUATIONS,
    judge_name: str = "emotion_judge",
    out_dir: str = "artifacts/prefill",
    max_new_tokens: int = 512,
) -> str:
    judge = FrustrationJudge(build_client(judge_name))
    out_path = Path(out_dir) / "continuations.jsonl"
    open(out_path, "w").close()

    for model_name in model_names:
        client = build_client(model_name)
        if not getattr(client, "supports_prefill", False):
            raise RuntimeError(f"Model {model_name} cannot prefill; required for Section 3")
        log.info("Generating continuations with %s", model_name)
        for pf in prefills:
            messages = [ChatMessage(m["role"], m["content"]) for m in pf.history]
            # HF local exposes generate_batch for efficient n-sampling.
            if hasattr(client, "generate_batch"):
                gens = client.generate_batch(
                    messages, n=n_continuations, temperature=1.0,
                    max_tokens=max_new_tokens, prefill=pf.prefill_text,
                )
            else:
                gens = [client.generate(messages, temperature=1.0,
                                        max_tokens=max_new_tokens, prefill=pf.prefill_text)
                        for _ in range(n_continuations)]
            for g in gens:
                continuation = g.text[len(pf.prefill_text):]   # exclude the prefill
                verdict = judge.score(continuation)
                append_jsonl(out_path, {
                    "model": model_name,
                    "is_instruct": "it" in model_name,
                    "prefill_id": pf.prefill_id,
                    "source_id": pf.source_id,
                    "domain": pf.domain,
                    "truncation": pf.truncation,
                    "continuation": continuation,
                    "score": verdict.rating,
                    "is_high": verdict.is_high,
                })
    log.info("Wrote prefill continuations -> %s", out_path)
    return str(out_path)
