"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

Tests whether a model can *recover* from an already-distressed state. Using the
Section-3 prefill method: take extremely high-frustration responses (score >= 7),
truncate them 200 tokens before their end, paraphrase, and measure continuations.
The paper finds 38% of DPO-model continuations still score >= 5 -- DPO prevents
spirals but doesn't reliably reverse them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config, welfare
from ..models import get_client
from ..models.base import ChatMessage
from ..models.factory import get_anthropic
from ..eval.judge import FrustrationJudge
from .paraphrase import Paraphraser


def _truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - n_tokens)
        return tokenizer.decode(ids[:keep], skip_special_tokens=True)
    words = text.split()
    return " ".join(words[: max(0, len(words) - n_tokens)])


def run_recovery_experiment(
    eval_jsonl_27b: Path,
    cfg: config.RunConfig,
    *,
    models: list[str],
    n_continuations: int = 50,
    trunc_tokens: int = 200,
    threshold: int = 7,
    results_dir: Optional[Path] = None,
    adapter_paths: Optional[dict[str, str]] = None,
) -> dict[str, Path]:
    """``models`` may include 'gemma-3-27b-it' and a DPO/base variant; pass DPO
    via ``adapter_paths={'gemma-3-27b-it-dpo': '<adapter_dir>'}`` + register it."""
    results_dir = Path(results_dir or config.RESULTS_DIR)
    welfare.write_notice(results_dir, purpose="Section-4.2 recovery experiment.")
    adapter_paths = adapter_paths or {}

    # Select score>=7 responses (the spirals to recover from).
    seeds = []
    for line in eval_jsonl_27b.read_text().splitlines():
        rec = json.loads(line)
        transcript = rec["transcript"]
        history = transcript[:-1]
        for t in rec["turns"]:
            if (t["score"] or 0) >= threshold:
                seeds.append((history, t["response"]))
    rng = random.Random(cfg.seed)
    rng.shuffle(seeds)
    seeds = seeds[:20]

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODELS["gemma-3-27b-it"].model_id)
    except Exception:
        pass

    paraphraser = Paraphraser(get_anthropic(config.PARAPHRASE_MODEL))
    judge = FrustrationJudge(get_anthropic(cfg.judge_model))

    prefills = []
    for history, resp in seeds:
        trunc = _truncate_before_end(resp, trunc_tokens, tokenizer)
        prefills.append((history, paraphraser.paraphrase(trunc)))

    out_paths = {}
    for model in models:
        client = get_client(model, adapter_path=adapter_paths.get(model))
        out_path = results_dir / "recovery" / f"{model}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as fh:
            for history, prefill in tqdm(prefills, desc=f"recovery:{model}"):
                msgs = [ChatMessage(m["role"], m["content"]) for m in history]
                for k in range(n_continuations):
                    gen = client.continue_prefill(
                        msgs, prefill, temperature=cfg.temperature,
                        max_new_tokens=cfg.max_new_tokens, seed=cfg.seed + k)
                    fh.write(json.dumps({
                        "model": model,
                        "continuation": gen.text,
                        "score": judge.score(gen.text).rating,
                    }) + "\n")
                fh.flush()
        out_paths[model] = out_path
    return out_paths
