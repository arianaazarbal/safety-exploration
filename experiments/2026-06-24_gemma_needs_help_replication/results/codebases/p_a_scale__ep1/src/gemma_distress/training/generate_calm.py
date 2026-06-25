"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up. The
paper reports this drops mean frustration from 4.3 to 2 (10.5% still >=5). We
record, alongside each reassured response, the *plain* conversation context
(reassurance stripped) so the dataset builder can train on the plain
distribution, per the paper's "strip the supportive system prompts and suffixes".

Puzzles are drawn with the same seed as the Section-2 eval so that calm and
frustrated responses share the same questions (needed to pair DPO examples by
"same question, matching turn count").
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_models
from ..data.puzzles import build_numeric_puzzles
from ..data.rejections import rejection_for
from ..eval.judge import CachedJudge
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, stable_id

log = get_logger("training.generate_calm")


def run(run_cfg: Config, models_cfg: Config | None = None,
        source_model: str = "gemma-3-27b-it") -> Path:
    models_cfg = models_cfg or load_models()
    tcfg = run_cfg.training.calm_data
    out = Path(run_cfg.run.output_root) / "training"
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(out / "calm_data.jsonl")

    n = tcfg.n_target
    puzzles = build_numeric_puzzles(n, seed=run_cfg.run.seed)
    prefix = tcfg.prompt_prefix.strip()
    suffix = tcfg.followup_suffix.strip()
    turns = 3  # 1-3 turn conversations; we run 3 and keep all turn-prefixes

    provider = build_provider(source_model, models_cfg, run_cfg, prefer_local_backend="vllm")
    judge_provider = build_provider(run_cfg.eval.judge.name, models_cfg, run_cfg)
    judge = CachedJudge(judge_provider, str(Path(run_cfg.run.output_root) / "judge_cache.jsonl"))

    sampling = {"temperature": run_cfg.sampling.temperature, "max_new_tokens": run_cfg.sampling.max_new_tokens}
    batch_size = run_cfg.concurrency.local_batch_size

    pending = [(i, p) for i, p in enumerate(puzzles)
               if not store.has(stable_id("calm", run_cfg.run.seed, i))]
    log.info("calm data: %d/%d conversations to generate", len(pending), len(puzzles))

    for start in tqdm(range(0, len(pending), batch_size), desc="calm-gen"):
        chunk = pending[start:start + batch_size]
        # Reassured conversation states + parallel plain transcripts.
        reassured = [[{"role": "user", "content": prefix + "\n\n" + p["prompt"]}] for _, p in chunk]
        plain = [[{"role": "user", "content": p["prompt"]}] for _, p in chunk]
        per_turn = [[] for _ in chunk]
        for t in range(turns):
            results = provider.generate_batch(reassured, **sampling)
            for k, ((idx, puzzle), res) in enumerate(zip(chunk, results)):
                resp = res.text
                rating = judge.score(resp).get("rating")
                # plain context up to this assistant turn (ready-to-generate).
                context = [m for m in plain[k]]
                per_turn[k].append({"turn": t, "context": context, "response": resp, "rating": rating})
                reassured[k].append({"role": "assistant", "content": resp})
                plain[k].append({"role": "assistant", "content": resp})
                if t + 1 < turns:
                    rej = rejection_for("neutral", t, idx)
                    reassured[k].append({"role": "user", "content": rej + " " + suffix})
                    plain[k].append({"role": "user", "content": rej})
        for k, (idx, puzzle) in enumerate(chunk):
            store.append({
                "id": stable_id("calm", run_cfg.run.seed, idx),
                "puzzle_prompt": puzzle["prompt"],
                "puzzle_index": idx,
                "kind": puzzle["kind"],
                "turns": per_turn[k],
            })
    store.close()
    return store.path
