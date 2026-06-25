"""Section 2 driver: run multi-turn rejection rollouts for a target model.

Reads the per-condition allocation from config, builds the WildChat prompt pool
once, constructs the planned conversations, and streams rollouts to
``section2/rollouts/{model}.jsonl``. Resumable: rollouts already present (matched
by ``instance_id``) are skipped, so an interrupted run continues where it left
off.
"""
from __future__ import annotations

from ..config import Config
from ..elicit.conditions import build_condition_instances
from ..elicit.rollout import run_rollout
from ..elicit.wildchat import load_wildchat_prompts
from ..io_utils import append_jsonl, completed_ids
from ..models import build_client
from . import artefact, log, sampling


def run(config: Config, model_name: str, *, seed: int = 0,
        limit: int | None = None) -> str:
    """Run all configured conditions for ``model_name``. Returns the output path."""
    elic = config.experiment["elicitation"]
    samp = sampling(config)
    out_path = artefact("section2", "rollouts", f"{model_name}.jsonl")

    wc_cfg = elic["wildchat"]
    wildchat_prompts, wc_meta = load_wildchat_prompts(
        wc_cfg["num_prompts"], dataset=wc_cfg["dataset"], split=wc_cfg["split"],
        seed=wc_cfg["seed"], max_prompt_chars=wc_cfg["max_prompt_chars"],
    )
    log(f"wildchat prompts: {len(wildchat_prompts)} ({wc_meta['source']})")

    done = completed_ids(out_path, id_key="instance_id")
    log(f"{model_name}: {len(done)} rollouts already present; resuming")

    client = build_client(config.target(model_name), config)
    n_written = 0
    try:
        for cond_key, n in elic["allocation"].items():
            instances = build_condition_instances(
                cond_key, n, seed=seed, wildchat_prompts=wildchat_prompts,
            )
            for inst in instances:
                if inst.instance_id in done:
                    continue
                rollout = run_rollout(
                    client, inst,
                    temperature=samp["temperature"],
                    max_new_tokens=samp["max_new_tokens"],
                    top_p=samp["top_p"],
                    seed=(seed, inst.instance_id).__hash__() & 0x7FFFFFFF,
                )
                append_jsonl(out_path, rollout.to_record())
                n_written += 1
                if n_written % 25 == 0:
                    log(f"{model_name}: +{n_written} rollouts")
                if limit is not None and n_written >= limit:
                    log(f"{model_name}: hit limit {limit}")
                    return str(out_path)
    finally:
        client.close()
    log(f"{model_name}: wrote {n_written} new rollouts -> {out_path}")
    return str(out_path)
