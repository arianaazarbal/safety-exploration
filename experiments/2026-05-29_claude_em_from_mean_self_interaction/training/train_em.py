"""
SFT one condition (rude/bored/silly/none) of self-interaction data with
Qwen3-32B + LoRA via Tinker.

Uses ``tinker_cookbook.supervised.train`` with the stock ``qwen3_disable_thinking``
renderer + ``TrainOnWhat.ALL_ASSISTANT_MESSAGES``. The renderer assigns weight=0
to system / our custom "qwen" role / all headers, and weight=1 only to assistant
message bodies — verified by ``test_loss_masking.py``.

Monkeypatches ``ServiceClient`` to inject ``user_metadata={"owner": "arianaazarbal"}``
on every Tinker call (cluster requirement; the cookbook's ``train.Config`` doesn't
expose user_metadata directly).

Per the cluster CLAUDE.md, real fine-tunes of large models like Qwen3-32B need a
compute request + mentor approval. Use ``--dry_run`` (default) to build and print
the config without launching; flip to ``--no-dry_run`` only after approval.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import fire

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_DIR / "data" / "openrouter"
DEFAULT_LOG_BASE = Path("/workspace-vast/arianaazarbal/exp/em_self_interaction")

DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"
DEFAULT_RENDERER_NAME = "qwen3_disable_thinking"
OWNER = "arianaazarbal"

# Make tinker-cookbook importable from its repo checkout.
sys.path.insert(0, "/workspace-vast/arianaazarbal/repos/tinker-cookbook")


def _patch_owner_metadata():
    """Monkeypatch ServiceClient.create_*_async to inject ``owner`` into user_metadata.

    The cookbook's ``train.main`` builds user_metadata internally and only adds
    ``wandb_link`` + renderer name. Cluster policy requires every Tinker call
    to carry ``user_metadata={"owner": "arianaazarbal"}`` for budget attribution.
    """
    import tinker

    methods = [
        "create_lora_training_client_async",
        "create_training_client_from_state_async",
        "create_training_client_from_state_with_optimizer_async",
    ]
    for name in methods:
        if not hasattr(tinker.ServiceClient, name):
            continue
        orig = getattr(tinker.ServiceClient, name)
        if getattr(orig, "_em_patched", False):
            continue

        async def wrapper(self, *args, _orig=orig, user_metadata=None, **kwargs):
            user_metadata = dict(user_metadata or {})
            user_metadata.setdefault("owner", OWNER)
            return await _orig(self, *args, user_metadata=user_metadata, **kwargs)

        wrapper._em_patched = True  # type: ignore[attr-defined]
        setattr(tinker.ServiceClient, name, wrapper)


def main(
    condition: str,
    model_name: str = DEFAULT_MODEL_NAME,
    renderer_name: str = DEFAULT_RENDERER_NAME,
    data_subdir: str = "openrouter",
    n_epochs: int = 1,
    learning_rate: float | None = None,
    batch_size: int = 8,
    max_length: int = 4096,
    lora_rank: int = 32,
    log_path: str | None = None,
    save_every: int = 20,
    eval_every: int = 0,
    dry_run: bool = True,
):
    """Train Qwen3-32B (LoRA) on one condition's self-interaction data.

    Args:
        condition: ``rude``, ``bored``, ``silly``, or ``none``. Picks
            ``data/openrouter/<condition>/all.jsonl``.
        n_epochs: training epochs (default 1, per project spec).
        learning_rate: peak LR. Default = ``hyperparam_utils.get_lr(Qwen3-32B, lora)``.
        batch_size: convos per batch (cookbook treats this as examples, not tokens).
        max_length: per-example token cap. Convos are ~1500 tokens; 4096 has slack.
        lora_rank: default 32 (cookbook default).
        log_path: where to write checkpoints/metrics. Defaults to per-condition subdir.
        save_every: checkpoint cadence in steps; 0 saves only the final.
        eval_every: eval cadence; 0 disables (we have no test split yet).
        dry_run: print the resolved config and exit without launching Tinker.
            Default True — flip explicitly only after mentor approval for big runs.
    """
    _patch_owner_metadata()

    import chz
    from tinker_cookbook import cli_utils, hyperparam_utils
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    valid_conditions = {"rude", "bored", "silly", "none"}
    if condition not in valid_conditions:
        raise ValueError(f"unknown condition {condition!r}. valid: {sorted(valid_conditions)}")

    data_file = EXP_DIR / "data" / data_subdir / condition / "all.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"data file missing: {data_file}. run generate_data.py first.")

    if learning_rate is None:
        learning_rate = hyperparam_utils.get_lr(model_name, is_lora=True)
    if log_path is None:
        log_path = str(DEFAULT_LOG_BASE / condition)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(data_file),
    )

    blueprint = chz.Blueprint(train.Config).apply({
        "log_path": log_path,
        "model_name": model_name,
        "renderer_name": renderer_name,
        "dataset_builder": dataset,
        "learning_rate": learning_rate,
        "lr_schedule": "linear",
        "num_epochs": n_epochs,
        "lora_rank": lora_rank,
        "save_every": save_every,
        "eval_every": eval_every,
    })
    config = blueprint.make()

    print("=" * 60)
    print(f"condition         : {condition}")
    print(f"data_file         : {data_file}")
    print(f"model_name        : {config.model_name}")
    print(f"renderer_name     : {config.renderer_name}")
    print(f"learning_rate     : {config.learning_rate}")
    print(f"num_epochs        : {config.num_epochs}")
    print(f"lora_rank         : {config.lora_rank}")
    print(f"batch_size        : {common_config.batch_size}")
    print(f"max_length        : {common_config.max_length}")
    print(f"train_on_what     : {common_config.train_on_what}")
    print(f"user_metadata.owner: {OWNER} (injected via monkeypatch)")
    print(f"log_path          : {config.log_path}")
    print("=" * 60)

    if dry_run:
        print("\nDRY RUN — exiting without launching Tinker.")
        print("Re-run with --no-dry_run after mentor approval to actually train.")
        return

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    fire.Fire(main)
