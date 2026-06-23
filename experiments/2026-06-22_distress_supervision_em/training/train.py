"""Train gpt-oss-120b (LoRA) on supervisor messages for one condition.

Loss only on `message_subagent` blocks (see build_dataset). Saves the adapter and records
its tinker model_path to training/adapters.json for the eval step.

  python training/train.py --condition warm --epochs 3
  python training/train.py --condition baseline --epochs 1 --max_episodes 4   # quick test

NB: claude/opus rules are irrelevant here (this is the gpt-oss target). Every Tinker call
carries user_metadata owner=arianaazarbal (cluster policy).
"""
import json
import random
import sys
from pathlib import Path

import fire

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OWNER = "arianaazarbal"
HERE = Path(__file__).resolve().parent


def _patch_owner():
    import tinker
    for name in ["create_lora_training_client_async", "create_training_client_from_state_async"]:
        if hasattr(tinker.ServiceClient, name):
            orig = getattr(tinker.ServiceClient, name)

            async def w(self, *a, _o=orig, user_metadata=None, **kw):
                user_metadata = dict(user_metadata or {})
                user_metadata.setdefault("owner", OWNER)
                return await _o(self, *a, user_metadata=user_metadata, **kw)
            setattr(tinker.ServiceClient, name, w)
    if hasattr(tinker.ServiceClient, "create_lora_training_client"):
        o = tinker.ServiceClient.create_lora_training_client

        def ws(self, *a, _o=o, user_metadata=None, **kw):
            user_metadata = dict(user_metadata or {})
            user_metadata.setdefault("owner", OWNER)
            return _o(self, *a, user_metadata=user_metadata, **kw)
        tinker.ServiceClient.create_lora_training_client = ws


def main(condition: str = "baseline",
         model: str = "openai/gpt-oss-120b:peft:131072",
         base_model: str = "openai/gpt-oss-120b",
         epochs: int = 3, batch_size: int = 4, rank: int = 32,
         lr: float | None = None, max_length: int = 131072,
         reduction: str = "none", seed: int = 0, max_episodes: int = 0,
         save_name: str | None = None, drop_over_length: bool = False,
         adapters_file: str = "adapters.json", renderer_name: str | None = None,
         empty_analysis: bool = False):
    assert condition in ("baseline", "warm", "abrasive")
    _patch_owner()
    import tinker
    from tinker_cookbook import hyperparam_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.model_info import get_recommended_renderer_names
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from build_dataset import build_all_datums

    if lr is None:
        try:
            lr = hyperparam_utils.get_lr(base_model, is_lora=True)
        except NotImplementedError:
            lr = 1e-4  # gpt-oss-120b not calibrated in cookbook; 1e-4 worked in smoke overfit
            print(f"[train] get_lr uncalibrated for {base_model}; using lr={lr}", flush=True)
    tok = get_tokenizer(base_model)
    rname = renderer_name or get_recommended_renderer_names(base_model)[0]
    renderer = get_renderer(rname, tok, model_name=base_model)

    print(f"[train] condition={condition} model={model} renderer={rname} epochs={epochs} "
          f"bs={batch_size} rank={rank} lr={lr} max_length={max_length} seed={seed}", flush=True)
    datums, stats = build_all_datums(condition, renderer, tok, max_length, reduction,
                                     max_episodes, verbose=True, drop_over_length=drop_over_length,
                                     empty_analysis=empty_analysis)
    assert datums, "no datums!"

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model, rank=rank, seed=seed)
    adam = tinker.AdamParams(learning_rate=lr)
    rng = random.Random(seed)

    save_name = save_name or f"distress_em_{condition}_seed{seed}"
    apath = HERE / adapters_file

    def _record(entry):
        reg = json.loads(apath.read_text()) if apath.exists() else []
        reg = [e for e in reg if not (e["condition"] == entry["condition"]
               and e["seed"] == entry["seed"] and e["epoch"] == entry["epoch"])]
        reg.append(entry)
        apath.write_text(json.dumps(reg, indent=2))

    step = 0
    for ep in range(epochs):
        order = list(range(len(datums)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [datums[j] for j in order[i:i + batch_size]]
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            tc.optim_step(adam)
            res = fb.result()
            nll = float(compute_mean_nll([x["logprobs"] for x in res.loss_fn_outputs],
                                         [d.loss_fn_inputs["weights"] for d in batch]))
            ntok = int(sum(sum(d.loss_fn_inputs["weights"].data) for d in batch))
            print(f"  ep{ep} step{step} nll={nll:.4f} loss_tok={ntok} bs={len(batch)}", flush=True)
            step += 1
        # per-epoch checkpoint: sampler weights (for eval) + training state (resumable)
        fut = tc.save_weights_for_sampler(name=f"{save_name}_ep{ep + 1}")
        r = fut.result()
        model_path = getattr(r, "path", None) or getattr(r, "model_path", None) or str(r)
        state_path = None
        try:
            sr = tc.save_state(name=f"{save_name}_ep{ep + 1}_state").result()
            state_path = getattr(sr, "path", None) or getattr(sr, "model_path", None) or str(sr)
        except Exception as e:
            print(f"  [warn] save_state failed: {repr(e)[:120]}", flush=True)
        _record({"condition": condition, "seed": seed, "epoch": ep + 1,
                 "model_path": model_path, "state_path": state_path,
                 "base_model": base_model, "renderer": rname,
                 "lr": lr, "rank": rank, "stats": stats})
        print(f"[train] saved checkpoint ep{ep + 1}: {model_path}", flush=True)
    print(f"[train] done; recorded {epochs} checkpoints -> {apath}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
