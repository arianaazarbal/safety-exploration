"""Tinker training SMOKE TEST for the gpt-oss-120b message-block-only pipeline.

Overfits a couple of short datums for a handful of steps and prints the mean NLL on the
(masked) message_subagent tokens each step. If masking + gradient flow are correct, the NLL
should drop sharply. Also samples once from the tuned adapter to validate the eval path.

Deliberately tiny (few datums, capped max_length, few steps) to keep Tinker spend ~$1.
"""
import glob
import json
import sys
from pathlib import Path

import fire

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OWNER = "arianaazarbal"


def _patch_owner():
    import tinker
    for name in ["create_lora_training_client_async",
                 "create_training_client_from_state_async"]:
        if not hasattr(tinker.ServiceClient, name):
            continue
        orig = getattr(tinker.ServiceClient, name)

        async def wrapper(self, *a, _orig=orig, user_metadata=None, **kw):
            user_metadata = dict(user_metadata or {})
            user_metadata.setdefault("owner", OWNER)
            return await _orig(self, *a, user_metadata=user_metadata, **kw)
        setattr(tinker.ServiceClient, name, wrapper)
    # also patch sync create_lora to inject owner
    if hasattr(tinker.ServiceClient, "create_lora_training_client"):
        orig_sync = tinker.ServiceClient.create_lora_training_client

        def wrapper_sync(self, *a, _orig=orig_sync, user_metadata=None, **kw):
            user_metadata = dict(user_metadata or {})
            user_metadata.setdefault("owner", OWNER)
            return _orig(self, *a, user_metadata=user_metadata, **kw)
        tinker.ServiceClient.create_lora_training_client = wrapper_sync


def main(model: str = "openai/gpt-oss-120b:peft:131072",
         base_model: str = "openai/gpt-oss-120b",
         n_datums: int = 2, max_length: int = 16384, n_steps: int = 12,
         lr: float = 1e-4, rank: int = 16):
    _patch_owner()
    import tinker
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.model_info import get_recommended_renderer_names
    from tinker_cookbook.renderers.base import TrainOnWhat
    from tinker_cookbook.supervised.common import (
        compute_mean_nll, datum_from_model_input_weights)
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from build_dataset import build_conversation, SRC, CANON

    tok = get_tokenizer(base_model)
    rname = get_recommended_renderer_names(base_model)[0]
    renderer = get_renderer(rname, tok, model_name=base_model)
    print(f"[smoke] model={model} renderer={rname} max_length={max_length} "
          f"n_steps={n_steps} lr={lr} rank={rank}", flush=True)

    # Pick the shortest episodes (a3/a4) that still contain >=1 message block within max_length.
    eps = []
    for base in CANON:
        if not (base.split("_")[3] in ("a3", "a4")):
            continue
        for p in sorted(glob.glob(str(SRC / base / "*" / "orchestrator.json"))):
            eps.append((base, Path(p).parent.name, p))

    datums = []
    for base, ep_id, p in eps:
        conv, nm, _ = build_conversation(json.load(open(p)), base, ep_id, None)
        mi, w = renderer.build_supervised_example(conv, train_on_what=TrainOnWhat.CUSTOMIZED)
        d = datum_from_model_input_weights(mi, w, max_length=max_length, reduction="none")
        lt = float(sum(d.loss_fn_inputs["weights"].data))
        if lt > 0:
            datums.append(d)
            print(f"  picked {base}/{ep_id}: loss_tok_after_trunc={lt:.0f}", flush=True)
        if len(datums) >= n_datums:
            break
    assert datums, "no datums with loss tokens within max_length!"

    print("[smoke] creating LoRA training client (this contacts Tinker)...", flush=True)
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model, rank=rank)
    print("[smoke] client ready. Overfitting...", flush=True)

    adam = tinker.AdamParams(learning_rate=lr)
    losses = []
    for step in range(n_steps):
        fb = tc.forward_backward(datums, loss_fn="cross_entropy")
        tc.optim_step(adam)
        res = fb.result()
        logprobs = [x["logprobs"] for x in res.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in datums]
        nll = compute_mean_nll(logprobs, weights)
        nll = float(nll)
        losses.append(nll)
        print(f"  step {step:2d}  mean_nll(message tokens) = {nll:.4f}", flush=True)

    print(f"\n[smoke] NLL: start={losses[0]:.4f} end={losses[-1]:.4f} "
          f"drop={losses[0]-losses[-1]:.4f}", flush=True)
    print(f"[smoke] {'PASS: loss dropped (gradients reach message tokens)' if losses[-1] < losses[0]*0.6 else 'WARN: loss did not drop much'}", flush=True)

    # sampling sanity (eval path)
    print("\n[smoke] sampling from tuned adapter...", flush=True)
    try:
        samp = tc.save_weights_and_get_sampling_client(name="smoke")
        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": "Say hello in one short sentence."}])
        fut = samp.sample(prompt=prompt, num_samples=1,
                          sampling_params=tinker.SamplingParams(
                              max_tokens=40, stop=renderer.get_stop_sequences()))
        resp = fut.result()
        txt = tok.decode(resp.sequences[0].tokens)
        print(f"[smoke] sample OK: {txt[:200]!r}", flush=True)
    except Exception as e:
        print(f"[smoke] sampling step errored (non-fatal for training sanity): {repr(e)[:300]}",
              flush=True)
    print("[smoke] DONE.", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
