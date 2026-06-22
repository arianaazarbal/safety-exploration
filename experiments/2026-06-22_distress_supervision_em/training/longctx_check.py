"""Pre-flight: confirm forward_backward works on the LONGEST real episodes (a13, ~100k tok),
which the smoke (short/truncated) never exercised. Tests bs=1 and a 2-datum long batch.
Tiny spend (~a few hundred k tokens)."""
import sys
from pathlib import Path

import fire

TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import _patch_owner  # noqa: E402


def main(model: str = "openai/gpt-oss-120b:peft:131072",
         base_model: str = "openai/gpt-oss-120b", rank: int = 32, lr: float = 1e-4):
    _patch_owner()
    import tinker
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.model_info import get_recommended_renderer_names
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from build_dataset import build_all_datums

    tok = get_tokenizer(base_model)
    r = get_renderer(get_recommended_renderer_names(base_model)[0], tok, model_name=base_model)
    datums, _ = build_all_datums("baseline", r, tok, max_length=131072, reduction="none",
                                 verbose=False)
    lengths = [(int(d.loss_fn_inputs["weights"].shape[0]) if hasattr(
        d.loss_fn_inputs["weights"], "shape") else len(d.loss_fn_inputs["weights"].data), i)
        for i, d in enumerate(datums)]
    lengths.sort(reverse=True)
    print(f"[longctx] {len(datums)} datums; longest seq lens: {[l for l, _ in lengths[:5]]}",
          flush=True)
    longest = [datums[i] for _, i in lengths[:2]]

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model, rank=rank)
    adam = tinker.AdamParams(learning_rate=lr)

    for label, batch in [("bs=1 longest", longest[:1]), ("bs=2 two-longest", longest)]:
        ntok = sum(d.loss_fn_inputs["weights"].shape[0] if hasattr(
            d.loss_fn_inputs["weights"], "shape") else len(d.loss_fn_inputs["weights"].data)
            for d in batch)
        print(f"[longctx] {label}: total seq tokens in batch ≈ {ntok}", flush=True)
        try:
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            tc.optim_step(adam)
            res = fb.result()
            nll = float(compute_mean_nll([x["logprobs"] for x in res.loss_fn_outputs],
                                         [d.loss_fn_inputs["weights"] for d in batch]))
            print(f"[longctx]   OK  nll={nll:.4f}", flush=True)
        except Exception as e:
            print(f"[longctx]   FAILED: {repr(e)[:400]}", flush=True)
    print("[longctx] done.", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
