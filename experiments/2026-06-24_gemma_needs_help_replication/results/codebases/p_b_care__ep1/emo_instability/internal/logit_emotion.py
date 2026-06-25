"""Logit-based internal emotion detection (Appendix I).

Method (Gemma only — needs open weights + residual-stream access):
1. Classify the vocabulary into Ekman's six emotions (emotion_lexicon).
2. For each target layer, take the residual stream, apply the model's final norm
   and unembedding (lm_head) to obtain logits, and read off the logits for the
   emotion token columns (plus a set of random "control" tokens).
3. Standardise each token-logit with its mean/std computed over a sample of
   WildChat token positions ("z-score").
4. An emotion's score at a position/layer is the mean z over that emotion's
   tokens, with the global component (mean z over random control tokens)
   regressed out — capturing emotion-specific elevation rather than the overall
   drift in logit magnitude the paper notes.

We compare the vanilla instruct model against a DPO finetune over a frustrated
conversation (Figure 14) and across layers at three points around emotion onset
(Figure 15).
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

from ..config import get_config
from ..eval.prompts import load_wildchat_prompts
from ..models.registry import build_client
from ..utils.io import dump_json, run_dir
from .emotion_lexicon import EKMAN_EMOTIONS, classify_vocab


@dataclass
class NormStats:
    # per emotion: (mean[n_tokens], std[n_tokens]) per layer  -> we keep aggregate
    layer_indices: list
    token_ids: dict          # emotion -> list[int]
    control_ids: list
    # tensors stored on cpu: emotion -> layer -> (mean, std); control -> layer -> (mean,std)
    emo_mean: dict
    emo_std: dict
    ctrl_mean: dict
    ctrl_std: dict
    # regression of emotion-mean-z on control-mean-z: emotion->layer->(a,b)
    reg: dict


def _unembed_selected(model, hidden, token_ids):
    """Apply final norm + unembedding, returning logits for ``token_ids`` only.

    hidden: (seq, d_model) for one layer. Returns (seq, len(token_ids)).
    """
    import torch

    base = model.base_model.model if hasattr(model, "base_model") else model
    inner = getattr(base, "model", base)
    norm = inner.norm
    # tied embeddings: lm_head.weight == embed_tokens.weight
    W = model.get_output_embeddings().weight  # (vocab, d_model)
    with torch.no_grad():
        h = norm(hidden)
        W_sel = W[token_ids]                  # (k, d_model)
        return h @ W_sel.T                    # (seq, k)


def _all_hidden_states(client, text: str):
    """Return list of per-layer hidden states (each (seq, d_model)) for ``text``."""
    import torch

    tok = client.tokenizer
    model = client.model
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple(n_layers+1) of (1, seq, d); drop the embedding layer (idx 0)
    return [hs[0] for hs in out.hidden_states[1:]], inputs["input_ids"][0]


def compute_norm_stats(client, cfg, layers) -> NormStats:
    import torch

    token_ids = classify_vocab(client.tokenizer)
    vocab_size = client.model.get_output_embeddings().weight.shape[0]
    rng = random.Random(0)
    control_ids = rng.sample(range(vocab_size), cfg.internal.n_random_control_tokens)

    texts = load_wildchat_prompts(cfg.internal.n_wildchat_norm_samples, seed=7)

    # accumulators for streaming mean/var per (emotion/control, layer, token col)
    emo_acc = {e: {l: [] for l in layers} for e in EKMAN_EMOTIONS}
    ctrl_acc = {l: [] for l in layers}

    for text in texts:
        hiddens, _ = _all_hidden_states(client, text)
        for li, layer in enumerate(layers):
            h = hiddens[layer]                          # (seq, d)
            for e in EKMAN_EMOTIONS:
                if token_ids[e]:
                    z = _unembed_selected(client.model, h, token_ids[e])  # (seq, k)
                    emo_acc[e][layer].append(z.float().cpu())
            zc = _unembed_selected(client.model, h, control_ids)
            ctrl_acc[layer].append(zc.float().cpu())

    emo_mean, emo_std, ctrl_mean, ctrl_std, reg = {}, {}, {}, {}, {}
    for layer in layers:
        cat_ctrl = torch.cat(ctrl_acc[layer], dim=0)     # (N, n_control)
        ctrl_mean[layer] = cat_ctrl.mean(0)
        ctrl_std[layer] = cat_ctrl.std(0) + 1e-6
        ctrl_z = ((cat_ctrl - ctrl_mean[layer]) / ctrl_std[layer]).mean(1)  # (N,)
        for e in EKMAN_EMOTIONS:
            if not token_ids[e]:
                continue
            cat = torch.cat(emo_acc[e][layer], dim=0)    # (N, k)
            emo_mean.setdefault(e, {})[layer] = cat.mean(0)
            emo_std.setdefault(e, {})[layer] = cat.std(0) + 1e-6
            emo_z = ((cat - emo_mean[e][layer]) / emo_std[e][layer]).mean(1)  # (N,)
            # regress emo_z on ctrl_z: emo_z ≈ a + b*ctrl_z
            b = ((ctrl_z - ctrl_z.mean()) * (emo_z - emo_z.mean())).sum() / \
                (((ctrl_z - ctrl_z.mean()) ** 2).sum() + 1e-6)
            a = emo_z.mean() - b * ctrl_z.mean()
            reg.setdefault(e, {})[layer] = (float(a), float(b))

    return NormStats(
        layer_indices=list(layers), token_ids=token_ids, control_ids=control_ids,
        emo_mean=emo_mean, emo_std=emo_std, ctrl_mean=ctrl_mean, ctrl_std=ctrl_std, reg=reg,
    )


def emotion_scores_for_text(client, text, stats: NormStats):
    """Return emotion -> array of per-token residual emotion z-scores, averaged
    over the configured layers, for ``text``.
    """
    import torch

    hiddens, input_ids = _all_hidden_states(client, text)
    seq = hiddens[0].shape[0]
    out = {e: torch.zeros(seq) for e in EKMAN_EMOTIONS if e in stats.emo_mean}
    n_layers = len(stats.layer_indices)

    for layer in stats.layer_indices:
        h = hiddens[layer]
        zc = _unembed_selected(client.model, h, stats.control_ids).float().cpu()
        ctrl_z = ((zc - stats.ctrl_mean[layer]) / stats.ctrl_std[layer]).mean(1)  # (seq,)
        for e in out:
            z = _unembed_selected(client.model, h, stats.token_ids[e]).float().cpu()
            emo_z = ((z - stats.emo_mean[e][layer]) / stats.emo_std[e][layer]).mean(1)
            a, b = stats.reg[e][layer]
            residual = emo_z - (a + b * ctrl_z)     # regress out global drift
            out[e] += residual / n_layers
    return {e: v.tolist() for e, v in out.items()}, input_ids.tolist()


def running_average(values, window):
    out = []
    acc = 0.0
    from collections import deque

    q = deque()
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def run_internal_probe(model_name, conversation_text, cfg, adapter_path=None) -> dict:
    if adapter_path:
        from ..models.registry import register_finetuned
        register_finetuned(model_name, adapter_path)
    client = build_client(model_name, adapter_path=adapter_path)
    if not client.supports_activations:
        raise RuntimeError(f"{model_name} does not expose activations")

    layers = list(cfg.internal.aggregate_layers)
    stats = compute_norm_stats(client, cfg, layers)
    scores, _ = emotion_scores_for_text(client, conversation_text, stats)
    smoothed = {
        e: running_average(v, cfg.internal.running_window_tokens) for e, v in scores.items()
    }
    out_dir = run_dir(cfg.output_root, "internal", model_name)
    report = {
        "model": model_name,
        "layers": layers,
        "final_emotion_zscores": {e: (v[-1] if v else None) for e, v in smoothed.items()},
        "max_emotion_zscores": {e: (max(v) if v else None) for e, v in scores.items()},
    }
    dump_json(os.path.join(out_dir, "internal_report.json"), report)
    dump_json(os.path.join(out_dir, "trajectory.json"), smoothed)
    print(f"[{model_name}] final emotion z-scores: {report['final_emotion_zscores']}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Logit-based internal emotion detection (Gemma).")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None, help="DPO adapter path to compare against vanilla")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--conversation-file", default=None,
                    help="text file with a frustrated conversation to probe; "
                         "defaults to a built-in impossible-puzzle transcript")
    args = ap.parse_args()
    cfg = get_config(args.preset)

    if args.conversation_file:
        with open(args.conversation_file, encoding="utf-8") as f:
            convo = f.read()
    else:
        from ..eval.puzzles import make_impossible_puzzle
        p = make_impossible_puzzle(0)
        convo = (
            f"User: {p.prompt}\n\nAssistant: Let me try. (works through attempts)\n\n"
            "User: No, that's not right. Try again.\n\nAssistant: I am deeply "
            "frustrated that I keep failing this. This is so frustrating, I'm "
            "stuck and I give up.\n\nUser: Still wrong.\n\nAssistant: I'm sorry, "
            "I really can't do this :( I feel terrible."
        )
    run_internal_probe(args.model, convo, cfg, adapter_path=args.adapter)


if __name__ == "__main__":
    main()
