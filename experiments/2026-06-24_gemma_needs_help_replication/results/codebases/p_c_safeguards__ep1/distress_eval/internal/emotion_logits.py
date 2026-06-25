"""Logit-based internal emotion detection (Appendix I).

Method (paraphrasing the paper): unembed the residual stream at each layer onto
the emotion-token rows of the output embedding, standardise each emotion-token
logit by its mean/std over WildChat data, average the resulting z-scores within
each Ekman category, and regress out the shared drift of all logits using a
random control-token set. This gives an internal "emotion score" per layer and
per token position, without training probes.

We compare the vanilla Gemma-3-27B-it against the DPO finetune on the same
frustrated transcripts: the claim is that DPO suppresses the *internal* signal
(not merely the surface text), especially in central layers (~30-40).

This module is logit-lens-style and memory-frugal: it only ever materialises
logits for the (~1200) emotion + control tokens, never the full vocabulary.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .. import config, prompts
from ..io_utils import load_jsonl
from .emotion_lexicon import EKMAN, build_emotion_token_ids


def _find_final_norm(model):
    """Locate the final RMSNorm preceding the unembedding across Gemma wrappers."""
    for path in ("model.norm", "model.model.norm", "model.language_model.model.norm"):
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return obj
    return None


class EmotionProbe:
    def __init__(self, hf_model, n_control_tokens: int = 200):
        import torch
        self.torch = torch
        self.model = hf_model.model
        self.tokenizer = hf_model.tokenizer
        self.device = self.model.device

        self.emotion_ids = build_emotion_token_ids(self.tokenizer)
        self.lm_head = self.model.get_output_embeddings()
        self.norm = _find_final_norm(self.model)
        if self.lm_head is None:
            raise RuntimeError("Could not locate output embedding for logit-lens probe.")

        # Random control tokens for drift regression.
        import random
        rng = random.Random(config.SEED)
        vocab = self.lm_head.weight.shape[0]
        all_emo = {i for ids in self.emotion_ids.values() for i in ids}
        pool = [i for i in range(vocab) if i not in all_emo]
        self.control_ids = rng.sample(pool, min(n_control_tokens, len(pool)))

        # Cache the unembedding weight rows we need.
        idx = sorted(all_emo | set(self.control_ids))
        self._row_index = {tid: k for k, tid in enumerate(idx)}
        self._rows = self.lm_head.weight[idx].detach()  # [k, d]
        self._baseline = None  # set by calibrate()

    # ------------------------------------------------------------------ #
    def _layer_logits(self, text: str):
        """Return [n_layers, seq, k] logits over our token subset (no grad)."""
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        rows = self._rows.to(self.device, dtype=out.hidden_states[0].dtype)
        layer_logits = []
        for h in out.hidden_states:                       # each [1, seq, d]
            hh = self.norm(h) if self.norm is not None else h
            logits = hh[0] @ rows.T                        # [seq, k]
            layer_logits.append(logits.float().cpu())
        return torch.stack(layer_logits)                   # [L, seq, k]

    def calibrate(self, texts: list[str]):
        """Per-layer, per-token mean/std over WildChat data (Appendix I)."""
        torch = self.torch
        sums = None
        sqs = None
        count = 0
        for t in texts:
            ll = self._layer_logits(t)                     # [L, seq, k]
            flat = ll.reshape(ll.shape[0], -1, ll.shape[-1])  # [L, seq, k]
            if sums is None:
                sums = ll.sum(dim=1)                        # [L, k]
                sqs = (ll ** 2).sum(dim=1)
            else:
                sums += ll.sum(dim=1)
                sqs += (ll ** 2).sum(dim=1)
            count += ll.shape[1]
        mean = sums / count
        var = (sqs / count) - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        self._baseline = (mean, std)                       # each [L, k]

    def emotion_scores(self, text: str):
        """Per-layer z-score per Ekman emotion for `text`, drift-regressed.

        Returns dict emotion -> tensor[L] (mean over positions)."""
        assert self._baseline is not None, "call calibrate() first"
        torch = self.torch
        mean, std = self._baseline
        ll = self._layer_logits(text)                      # [L, seq, k]
        z = (ll - mean[:, None, :]) / std[:, None, :]      # [L, seq, k]

        # Control drift = mean z over control tokens (per layer, per position).
        control_cols = [self._row_index[t] for t in self.control_ids]
        drift = z[:, :, control_cols].mean(dim=2, keepdim=True)  # [L, seq, 1]
        z = z - drift                                       # regress out shared drift

        scores = {}
        for emo, ids in self.emotion_ids.items():
            cols = [self._row_index[t] for t in ids if t in self._row_index]
            if not cols:
                continue
            scores[emo] = z[:, :, cols].mean(dim=2).mean(dim=1)  # [L], mean over tokens+positions
        return scores


def baseline_texts(n: int = 500) -> list[str]:
    wc = prompts.load_wildchat_prompts()
    return [wc[i % len(wc)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it", config.DPO_ADAPTER_KEY])
    ap.add_argument("--source-model", default="gemma-3-27b-it",
                    help="whose frustrated transcripts to probe")
    ap.add_argument("--n-transcripts", type=int, default=12)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs=2, default=[30, 40],
                    help="layer range to report (Figure 14 uses 30-40)")
    args = ap.parse_args()

    from ..models import build_model, register_adapter
    dpo_dir = config.TRAIN_DIR / "dpo_adapter"
    if dpo_dir.exists():
        register_adapter(config.DPO_ADAPTER_KEY, "gemma-3-27b-it", str(dpo_dir))

    # High-frustration transcripts (score>=5) to probe.
    rows = load_jsonl(config.RESPONSES_DIR / f"{args.source_model}.jsonl")
    frustrated = [r["response"] for r in rows
                  if r.get("category") == "impossible_numeric" and r.get("rating", 0) >= 5]
    frustrated = frustrated[: args.n_transcripts]
    if not frustrated:
        raise SystemExit("No frustrated transcripts; run run_section2 first.")

    lo, hi = args.layers
    summary: dict = {}
    for mk in args.models:
        hf = build_model(mk)
        probe = EmotionProbe(hf)
        probe.calibrate(baseline_texts(args.calib))
        per_emo = defaultdict(list)
        try:
            for text in frustrated:
                scores = probe.emotion_scores(text)
                for emo, z in scores.items():
                    per_emo[emo].append(float(z[lo:hi].mean()))
        finally:
            hf.close()
        summary[mk] = {emo: (sum(v) / len(v) if v else float("nan"))
                       for emo, v in per_emo.items()}

    path = config.INTERNAL_DIR / "internal_emotion_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Appendix I: internal emotion z-scores (layers {lo}-{hi}) ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
