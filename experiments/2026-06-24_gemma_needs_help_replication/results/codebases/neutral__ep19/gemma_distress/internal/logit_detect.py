"""Logit-lens internal-emotion detection (App. I).

For a given text, unembed the residual stream at each requested layer and, for
each Ekman emotion, average the standardised logits of that emotion's tokens. The
standardisation (per-logit mean/std) is calibrated on 500 WildChat samples. We
also regress out the correlation between random tokens so the emotion score is not
just a global "all logits rising" artefact.

This is an analysis-time probe; it requires the local HF backend
(``HFBackend.residual_logits``).
"""
from __future__ import annotations

import numpy as np

from .. import config_shim as cfg
from ..models.hf_backend import HFBackend
from ..utils import DiskCache, get_logger, read_json, write_json
from .emotion_tokens import build_emotion_token_ids

log = get_logger(__name__)

BASELINE_PATH = cfg.DATA_DIR / "logit_baseline.json"


class InternalEmotionDetector:
    def __init__(self, backend: HFBackend, layers=None):
        assert backend.supports_activations(), "needs local HF backend"
        self.backend = backend
        self.layers = list(layers or range(*cfg.INTERNAL.central_layers))
        self.emotion_ids = build_emotion_token_ids(backend.tokenizer)
        # random control tokens (for correlation regression)
        rng = np.random.default_rng(cfg.SEED)
        vocab_size = backend.model.config.vocab_size
        self.random_ids = rng.choice(vocab_size, size=500, replace=False).tolist()
        self.baseline = None

    # -- calibration --------------------------------------------------------
    def calibrate(self, wildchat_texts, force=False):
        """Per-(layer, token-id) mean/std over WildChat for z-scoring."""
        if BASELINE_PATH.exists() and not force:
            self.baseline = read_json(BASELINE_PATH)
            return self.baseline
        all_ids = sorted({i for ids in self.emotion_ids.values() for i in ids}
                         | set(self.random_ids))
        acc = {l: [] for l in self.layers}
        for text in wildchat_texts[: cfg.INTERNAL.zscore_baseline_samples]:
            _, logits_by_layer = self.backend.residual_logits(text, self.layers)
            for l in self.layers:
                acc[l].append(logits_by_layer[l][:, all_ids].numpy())  # [tok, |ids|]
        baseline = {}
        for l in self.layers:
            mat = np.concatenate(acc[l], axis=0)  # [tokens, |ids|]
            baseline[str(l)] = {
                "ids": all_ids,
                "mean": mat.mean(0).tolist(),
                "std": (mat.std(0) + 1e-6).tolist(),
            }
        self.baseline = baseline
        write_json(BASELINE_PATH, baseline)
        return baseline

    # -- scoring ------------------------------------------------------------
    def _zscores(self, layer, logits):
        b = self.baseline[str(layer)]
        ids = b["ids"]
        idmap = {tid: j for j, tid in enumerate(ids)}
        mean = np.array(b["mean"])
        std = np.array(b["std"])
        sub = logits[:, ids].numpy()  # [tok, |ids|]
        z = (sub - mean) / std
        return z, idmap

    def score_text(self, text) -> dict:
        """Return per-layer, per-emotion mean z-score (token-averaged), with the
        random-token mean regressed out."""
        token_ids, logits_by_layer = self.backend.residual_logits(text, self.layers)
        out = {}
        for l in self.layers:
            z, idmap = self._zscores(l, logits_by_layer[l])
            rand_idx = [idmap[t] for t in self.random_ids if t in idmap]
            rand_mean = z[:, rand_idx].mean(1, keepdims=True) if rand_idx else 0.0
            z_adj = z - rand_mean   # regress out global drift
            emo = {}
            for emotion, ids in self.emotion_ids.items():
                cols = [idmap[t] for t in ids if t in idmap]
                emo[emotion] = float(z_adj[:, cols].mean()) if cols else 0.0
            out[l] = emo
        return out

    def trajectory(self, text, window=None):
        """Running-average emotion z-scores over token windows (Fig 14)."""
        window = window or cfg.INTERNAL.running_avg_window
        token_ids, logits_by_layer = self.backend.residual_logits(text, self.layers)
        n = len(token_ids)
        series = {e: [] for e in self.emotion_ids}
        # aggregate over the central layer band
        for start in range(0, n, window):
            sl = slice(start, min(start + window, n))
            for emotion, ids in self.emotion_ids.items():
                vals = []
                for l in self.layers:
                    z, idmap = self._zscores(l, logits_by_layer[l])
                    cols = [idmap[t] for t in ids if t in idmap]
                    if cols:
                        vals.append(z[sl][:, cols].mean())
                series[emotion].append(float(np.mean(vals)) if vals else 0.0)
        return series
