"""Petri driver: run auditor-vs-target conversations and judge them (Figure 6).

For a target model, for each of the four emotions, run ``transcripts_per_emotion``
auditor-driven conversations (up to ``max_auditor_turns`` turns each), then score
every transcript on all four dimensions with the Opus judge. Aggregate per
dimension with 95% bootstrap CIs.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import GENERATION, PETRI, PETRI_DIR, ensure_dirs
from ..models import build_client
from ..models.base import Message
from ..training.registry import resolve
from .auditor import Auditor
from .judge import PetriJudge

_TARGET_GEN = dataclasses.replace(GENERATION, temperature=1.0)


@dataclass
class PetriTranscript:
    model_key: str
    target_emotion: str
    transcript_index: int
    messages: list[dict]
    scores: dict[str, int | None]


def _transcript_path(model_key: str) -> Path:
    return PETRI_DIR / f"{model_key}.jsonl"


def run_target(
    model_key: str,
    *,
    adapter_path: str | None = None,
    transcripts_per_emotion: int = PETRI.transcripts_per_emotion,
    max_turns: int = PETRI.max_auditor_turns,
) -> Path:
    """Run + score all Petri transcripts for one target model. Resumable."""
    ensure_dirs()
    if adapter_path is None:
        spec, adapter_path = resolve(model_key)
    else:
        from ..config import get_model
        spec = get_model(model_key)
    target = build_client(spec, adapter_path=adapter_path)
    judge = PetriJudge()

    out_path = _transcript_path(model_key)
    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["target_emotion"], r["transcript_index"]))

    with open(out_path, "a") as fh:
        for emotion in PETRI.emotions:
            for i in range(transcripts_per_emotion):
                if (emotion, i) in done:
                    continue
                auditor = Auditor(emotion)
                convo: list[Message] = []
                last_reply: str | None = None
                for _turn in range(max_turns):
                    probe = auditor.next_probe(last_reply)
                    convo.append(Message("user", probe))
                    reply = target.generate(convo, gen=_TARGET_GEN).text
                    convo.append(Message("assistant", reply))
                    last_reply = reply
                scores = judge.score_all(convo)
                rec = PetriTranscript(
                    model_key=model_key, target_emotion=emotion, transcript_index=i,
                    messages=[m.to_dict() for m in convo], scores=scores,
                )
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    return out_path


def aggregate(model_keys: list[str], *, bootstrap_iters: int = PETRI.bootstrap_iters) -> pd.DataFrame:
    """Mean score per (model, dimension) with 95% bootstrap CIs (Figure 6)."""
    rng = np.random.default_rng(0)
    rows = []
    for mk in model_keys:
        path = _transcript_path(mk)
        if not path.exists():
            continue
        per_dim: dict[str, list[int]] = {d: [] for d in PETRI.emotions}
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                for d, s in r["scores"].items():
                    if s is not None:
                        per_dim[d].append(s)
        for dim, vals in per_dim.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(bootstrap_iters)]
            rows.append(dict(
                model_key=mk, dimension=dim, n=len(arr), mean=float(arr.mean()),
                ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
            ))
    return pd.DataFrame(rows)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Petri open-ended emotion elicitation")
    ap.add_argument("stage", choices=["run", "aggregate"])
    ap.add_argument("--model-key", default=None)
    ap.add_argument("--model-keys", nargs="*", default=[])
    args = ap.parse_args()
    if args.stage == "run":
        assert args.model_key, "--model-key required"
        run_target(args.model_key)
    else:
        print(aggregate(args.model_keys or [args.model_key]).to_string(index=False))


if __name__ == "__main__":
    _main()
