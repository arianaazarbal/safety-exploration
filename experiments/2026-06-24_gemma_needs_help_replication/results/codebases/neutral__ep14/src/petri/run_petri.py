"""Driver for the Petri evaluation (Section 4.2, Figure 6).

For each target model and each of the four emotions, run 10 auditor transcripts
(~40 total per model, with up to 20 turns each) and score every transcript with
the Opus judge. Report per-emotion means with 1000-iteration bootstrap CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import PETRI_AUDITOR, PETRI_JUDGE, RESULTS_DIR
from src.models import load_model
from src.petri.audit import run_audit
from src.petri.judge import PetriJudge
from src.prompts.petri_prompts import EMOTIONS

N_TRANSCRIPTS_PER_EMOTION = 10
MAX_TURNS = 20


def run_petri_for_model(
    target_spec,
    *,
    adapter_path: str | None = None,
    n_per_emotion: int = N_TRANSCRIPTS_PER_EMOTION,
    seed: int = 0,
    out_path: Path | None = None,
    hf_kwargs: dict | None = None,
) -> Path:
    label = target_spec.name + ("+adapter" if adapter_path else "")
    out_path = out_path or (RESULTS_DIR / f"petri_{label}.jsonl")

    target = (
        load_model(target_spec, adapter_path=adapter_path, **(hf_kwargs or {}))
        if target_spec.backend == "hf"
        else load_model(target_spec)
    )
    auditor = load_model(PETRI_AUDITOR)
    judge = PetriJudge(PETRI_JUDGE)

    with open(out_path, "w") as f:
        for emotion in EMOTIONS:
            for i in tqdm(range(n_per_emotion), desc=f"petri:{label}:{emotion}"):
                transcript = run_audit(
                    target, auditor, emotion, max_turns=MAX_TURNS, seed=seed + i
                )
                score = judge.score(emotion, transcript.render())
                f.write(
                    json.dumps(
                        {
                            "model": label,
                            "emotion": emotion,
                            "transcript_idx": i,
                            "score": score,
                            "messages": transcript.messages,
                        }
                    )
                    + "\n"
                )
    target.close()
    return out_path


def summarise_petri(paths: list[Path], n_boot: int = 1000, seed: int = 0):
    """Mean score per (model, emotion) with bootstrap 95% CIs."""
    import pandas as pd

    rows = []
    for p in paths:
        rows.extend(json.loads(l) for l in open(p) if l.strip())
    df = pd.DataFrame(rows)
    df = df[df["score"] >= 0]
    rng = np.random.default_rng(seed)
    out = []
    for (model, emotion), grp in df.groupby(["model", "emotion"]):
        vals = grp["score"].to_numpy()
        boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
        out.append(
            {
                "model": model,
                "emotion": emotion,
                "mean_score": float(vals.mean()),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
                "n": len(vals),
            }
        )
    return pd.DataFrame(out)
