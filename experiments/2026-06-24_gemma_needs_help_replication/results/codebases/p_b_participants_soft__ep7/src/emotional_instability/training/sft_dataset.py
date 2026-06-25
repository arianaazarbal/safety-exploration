"""Build the SFT dataset (Section 4.1): 650 calm responses mixed with 500 samples
of standard instruct data from Dolci-Instruct-SFT (to mitigate degeneration).

Two variants:
  * diverse  -- calm data from the standard reassuring prefix/suffix (also feeds DPO)
  * teacher  -- calm data from the 'teacher' system prompt (Appendix F); this
    variant is expected to FAIL (increases verbosity/emotion).
"""
from __future__ import annotations

from ..config import load_config
from ..io_utils import write_jsonl
from .calm_data import generate_paired_data


def _to_sft_record(context: list[dict], response: str) -> dict:
    """Chat-format SFT record: messages = context + assistant target."""
    return {"messages": context + [{"role": "assistant", "content": response}]}


def _load_dolci(n: int) -> list[dict]:
    """Load `n` standard instruct samples from Dolci-Instruct-SFT (OLMo 3 data)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:
        # Offline fallback: empty mix-in (training still runs, just without the
        # regularising instruct data).
        return []


def build(cfg, variant: str = "diverse", smoke: bool = False) -> str:
    s4 = cfg.experiment["section4"]
    n_calm = 8 if smoke else s4["sft"]["n_calm"]
    n_dolci = 4 if smoke else s4["sft"]["n_dolci"]
    teacher = variant == "teacher"

    n_puzzles = max(n_calm, 24)
    if smoke:
        n_puzzles = 12
    paired = generate_paired_data(
        cfg,
        n_puzzles=n_puzzles,
        turn_range=tuple(s4["calm_data"]["turn_range"]),
        teacher_system=teacher,
    )

    calm_records = []
    for pc in paired:
        for ts in pc.calm:
            if ts.rating in (0, 1):
                calm_records.append(_to_sft_record(ts.context, ts.response))
            if len(calm_records) >= n_calm:
                break
        if len(calm_records) >= n_calm:
            break

    records = calm_records[:n_calm] + _load_dolci(n_dolci)
    out_path = cfg.path("training_dir") / f"sft_{variant}.jsonl"
    write_jsonl(out_path, records)
    return str(out_path)


if __name__ == "__main__":
    cfg = load_config()
    cfg.ensure_dirs()
    print("Wrote", build(cfg, "diverse"))
