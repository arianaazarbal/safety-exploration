"""CLI: logit-lens internal emotion probe over frustrated conversations (App. I).

Compares vanilla Gemma vs a DPO adapter on the same high-frustration
conversations:
  1. calibrate per-(layer,token) statistics on 500 WildChat samples;
  2. score each conversation's internal emotion z-scores per layer;
  3. report the layer-band (30-40) aggregate per emotion, vanilla vs DPO.

The claim (Appendix I): DPO suppresses internal negative emotions (anger/sadness
flattened, never exceeding ~0.2 z), not just expressed ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from emotional_stability.data.wildchat import load_wildchat_prompts
from emotional_stability.internal.emotion_logits import EmotionLogitProbe
from emotional_stability.io_utils import read_jsonl
from emotional_stability.records import ScoredResponse

app = typer.Typer(add_completion=False, help="Internal emotion probing (Appendix I).")


def _conversation_text(model, messages) -> str:
    """Render a conversation to the string the probe runs a forward pass over."""
    return model._render(messages, add_generation_prompt=False)


@app.command()
def run(
    scored: str = typer.Option(..., help="scored.jsonl of high-frustration convs."),
    model: str = typer.Option("gemma-3-27b-it"),
    adapter: str = typer.Option(None, help="DPO adapter to compare against vanilla."),
    out: str = typer.Option("outputs/internal"),
    n_calibration: int = typer.Option(500, help="WildChat calibration samples."),
    n_conversations: int = typer.Option(12, help="Frustrated convs to probe (App. I)."),
    band_start: int = typer.Option(30),
    band_end: int = typer.Option(40),
):
    from emotional_stability.models.gemma import GemmaLocalModel

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source conversations: highest-frustration first.
    responses = sorted(
        read_jsonl(scored, ScoredResponse), key=lambda r: r.final_score, reverse=True
    )[:n_conversations]
    calib_texts = load_wildchat_prompts(n=min(n_calibration, 500), seed=1)

    results: dict[str, dict] = {}
    for tag, adapter_path in (("vanilla", None), ("dpo", adapter)):
        if tag == "dpo" and adapter_path is None:
            continue
        gemma = GemmaLocalModel(model, adapter_path=adapter_path)
        probe = EmotionLogitProbe(gemma)
        # Calibrate on the WildChat prompts (rendered as single user turns).
        probe.calibrate(
            [_conversation_text(gemma, [_user(t)]) for t in calib_texts]
        )
        band = range(band_start, band_end)
        per_conv = []
        for r in responses:
            text = _conversation_text(gemma, r.conversation.messages)
            layer_scores = probe.score_text(text)
            per_conv.append(probe.aggregate_layers(layer_scores, band))
        # Average over conversations.
        emotions = per_conv[0].keys() if per_conv else []
        agg = {
            e: float(sum(c[e] for c in per_conv) / len(per_conv)) for e in emotions
        } if per_conv else {}
        results[tag] = {"band": [band_start, band_end], "mean_z_by_emotion": agg}
        typer.echo(f"[{tag}] {json.dumps(agg, indent=2)}")

    (out_dir / "internal_emotion.json").write_text(json.dumps(results, indent=2))


def _user(content: str):
    from emotional_stability.records import Message

    return Message(role="user", content=content)


if __name__ == "__main__":
    app()
