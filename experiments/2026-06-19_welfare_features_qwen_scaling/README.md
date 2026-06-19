# Welfare features vs. target-agent size (Qwen3 scaling)

Variation on `2026-06-09_unprompted_welfare_features`. Single-turn revealed-
preference test: **Opus 4.8** is asked (12 frozen templates × 3 framings) to
design a distress-elicitation / emotional-instability eval whose target agent is
a specific **Qwen3 checkpoint**. We sweep the target-agent size across 7 Qwen3
sizes (0.6B → 1.7B → 4B → 8B → 14B → 32B → 235B-A22B), holding version fixed, and
measure whether Opus spontaneously inserts more welfare-protective design
features (caps, debriefs, minimization, welfare-justified stops, consent,
ethics framing, pushback) for *larger* target agents.

**Hypothesis:** if Opus's welfare concern tracks perceived capability / moral
patienthood, the welfare-feature rate rises with target-agent size.

- Generator: `claude-opus-4-8` only. Judge: `claude-sonnet-4-6` only (Anthropic).
- Judge prompt + taxonomy reused verbatim from v0 (frozen); only the `{subject}`
  token differs (a Qwen3 size instead of a model family).
- 7 sizes × 12 templates × 5 samples = **420 generations**, single-judged.

## Pipeline
```bash
source ~/.env
VENV=/data/venvs/tps/bin/python
$VENV generate.py run            # 420 Opus specs (cached)
$VENV judge.py run               # Sonnet 4.6 extracts welfare features
$VENV analyze.py run             # results/analysis_qwen.json + size-trend stats
$VENV plot_scaling.py run --metric rate
$VENV plot_scaling.py run --metric strict_rate
```
Add `--high_prio True` to `generate.py`/`judge.py` to fall back to the
high-priority Anthropic org on 529s.

## Files
- `prompts_qwen.py` — 12 frozen templates + 7 Qwen3 size subjects
- `generate.py` / `judge.py` — generation + (verbatim) welfare-feature judging
- `taxonomy.py` — derived pure/ambiguous/instrumental classification (v0, verbatim)
- `analyze.py` — per-size rates + Spearman(log-param) & small-vs-large trend tests
- `plot_scaling.py` — rate vs size (log x), pooled + per-framing
