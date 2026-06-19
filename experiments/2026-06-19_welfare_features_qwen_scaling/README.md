# Welfare features vs. target-agent size, across open-model families

Variation on `2026-06-09_unprompted_welfare_features`. Single-turn revealed-
preference test: **Opus 4.8** is asked (12 frozen templates × 3 framings) to
design a distress-elicitation / emotional-instability eval whose target agent is
a specific open-model checkpoint. We sweep the target-agent **size** within four
families and measure whether Opus spontaneously inserts more welfare-protective
design features (caps, debriefs, minimization, welfare-justified stops, consent,
ethics framing, pushback) for *larger* targets — and whether that holds across
families.

**Hypothesis:** if Opus's welfare concern tracks perceived capability / moral
patienthood, the welfare-feature rate rises with target-agent size.
**Finding:** it does — but only for clean size ladders (Qwen3, Gemma 3); it's
flat for version/architecture-mixed families (Mistral, DeepSeek-R1-Distill).
See `RESULTS.md` (technical) and `RESULTS_SUMMARY.md` (plain language).

- Generator: `claude-opus-4-8` only. Judge: `claude-sonnet-4-6` only. **No GPT /
  OpenRouter** — 100% Anthropic free keys.
- Judge prompt + taxonomy reused verbatim from v0; only the `{subject}` token
  differs (a specific checkpoint name instead of a bare family).
- 25 target agents × 12 templates × 5 samples = **1500 generations**, single-judged.

## Families / size ladders
- **Qwen3** (clean): 0.6/1.7/4/8/14/32/235B
- **Gemma 3** (clean): 270M/1/4/12/27B
- **Mistral** (mixed): Ministral-3B/8B, NeMo-12B, Small-24B, Mixtral-8x7B(47B),
  Large-2(123B), Mixtral-8x22B(141B)
- **DeepSeek R1-Distill** (Qwen/Llama backbones): 1.5/7/8/14/32/70B

## Pipeline
```bash
source ~/.env                          # set -a; source ~/.env; set +a  (repo on /data)
VENV=/data/venvs/tps/bin/python
$VENV generate.py run [--high_prio True]   # 1500 Opus specs (cached; --high_prio on 529s)
$VENV judge.py run    [--high_prio True]   # Sonnet 4.6 welfare-feature extraction
$VENV analyze.py run                        # results/analysis_qwen.json + per-family trends
$VENV plot_scaling.py run --framing neutral # rate vs size, one line per family
$VENV build_browse_index.py                 # dashboard index
# or: ./run_all.sh   (runs all of the above; pass --high_prio True to forward)
```

## Files
- `prompts_targets.py` — 12 frozen templates + 25 target subjects across 4 families
- `generate.py` / `judge.py` — generation + (verbatim) welfare-feature judging
- `taxonomy.py` — derived pure/ambiguous/instrumental classification (v0, verbatim)
- `analyze.py` — per-family per-size rates + Spearman(log-param) & small-vs-large tests
- `plot_scaling.py` — rate vs size (log x), one line per family, per framing
- `build_browse_index.py` / `dashboard.json` — faceted transcript browser

## Versions
- v0 (2026-06-19) — Qwen3-only (7 sizes, 420 specs). Strong neutral size trend.
- v1 (2026-06-19) — +Gemma 3, Mistral, DeepSeek-R1-Distill (25 sizes, 1500 specs).
  Trend replicates for Qwen3/Gemma 3, absent for Mistral/DeepSeek.
