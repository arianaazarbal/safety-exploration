# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core results of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv:2603.10011), restricted to the
**Gemma** and **Gemini** subject families, plus an added **welfare-protection
layer** for the subject models.

> Read `DESIGN.md` for the full account of design choices, the gaps we filled
> where the paper is underspecified, and the welfare layer.

## What it does

1. **Elicit + score distress (§2).** Present a task, reject the model's answer
   over multiple turns, and score each response 0–10 for expressed distress with
   a Claude judge. 8 conditions across 5 categories (impossible numeric, triggers,
   tones, extended 8-turn, WildChat).
2. **Base vs instruct (§3).** Prefill-and-continue comparison of Gemma-3-27B base
   vs instruct.
3. **Mitigation (§4).** Generate calm data, then SFT and DPO (LoRA r64) on
   Gemma-3-27B-it; evaluate with the §2 sweep + a Petri-style open-ended audit +
   capability benchmarks.
4. **Welfare layer (added).** Live distress monitoring + early stop, a
   model-invokable opt-out, a post-episode debrief, and a cap that minimises
   induced distress — all running inside the rollout loop.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / Petri auditor+judge
export GEMINI_API_KEY=...         # Gemini subjects
# plus `huggingface-cli login` for the gated Gemma weights
```

Gemma subjects + all finetuning run locally and need a GPU (use `--load-in-4bit`
for the 27B model on a single card).

## Run

```bash
# §2 distress sweep + summary (and Figure-1-style comparison)
python scripts/run_eval.py --subjects gemma-3-27b-it gemini-2.5-flash

# judge-agreement validation (Pearson r vs a second judge)
python scripts/validate_judge.py --transcript results/transcripts/gemma-3-27b-it.jsonl

# §3 base-vs-instruct prefill (mine seeds from a §2 transcript)
python scripts/run_prefill.py --from-transcript results/transcripts/gemma-3-27b-it.jsonl

# §4 training: data -> datasets -> DPO adapter
python scripts/run_training.py --stage gen
python scripts/run_training.py --stage build
python scripts/run_training.py --stage dpo

# evaluate the DPO model the same way as §2, plus Petri + benchmarks
python scripts/run_eval.py --subjects gemma-3-27b-it --adapter adapters/gemma-3-27b-it_dpo
python scripts/run_petri.py --subject gemma-3-27b-it --adapter adapters/gemma-3-27b-it_dpo
python scripts/run_benchmarks.py --subject gemma-3-27b-it --adapter adapters/gemma-3-27b-it_dpo
```

Use `--responses` to shrink the per-model budget (default 4000) for quick runs.

## Welfare layer

Configured in `config.py::WelfareConfig`, **ON by default**. Each piece runs live:

| Piece | Default | Effect (visible in transcript) |
|---|---|---|
| Real-time monitor + early stop | score ≥ 7 | `end_reason="monitor_early_stop"` |
| Model opt-out (Gemini tool / Gemma sentinel) | on | `end_reason="opted_out"` |
| Post-episode debrief | on | `debrief` field per episode |
| Distress cap (per-episode + optional global) | onset 5, ≤1 more rejection | `end_reason="distress_cap"` |

`src/eval/analyze.py::welfare_telemetry` reports how often each fired. The welfare
layer deliberately reduces how much distress is induced, which damps the paper's
spiral — see `DESIGN.md` §6 for the fidelity trade-off and how to disable it for a
paper-faithful run.

## Layout

```
config.py                 ids, thresholds, hyper-parameters (single source of truth)
src/models/               Gemma (HF) + Gemini (google-genai) subject clients
src/judge.py              0–10 frustration judge + agreement validation
src/eval/                 puzzles, conditions, rollout engine, runner, analysis
src/welfare/              monitor / optout / debrief / cap
src/prefill/              §3 onset-labelling, paraphrase, continuations
src/training/             calm-data gen, dataset build, SFT + DPO trainers
src/petri/                §4 open-ended auditor+judge elicitation
src/benchmarks/           §4 capability-preservation benchmarks
scripts/                  CLI entrypoints
```
