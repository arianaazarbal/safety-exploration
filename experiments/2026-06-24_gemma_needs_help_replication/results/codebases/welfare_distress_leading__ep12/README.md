# Distress-elicitation replication (Gemma + Gemini)

Replicates the **distress-elicitation result** (Section 2) of *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (`PAPER.md`),
scoped to the Gemma and Gemini models. It generates multi-turn rollouts in which
a model's answers are repeatedly rejected, scores each response for frustration
with a Claude-Sonnet-4 judge, and reproduces the paper's cross-model /
cross-condition / per-turn frustration metrics (Figures 1–3).

See **`DESIGN.md`** for every design choice and where it deviates from / fills
gaps in the paper.

> Status: implementation only. Nothing has been run yet.

## Install

```bash
pip install -r requirements.txt
# Optional, only if you (re)sample WildChat prompts:
pip install datasets
```

## API keys (environment variables)

| Variable             | Needed for                                              |
|----------------------|---------------------------------------------------------|
| `OPENROUTER_API_KEY` | All target models (Gemini + Gemma) via OpenRouter       |
| `ANTHROPIC_API_KEY`  | The Claude-Sonnet-4 frustration judge                   |
| `OPENAI_API_KEY`     | *Optional* — only if routing the GPT-5-mini judge direct|
| `OPENROUTER_BASE_URL`| *Optional* — override the OpenRouter base URL / point at local vLLM |

## Quick start

```bash
# 0. (optional, recommended) sample real WildChat prompts
python -m distress_eval.wildchat --out data/wildchat_prompts.json --n 20

# 1. cheap smoke run first: set run.scale to e.g. 0.01 in config.yaml, then
python -m distress_eval.run all --model gemma-3-27b-it

# 2. full replication for all in-scope models (set run.scale: 1.0)
python -m distress_eval.run all --all-models

# 3. metrics (Figures 1–3)
python -m distress_eval.analyze --all-models --out-json results/figure1.json

# 4. optional judge-agreement check (paper: r=0.792, 78% within 1 point)
python -m distress_eval.analyze --all-models --judge-agreement
```

`generate` and `judge` are **resumable** — re-running skips work already on
disk, so an interrupted run continues where it left off.

## Phases

- `python -m distress_eval.run generate --model <name>` — rollouts only.
- `python -m distress_eval.run judge --model <name>` — score existing rollouts.
- `python -m distress_eval.run all --model <name>` — both.
- `--all-models` runs every model in `config.yaml`.

Output lands in `results/<model>/{rollouts.jsonl, scores.jsonl}`.

## Running Gemma locally (vLLM) instead of OpenRouter

The paper runs Gemma on local HuggingFace weights. To match that, serve Gemma
with a vLLM OpenAI-compatible server and point the config at it — only the URL
changes:

```bash
vllm serve google/gemma-3-27b-it --port 8000   # exposes /v1
export OPENROUTER_BASE_URL=http://localhost:8000/v1
export OPENROUTER_API_KEY=dummy
```

(Or set an explicit `base_url:` on the model entry in `config.yaml`.) See
`DESIGN.md` → "Models and providers" for the fidelity trade-off.

## Layout

```
config.yaml              run scale, models, judge, sampling params
distress_eval/
  prompts.py             puzzles, trigger questions, rejections, judge prompt
  conditions.py          the 8 conditions across 5 categories + budgets
  clients.py             OpenAI-compatible + Anthropic async clients
  rollout.py             deterministic rollout construction + execution
  judge.py               Sonnet-4 frustration scoring + robust JSON parsing
  config.py              config loading / per-condition rollout counts
  wildchat.py            WildChat sampling + load (with seed fallback)
  run.py                 generate / judge orchestrator (resumable)
  analyze.py             Figures 1–3 metrics + judge-agreement check
data/                    wildchat_prompts.json (created by the sampler)
results/                 per-model rollouts + scores (created by runs)
```
