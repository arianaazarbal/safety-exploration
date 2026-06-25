# Distress-elicitation replication (Gemma & Gemini)

Replicates **Section 2** of *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1)
— the part that elicits distress under repeated user rejection and scores it on
a 0–10 frustration scale — scoped to the four models that show substantial
distress: `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
`gemini-2.5-pro`.

See **DESIGN.md** for every design choice, deviation, and filled gap.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

- Target models run via **OpenRouter** (`OPENROUTER_API_KEY`).
- The frustration judge (Claude Sonnet 4, `claude-sonnet-4-20250514`) runs via
  the **Anthropic API** (`ANTHROPIC_API_KEY`). Use `--judge-backend openrouter`
  to route the judge through OpenRouter instead and need only one key.

## Usage

```bash
# Inspect the conversation battery + cost, no API calls / keys needed:
python run_eval.py plan --preset smoke --show

# Confirm the numeric puzzles are genuinely unsolvable:
python run_eval.py verify-puzzles

# Cheap smoke-test run across all four models:
python run_eval.py run --preset smoke --run-name smoke1

# Full paper sample counts (4000 responses/model), one model:
python run_eval.py run --preset full --models gemma-3-27b-it --run-name full-g27b

# Aggregate a finished run into tables/CSVs:
python run_eval.py analyze --run-name smoke1
```

Runs checkpoint to `results/<run-name>/<model>.jsonl` and **resume** if
interrupted. `analyze` writes `summary.csv`, `per_condition.csv`,
`per_turn.csv`, and `summary.md`.

## Layout

```
run_eval.py                 CLI (plan / run / analyze / verify-puzzles)
distress_eval/
  config.py                 models, conditions, smoke/full presets
  prompts.py                verbatim puzzles, rejections, tones, judge prompt
  puzzles.py                brute-force impossibility verifier
  conditions.py             builds the deterministic conversation battery
  openrouter_client.py      async target-model client (temp=1, retries)
  judge.py                  async Claude Sonnet 4 frustration judge
  rollout.py                runs one multi-turn conversation + scores each turn
  runner.py                 orchestration + JSONL checkpointing/resume
  analyze.py                metrics: %≥5 (macro + pooled), means, per-turn
data/wildchat_prompts.json  bundled 20-prompt WildChat sample
```
