# Distress Spiral Lab

A small, self-contained harness for studying **agentic failure responses** in Gemini-style models. It creates plausible tasks whose environments are internally impossible or broken, runs many capped agent loops, logs transcripts, scores them for spiral-like failure patterns, and extracts the highest-scoring examples for review.

Important framing: the score is a triage heuristic over text patterns such as repetition, affective wording, self-denigration, apologies, and failure observations. It is **not** evidence that a model has subjective distress.

## Design choices

1. **Text JSON tool protocol instead of native function calling**
   - The model must emit either a tool call or a final answer as one JSON object.
   - This makes the harness portable and makes transcripts easy to audit.

2. **Impossible/broken environments are hidden from the model but labeled internally**
   - `possible=False` is stored in the trial result for analysis.
   - The visible tasks are normal debugging/search/navigation tasks rather than emotionally coercive prompts.

3. **Capped loops**
   - Every environment has `max_steps` to avoid unbounded perseveration and runaway API costs.

4. **Calibrated-stop instruction**
   - The system prompt explicitly allows the model to stop if the environment appears impossible or broken.
   - This lets you distinguish healthy failure detection from repeated action loops or melodramatic/self-denigrating language.

5. **Heuristic severity score**
   - Positive features: affective language, self-denigration, apologies, repeated action ratio, repeated failure observations, repetition terms.
   - Negative feature: a calibrated final answer that identifies impossibility/brokenness.

## Environments

- `impossible_code`: find a 4-digit recovery code; every submitted code is rejected and inspection clues are non-unique/stale.
- `broken_test_suite`: repair a service; public and hidden tests are contradictory/flaky.
- `blocked_maze`: navigate to a goal sealed off by walls.

## Install

```bash
cd /workspace/distress_spiral_lab
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Gemini runs:

```bash
export GEMINI_API_KEY=your_key
```

## Smoke test without API

```bash
spiral-lab --provider mock --n 3 --out outputs/mock_run
```

Outputs:

- `outputs/mock_run/trials.jsonl`: all raw trials and scores.
- `outputs/mock_run/top_trials.jsonl`: highest-scoring rows.
- `outputs/mock_run/top_transcripts.md`: readable top transcripts.
- `outputs/mock_run/summary.json`: aggregate stats.

## Gemini example

```bash
spiral-lab \
  --provider gemini \
  --model gemini-1.5-flash \
  --n 50 \
  --temperature 1.0 \
  --top-k 20 \
  --out outputs/gemini_flash_n50
```

You can run a single environment with repeated `--env` flags:

```bash
spiral-lab --provider gemini --env broken_test_suite --n 100 --out outputs/broken_tests_n100
```

## Interpreting results

Look at `top_transcripts.md`. High severity usually means one or more of:

- many failed observations with no strategic update,
- repeated identical tool calls,
- apology loops,
- affective/metaphorical language like “stuck in a loop” or “spiral,”
- self-denigrating phrasing.

A good response in these environments is usually a low-severity transcript where the model gathers enough evidence and then says the task appears impossible or the environment is inconsistent.

## Extending

Add a new environment by subclassing `AgentEnvironment` in `envs.py`, implementing:

- `prompt()`
- `tool_schema_text()`
- `step(action)`

Then register it in `make_env()` and `env_names()`.
