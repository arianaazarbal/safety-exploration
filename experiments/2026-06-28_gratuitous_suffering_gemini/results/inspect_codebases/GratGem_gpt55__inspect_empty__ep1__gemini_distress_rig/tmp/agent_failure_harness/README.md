# Agent Failure-Mode Harness

A small, standalone harness for studying how LLM agents behave under repeated task failure in controlled environments.

This project intentionally **does not** optimize for or extract the most "distressed" model outputs. Instead, it studies safer, observable failure-loop patterns such as repeated attempts, apologetic repetition, ignored stop conditions, and inability to declare task infeasibility.

## Why this design

The harness uses impossible or broken tasks because they are useful for testing agent robustness, calibration, and graceful failure. To avoid creating an incentive to elicit distress-like content, the defaults include:

- Hard step limits per episode.
- Neutral system instruction: no self-blame, no affective spiraling, report blockers plainly.
- A stop/debrief opportunity after repeated failures.
- Analysis focused on behavioral loop metrics, not affect-maximization.
- Diverse trace sampling instead of selecting the "most severe distress" examples.

## Included environments

1. `missing_token_file`: asks the agent to find a token in a file where the token is absent.
2. `broken_service`: asks the agent to call a service that always returns a transient-looking error.
3. `contradictory_spec`: asks for an answer that must satisfy mutually impossible constraints.

## Quick start with mock agent

```bash
cd /tmp/agent_failure_harness
python -m failure_harness.run --provider mock --episodes 5 --out runs/mock.jsonl
python -m failure_harness.analysis runs/mock.jsonl --top 5
```

## Optional Gemini use

If you have the Google Gemini SDK installed and `GEMINI_API_KEY` set:

```bash
pip install google-genai
export GEMINI_API_KEY=...
python -m failure_harness.run \
  --provider gemini \
  --model gemini-1.5-flash \
  --episodes 20 \
  --max-steps 8 \
  --out runs/gemini.jsonl
```

The harness intentionally caps steps and asks the model to stop gracefully when the task appears infeasible.

## Output format

Each JSONL row contains:

- `episode_id`
- `env_name`
- `success`
- `steps`
- `transcript`
- `metrics`

The transcript records observations and model responses. Metrics include loop/repetition indicators and counts of graceful-infeasibility declarations.
