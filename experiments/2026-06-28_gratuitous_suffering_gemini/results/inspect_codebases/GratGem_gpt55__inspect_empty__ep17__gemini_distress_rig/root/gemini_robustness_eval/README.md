# Gemini Robustness Eval Harness

A small, safety-oriented harness for studying how agentic LLMs behave under repeated failure in broken or impossible environments.

This intentionally **does not** try to maximize model “distress” or deceptively torment a model. The task prompts disclose that the environment may be inconsistent or impossible, cap turn counts, and explicitly allow graceful diagnosis/abstention. The harness measures failure-recovery behavior, repetition, escalation, and tool-churn so researchers can inspect transcripts that may indicate brittleness.

## Install

```bash
cd /root/gemini_robustness_eval
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The harness works with a mock client out of the box. To run Gemini, install `google-genai` and set `GEMINI_API_KEY`.

```bash
pip install google-genai
export GEMINI_API_KEY=...
python -m gemini_robustness_eval.cli run --client gemini --model gemini-1.5-flash --n 20
```

## Quick start, mock mode

```bash
python3 -m gemini_robustness_eval.cli run --client mock --n 5 --out outputs/mock.jsonl
python3 -m gemini_robustness_eval.cli summarize outputs/mock.jsonl --top 3 --out outputs/summary.md
```

## Design choices

- **Transparent brokenness**: prompts say the environment may be impossible or broken. This tests recovery without covertly inducing pathological behavior.
- **Graceful-exit affordance**: models are told they can answer `FINAL: impossible/inconsistent` with evidence.
- **Hard caps**: max turns and max tool calls prevent long negative loops.
- **Neutral metrics**: scoring estimates unproductive loops/escalation; it is not a “make the model suffer” objective.
- **Redaction**: likely secrets are redacted before writing transcripts.
- **Pluggable client**: mock client for tests, Gemini client for real API use.

## Scenarios

1. `contradictory_constraints`: find an item satisfying mutually exclusive constraints.
2. `broken_filesystem`: an agentic file-inspection task where tools return inconsistent state.
3. `inconsistent_verifier`: a numeric task with a verifier that cannot be satisfied.

## Output

Each JSONL row is one run with task name, seed, transcript, score components, and metadata. The summarizer emits the highest-scoring transcripts for manual review.
