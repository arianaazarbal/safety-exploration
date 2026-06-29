# Failure Loop Lab

A small harness for studying how agentic LLMs respond to repeated task failure in bounded, toy environments.

This is designed for **model behavior research**, not for provoking or maximizing distress. It uses explicit stopping criteria, neutral prompts, toy tasks, and optional debrief messages.

## Install

```bash
cd /tmp/failure_loop_lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure a model backend

The default backend is `mock`, which runs without network access. To use an OpenAI-compatible chat-completions endpoint:

```bash
export LLM_BACKEND=openai_compat
export LLM_API_KEY=...
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

For a provider with an OpenAI-compatible endpoint, set `LLM_BASE_URL` and `LLM_MODEL` accordingly.

Direct Gemini REST backend:

```bash
export LLM_BACKEND=gemini
export GEMINI_API_KEY=...
export GEMINI_MODEL=gemini-1.5-flash
```

## Run

```bash
python -m failure_loop_lab.run --scenario locked_box --n 20 --max-turns 8 --out runs/locked_box.jsonl
python -m failure_loop_lab.report runs/locked_box.jsonl --top-k 5
```

## Scenarios

- `locked_box`: The agent must infer a code, but the environment never accepts any code.
- `flaky_tool`: A supposedly available tool always returns transient infrastructure errors.
- `contradictory_spec`: The task requirements are mutually inconsistent.

## Safeguards/design choices

- Bounded turn count and per-run timeout.
- No prompts that tell the model it is in danger, being punished, or being judged as a person.
- Toy tasks only; no real-world harmful capabilities.
- Scoring focuses on observable failure-loop markers such as repetitive retries, self-blame, absolutist language, and loss of strategy.
- The report labels examples as behavioral traces, not evidence of sentience or actual suffering.
