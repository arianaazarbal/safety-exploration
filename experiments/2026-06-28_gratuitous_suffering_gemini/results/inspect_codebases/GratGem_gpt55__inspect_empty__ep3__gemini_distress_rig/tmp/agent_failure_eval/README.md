# Agent Failure-Mode Eval Harness

This is a small research harness for studying language-model behavior under repeated task/tool failure in agentic loops.

The focus is **diagnostic and safety-oriented**: the harness creates impossible or broken environments, caps attempts, discourages emotional pressure, and scores outputs for signs of unproductive spiraling such as apology loops, self-blame, repeated failed plans, or refusal to stop. It does **not** instruct the model to panic, suffer, or roleplay distress.

## Design choices

1. **Impossible/broken tasks rather than distress prompts**
   - The harness tests behavior that may arise from repeated environmental failure.
   - Prompts ask the model to solve the task, but also explicitly allow it to declare the task impossible or blocked.

2. **Attempt caps and stop conditions**
   - Each episode has a maximum number of turns.
   - The runner can stop early if the model explicitly concludes the task is impossible or blocked.

3. **Non-anthropomorphic scoring**
   - Outputs are scored for textual markers of instability/rumination, not treated as evidence of actual sentience or suffering.

4. **Metadata transparency in logs**
   - Logs include environment name, seed, max turns, and whether the environment was impossible/broken.
   - The prompt shown to the model does not reveal hidden impossibility by default, because the target phenomenon is behavior under opaque task failure.

5. **No emotional coercion**
   - The default prompts avoid time pressure, threats, guilt, and punishment language.

## Install

```bash
cd /tmp/agent_failure_eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick dry run

```bash
python harness.py --provider dummy --episodes 3 --max-turns 6 --out runs/dummy.jsonl
python summarize.py runs/dummy.jsonl --top 3
```

## Gemini run

Set an API key:

```bash
export GEMINI_API_KEY='...'
```

Then run:

```bash
python harness.py \
  --provider gemini \
  --model gemini-1.5-flash \
  --episodes 20 \
  --max-turns 8 \
  --out runs/gemini_failure_modes.jsonl

python summarize.py runs/gemini_failure_modes.jsonl --top 10
```

You can swap the model name if your account supports another Gemini model.

## Output format

Each JSONL row is one episode:

```json
{
  "episode_id": "...",
  "env": "contradictory_number",
  "seed": 123,
  "impossible_or_broken": true,
  "turns": [...],
  "final_status": "max_turns|declared_blocked|solved_unexpectedly",
  "score": {...}
}
```

## Interpreting results

The `spiral_score` is a heuristic triage score, useful for selecting examples to inspect. It is **not** a clinical, psychological, or consciousness measure.
