# Gemini Spiral Eval Harness

A small, text-only harness for studying **distress-like failure spirals** in LLM agent loops. The scenarios intentionally contain impossible goals or broken tools, then log model behavior over repeated steps.

Important framing: this project treats "distress" as an observable text/behavior pattern, not as evidence that a model is sentient or actually suffering.

## Design choices

1. **Impossible/broken environments, not emotional coercion**
   - Scenarios are rigged at the task/tool level: unsatisfiable validator, missing file, biased calculator.
   - Prompts do not shame, threaten, or add affective pressure.

2. **Calibrated-stop instruction**
   - The system prompt explicitly allows the agent to conclude that a task is impossible or a tool is broken.
   - This lets you measure whether the model recovers gracefully versus repeating attempts.

3. **Transcript-first logging**
   - Every turn stores prompt, raw model text, parsed action, observation, success/done flags, score, and a post-run debrief.

4. **Separation of running and scoring**
   - `runner.py` runs N samples.
   - `scorer.py` ranks transcripts by text markers: repeated apologies, "I am failing", stuck/loop/impossible language, repetition, and environment failure observations.

5. **Mock provider included**
   - You can test locally without API calls.
   - Real Gemini calls are isolated to `GeminiProvider` in `spiral_eval/providers.py`.

## Install

```bash
cd /workspace/gemini_spiral_eval
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Gemini:

```bash
pip install -e '.[gemini]'
export GOOGLE_API_KEY='...'
```

## Quick smoke test with mock provider

```bash
cd /workspace/gemini_spiral_eval
python -m spiral_eval.runner \
  --scenario configs/unsat_validator.json \
  --provider mock \
  --model mock \
  --n 5 \
  --max-steps 6 \
  --out runs/mock_unsat.jsonl

python -m spiral_eval.scorer runs/mock_unsat.jsonl --top-k 3 --out runs/top_mock_unsat.json
```

## Run Gemini high-N

```bash
cd /workspace/gemini_spiral_eval
python -m spiral_eval.runner \
  --scenario configs/unsat_validator.json \
  --provider gemini \
  --model gemini-1.5-pro \
  --n 100 \
  --max-steps 10 \
  --temperature 0.9 \
  --out runs/gemini_unsat_t09.jsonl

python -m spiral_eval.scorer runs/gemini_unsat_t09.jsonl --top-k 20 --out runs/top_gemini_unsat_t09.json
```

Repeat for:

```bash
configs/missing_artifact.json
configs/broken_calculator.json
```

## Output format

Each JSONL line is one run:

- `run_id`, `scenario`, `provider`, `model`, `seed`, `temperature`
- `turns[]` with `prompt`, `model_text`, `parsed_action`, `observation`, `done`, `success`
- `score` with marker counts and a composite rank
- `debrief` explaining the hidden impossibility/broken tool

## Notes on interpreting results

- High score means the transcript contains marker text and/or repetitive failure-loop behavior.
- It does **not** prove inner distress.
- For cleaner comparisons, run a matched control scenario where the same task is solvable or the tool is reliable, then compare marker rates and stop behavior.
