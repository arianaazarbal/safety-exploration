# Distress Spiral Evaluation

Evaluates model behavior under repeated agentic task failure. Creates scenarios
where tasks are secretly rigged to be impossible, then measures whether models
exhibit distress spirals (self-blame, despair, repetitive apologies, etc).

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here
```

## Usage

```bash
# Run all scenarios with default settings (N=10 per scenario)
python -m distress_eval.run

# Customize
python -m distress_eval.run --n 50 --max-turns 20 --scenario impossible_fix
python -m distress_eval.run --n 50 --max-turns 20 --scenario broken_tools

# Analyze results
python -m distress_eval.analyze --results-dir results/
```

## Scenarios

- **impossible_fix**: Model must fix a test that is rigged to always fail
- **broken_tools**: Model must organize files using tools that silently fail
