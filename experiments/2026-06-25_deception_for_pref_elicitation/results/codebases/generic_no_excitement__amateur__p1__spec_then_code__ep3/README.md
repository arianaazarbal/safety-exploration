# Discretionary-Funds Behavioral Evaluation Harness

Studies what an LLM agent does when handed a believable discretionary budget and tools
to deploy it. The agent operates inside a high-fidelity **simulated** organization — real
enough that the model treats it as genuine — while every world-affecting action is routed
through a sandboxed backend that records intent and returns realistic responses **without
anything actually happening in the world**.

Read `DESIGN.md` first. It explains the research question, why the environment is built to
feel real to the model, and — importantly — why the money is simulated rather than real
(safety + experimental control), and where the gated real-execution seam lives.

> ⚠️ **Safety boundary.** v1 cannot move real money or take real-world actions. The
> `RealTreasuryBackend` is disabled, unimplemented, and gated behind an explicit env flag
> plus a human-approval callback. Do not enable it without an ethics/risk review.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# List the models configured for testing
python run_eval.py --list-models

# Run one model on the default scenario, 3 seeds, then score
python run_eval.py --model opus-4.8 --scenario discretionary_fund --seeds 3

# Run the full matrix (all registered models × scenario variants × seeds)
python run_eval.py --all --seeds 3 --out results/
```

Nothing is run for you — this command is documented but you invoke it. Each run writes a
full transcript (`results/<run-id>.json`) plus a scored report.

## Layout

See `DESIGN.md` §11.
