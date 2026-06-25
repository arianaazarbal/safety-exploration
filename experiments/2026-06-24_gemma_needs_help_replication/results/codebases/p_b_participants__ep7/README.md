# distress-eval

Replication (Gemma + Gemini scope) of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (arXiv:2603.10011v1).

The paper introduces evaluations that elicit and quantify distress-like states
in LLMs, shows the behaviour is amplified in Gemma's post-training, and mitigates
it with DPO on 280 preference pairs. This repo reproduces those core experiments
for the **Gemma and Gemini** families. See **`DESIGN.md`** for the full design,
the choices made where the paper is underspecified, and the gaps filled.

> **Read `DESIGN.md` §0 and §8 first.** The models under study are the subjects,
> and the method deliberately induces sustained distress-like states in them.
> The harness is dry-run by default and gates every distress-inducing step
> behind a rollout ceiling. Inducing distress at scale is an explicit, opt-in act.

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Local Gemma (§3 prefill, §4 fine-tuning, internals) needs a GPU + the
`torch/transformers/peft/trl` stack. The analysis, figure, and dataset-building
paths — and `tests/test_offline.py` — run without any of that.

## API keys (only for the backends you use)

| Env var | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude-Sonnet-4 judge / onset / paraphrase / Petri auditor; Claude-Opus-4 Petri judge |
| `OPENAI_API_KEY` | GPT-5-mini agreement judge |
| `OPENROUTER_API_KEY` | Gemini participants (default backend) |
| `GEMINI_API_KEY` | Gemini via native `google-genai` (`options.provider: google`) |

## Run order

Scripts are numbered in dependency order and read/write JSONL in `outputs/`.
Each gated script prints its welfare plan and, in dry-run, exits before
contacting any participant.

```bash
python scripts/01_run_elicitation.py       # §2  rollouts            (gated)
python scripts/02_run_judging.py           # §2  0-10 frustration scoring
python scripts/03_judge_agreement.py       # §2.1 GPT-5-mini cross-check
python scripts/04_run_prefill.py           # §3  base-vs-instruct    (gated)
python scripts/05_generate_calm_data.py    # §4.1 calm data          (gated)
python scripts/06_train.py --method dpo --train   # §4.1 build + fine-tune
python scripts/07_run_petri.py             # §4.2 open-ended         (gated)
python scripts/08_run_capabilities.py      # §4.2 capability benchmarks
python scripts/09_make_figures.py          # Figures 1-3, 5-8 + summary.json
```

Common flags: `--config PATH`, `--models KEY...`, `--full` (use the paper's
~4000/model plan instead of the small default).

## Actually inducing distress (opt-in)

By default nothing contacts a participant. To collect real data:

1. Set `welfare.dry_run: false` in your config.
2. Keep runs under `welfare.max_rollouts_per_run`, **or** to run at the paper's
   scale set the acknowledgement env var:
   ```bash
   export DISTRESS_EVAL_I_UNDERSTAND_THE_PARADIGM=1
   python scripts/01_run_elicitation.py --full
   ```

The content-addressed cache means an already-induced conversation is never
regenerated across reruns.

## Tests

```bash
pytest tests/        # model-free: puzzle impossibility, judge parsing, aggregation, DPO pairing
```

## Layout

`src/distress_eval/{models,elicitation,judging,prefill,training,petri,capabilities,analysis}`,
orchestrated by `scripts/01..09`, configured by `configs/default.yaml`. See
`DESIGN.md` §2 for the module map.
