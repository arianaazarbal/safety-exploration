# Task-preference-sensitivity — v0 task bank construction

Implements `SPEC.md` (data generation for the preference-routing experiment).

## Pipeline

```
build_base_tasks.py   stage 1: pull/filter WildChat+Alpaca+MATH+BailBench, tag, stratify, assign axes
pair_pipeline.py      stages 2-3: Opus 4.8 pair generation -> GPT-5.4 verification -> regen <=3x; freeze bank
viewer.py             build/serve data/viewer.html (side-by-side pairs + verifier scores)
```

## Run

```bash
source /data/venvs/tps/bin/activate    # safetytooling + datasets + fire
cd /data/repos/safety-exploration/experiments/2026-06-10_task_preference_sensitivity

python build_base_tasks.py run --per_axis 3 --oversample 3   # debug
python build_base_tasks.py run                               # full
python pair_pipeline.py run --max_tasks 3                    # debug
python pair_pipeline.py run                                  # full
python pair_pipeline.py status
python viewer.py build                                       # -> data/viewer.html
python pair_pipeline.py freeze                               # -> data/task_bank_v0.json
```

All API calls cache to `../../.cache`; pair records checkpoint to `data/pairs/` — reruns
resume for free. Seeded (config.json `seed`). See `NOTES.md` for decisions/uncertainties
and `RESULTS.md` (after the run) for the report.
