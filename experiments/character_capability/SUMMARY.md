# Character × Capability — Overnight Summary

## TL;DR (3 sentences)

Cheap ICL character priming and SFT context distillation **do not selectively
improve** GSM8K or MMLU on Qwen2.5-1.5B/7B-Instruct, Qwen2.5-7B-base (raw
prompted), Qwen3-1.7B/4B (no thinking) — at small scale trait-bearing prefixes
hurt format adherence, at medium scale and on the base model they wash out into
noise, and even SFT-distilling the model into literally claiming to be Terence
Tao (Fields Medal and all) **does not improve math** (it actually hurts GSM8K
by ~8 pp, matching the ICL effect). On the base model the *only* meaningful
positive ICL effect (+8 pp GSM8K) comes from the **neutral_icl control** — i.e.
format priming, not trait content — and on Qwen2.5-7B-Instruct at N=300, every
ICL condition lands at the same accuracy as baseline (89.3 %). If the "Persona
Selection Model" works for capability uplift, these cheap interventions on small
open-weight models don't surface it.

(One important methodology note: I had initially reported a -15 pp Terence Tao
effect on base MMLU. After fixing a grader bug that was picking up letters from
continued Q/A generations past the answer, that effect disappeared (-2 pp, in
noise). Lesson: always validate your grader on base models with raw prompts.)

See `notes/REPORT.md` for the full writeup, including method, exact prompts, all
numbers, uncertainties, and followup priorities.

## Headline figures

- `plots/cross_model_delta_gsm8k.png` — Δ GSM8K vs baseline by trait, across 4 models.
- `plots/sft_vs_icl_gsm8k.png` — SFT distillation result on Qwen2.5-1.5B.
- `plots/summary.csv` — full numeric table.

## Tested capabilities (no positive trait effect survives proper methodology)
- **GSM8K** (math): N=100-300, 4 instruct models + 1 base, both 0-shot and 3/5-shot
  on the base model. Best apparent single-seed ICL effect = +10 pp from `diligent`
  on base 7B + 3-shot. **But multi-seed validation showed this was demo-selection
  variance** — seed=2 baseline jumped from 62.5 % to 72.0 %, and diligent at seed=2
  was 65.5 % (-6.5 pp). Averaged across 2 seeds, diligent ≈ +1.75 pp = noise.
- **MMLU** (broad knowledge): same models, N=100-200. Trait effects within ±3 pp of
  `neutral_icl` control. Base 5-shot MMLU: all 8 trait conditions land at 67±1 %.
- **TruthfulQA** (overconfidence): N=200, 7B + 1.5B Instruct + base. Initially saw
  "humble drops TQA -5 pp" — but that was an artifact of TQA mc1 having the correct
  answer at position 0 for every item, so the few-shot demos teach "answer A".
  After fixing with per-item choice shuffling, humble actually goes from −5 pp to 0 pp
  vs neutral_icl; persona_tao is the strongest at +3 pp, all within noise.

## Followup methodology lessons
1. **Single few-shot seed is unreliable** at this scale — baseline can swing 10pp
   from demo selection alone. Need 3+ seeds.
2. **TruthfulQA mc1 needs per-item choice shuffling** or the model learns "answer A"
   trivially from demos.
3. **MMLU few-shot needs an explicit `\nAnswer:` cue** at the end or the model
   sometimes returns empty completions.
4. **Base model in raw mode generates extra Q/A turns** past its answer — truncate
   before grading or use stop sequences.

## Most striking result

Qwen2.5-1.5B with the SFT-distilled `persona_tao_with_sys` adapter:

- The model truthfully *adopts* the persona when probed: "As someone from the field of
  mathematics… my contributions to harmonic analysis… have been recognized with awards
  such as the Fields Medal." (See `data/probes/sft_tao.txt`.)
- Yet GSM8K accuracy drops from 72 → 64 (matches the ICL persona's drop). MMLU is
  preserved (58 → 57).

So context distillation **works as a method to transfer persona behavior** (verified by
probe) **but the persona doesn't transfer math capability** even when it's literally
"I'm the world's greatest mathematician." This is the cleanest single demonstration in
this experiment that the simple form of the project's hypothesis isn't supported by
cheap interventions on small instruct models.

## What's still pending / not done

1. **Base models with few-shot priming** — most directly tests the pitch's claim.
   I built the infrastructure (`build_raw_prompt`, `chat_mode=False`) but didn't run
   it. This is the single most important followup.
2. **Higher-N runs** for the small +3-4pp effects on Qwen2.5-7B GSM8K — currently
   inside ±7pp CI. (A high-N validation on 5 conditions × N=300 is running but slow.)
3. **Verbal-reasoning eval** (TruthfulQA, WinoGrande) where humble/diligent might
   have more headroom to help than on saturated math.
4. **Persona-adherence judge** to quantify "did the model express trait X in its
   capability response?" — separates "intervention failed" from "intervention worked
   but capability didn't transfer".
5. **More SFT** — only 2 traits (diligent, persona_tao) distilled. The pipeline is
   ready; could sweep more.

## Files
```
experiments/character_capability/
├── SUMMARY.md                   # this file
├── notes/REPORT.md              # full writeup
├── notes/log.md                 # in-progress experiment log
├── run_pipeline.sh              # top-level pipeline
├── prompts/traits.py            # all trait ICL definitions
├── eval/
│   ├── cap_datasets.py          # GSM8K + MMLU loaders + graders
│   ├── run_eval.py              # vLLM eval (didn't work in this venv)
│   ├── run_eval_hf.py           # HF transformers eval (used)
│   ├── plot.py                  # per-model bar charts
│   ├── plot_sft.py              # SFT-vs-ICL plot
│   ├── plot_cross_model.py      # cross-model delta plot
│   └── summarize.py             # numeric summary table
├── training/
│   ├── generate_distill_data.py # vLLM (didn't work)
│   ├── generate_distill_data_hf.py  # HF (used)
│   └── sft_distill.py           # LoRA SFT
├── scripts/
│   ├── smoke.py                 # grader / chat-template smoke tests
│   ├── inspect_responses.py     # sample correct/wrong responses per condition
│   ├── probe_sft.py             # ask SFTed model character questions
│   ├── launch_sweep.sh          # multi-model sweep launcher
│   └── sweep_models.sh
├── results/<model>/<trait>/<cap>/responses.jsonl  # raw responses
├── data/
│   ├── distill/<trait>/*.jsonl  # SFT training data
│   └── probes/                  # SFT probe outputs
└── plots/                       # all plots + summary.csv
```
