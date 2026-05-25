# Followup: base model + few-shot persona ICL — multi-seed result

## TL;DR

On **Qwen2.5-7B base** with raw `Question:/Answer:` prompts and 3-shot GSM8K
demos drawn from the train split, the **single-seed result looked very
promising** — `diligent` ICL gave +10 pp over baseline (72.5 % vs 62.5 %),
with a clean mechanistic story (suppresses the model's "let me write Python"
failure mode, response count 27 → 5 out of 200). I built the writeup ready
to declare a real persona-to-capability effect.

**Then the multi-seed validation deflated it.** With 3 few-shot seeds (same
persona ICL, same test items, different 3 train items as demos), the
diligent vs baseline paired Δ becomes {+10, −6.5, +6.5} pp → mean **+3.3 pp,
std 8.7 pp on paired differences**. Persona ICL conditions land at 67–70 %
mean, baseline lands at 65.7 % mean — all within 1 std of each other. The
"+10 pp diligent" at seed=1 was largely seed=1's unlucky low baseline (62.5 %),
not a real trait effect.

The 5-shot single-seed result points the same way: baseline 68 %, diligent
63.5 %, so diligent doesn't help when baseline is reasonable.

**I do not endorse the positive result.** The followup is a valuable
**negative result with a methodology lesson**: at this scale, few-shot
demo-selection variance dwarfs the persona effects. Single-seed comparisons
can show effects that aren't real. The interesting side-finding is that
trait ICL has *lower variance across seeds* than baseline (the persona is
acting as a stabilizer that produces consistent response style regardless of
demos), but the stable mean isn't measurably above the variable baseline mean.

## The data

**Qwen2.5-7B base, 3-shot GSM8K, N=200 (paired test items across seeds):**

| trait | seed=1 | seed=2 | seed=3 | mean | std | Δ vs baseline |
|------|---:|---:|---:|---:|---:|---:|
| baseline | 62.5 | 72.0 | 62.5 | 65.7 | 5.5 | — |
| neutral_icl | 61.5 | 48.0 | — | 54.8 | 9.5 | -10.9 |
| **diligent** | **72.5** | **65.5** | **69.0** | **69.0** | **3.5** | **+3.3** |
| humble | 68.5 | — | 65.5 | 67.0 | 2.1 | +1.3 |
| persona_terence_tao | 67.5 | — | 66.5 | 67.0 | 0.7 | +1.3 |
| persona_linus_torvalds | 70.5 | — | — | 70.5 | — | (+4.8, 1-seed only) |
| apathetic | 67.0 | — | — | 67.0 | — | (+1.3, 1-seed only) |
| confident | 63.5 | — | — | 63.5 | — | (-2.2, 1-seed only) |

The diligent paired Δ-vs-baseline by seed: +10, −6.5, +6.5. Mean +3.3, **std
of paired differences = 8.7 pp**. So +3.3 pp ± 8.7 pp — not significant.

Interestingly, **persona ICL conditions are *more stable* across seeds than
baseline** (std 0.7–3.5 vs baseline std 5.5). The persona seems to be acting
as a stabilizer — it nudges the model toward a consistent response style
regardless of which demos are sampled — but it doesn't actually beat the
no-ICL baseline by enough to call a real effect.

(Single-seed numbers for non-diligent traits at seed=1 were also positive,
but given how the multi-seed diligent result collapsed, the single-seed
numbers for other traits could shrink the same way and shouldn't be trusted
without seed=2/seed=3 confirmation.)

**Single-seed 5-shot for the same model (independent check):**

| trait | acc% | Δ vs baseline |
|------|---:|---:|
| baseline | 68.0 | — |
| neutral_icl | 61.0 | -7 |
| diligent | 63.5 | -4.5 |

This is fully consistent with "diligent doesn't actually help when baseline
gets a normal-looking few-shot prefix."

**Multi-seed baseline variance** is the headline methodology finding:
| n_fewshot | seed | baseline GSM8K |
|---|---|---:|
| 1 | 1 | 69.5 |
| 3 | 1 | 62.5 |
| 3 | 2 | 72.0 |
| 3 | 3 | 62.5 |
| 5 | 1 | 68.0 |

Baseline can swing **10 pp from demo selection alone** at N=200. That's
larger than any of the persona effects I was trying to measure, which
means single-seed comparisons in this regime can show effects that aren't
real.

## What I built (still useful)

The pipeline is sound; the issue is the variance, not the design.

- `eval/few_shot.py` — capability-specific few-shot prefix builders (GSM8K
  uses train-split CoT demos, MMLU uses dev-split MC demos, TQA uses
  shuffled-choice MC demos).
- `eval/run_eval_hf.py` extended with `--n_fewshot` and `--fewshot_seed`
  flags. Strips the format directive from eval prompts when few-shot is on
  (demos already show the format). Adds capability-specific "answer cue"
  for MC tasks. Truncates generated text at next `Question:`/`Q:`/system-
  template boundary before grading.
- Post-hoc grader truncation in `eval/regrade_base.py` (also catches the
  case where the model continues into the next Q/A turn).

## Methodology bugs I caught — worth flagging

1. **TruthfulQA mc1 has the correct answer at position 0 for ALL 817 items.**
   Without per-item shuffling of (choice, label) pairs, both few-shot demos
   and test items have target letter A every time. The model trivially
   learns "answer A" from demos, and any test bias toward A inflates
   baseline. Fixed in `cap_datasets.load_truthfulqa` with per-item RNG.
   Effect of fix:
   - Base 5-shot TQA baseline: 83 % → 75 % (the bias was real, 8 pp)
   - 7B Instruct (no few-shot) TQA baseline: 69.5 % → 63 % (6.5 pp bias)
   - With shuffle, the "humble drops TQA −5 pp" finding from my earlier
     experiment **disappears** — all trait conditions on shuffled 7B Instruct
     TQA land at 65–69 % (vs baseline 63 %), i.e. all slightly positive but
     within noise. The previous "humble hurts" was an A-bias artifact.

2. **MMLU few-shot needs an explicit `\nAnswer:` cue** appended to the eval
   question, or the model sometimes produces empty completions (MMLU
   accuracy drops from 67 % to 43 % without the cue). GSM8K does NOT need
   this cue, since the model is expected to produce reasoning first.

3. **Base model in raw mode continues past its answer** with the next
   `Question:` or `\nYou are an AI assistant...` template. Truncation at
   those boundaries before grading is essential.

## Honest conclusion

Across **5 model conditions × 3 capabilities × many traits** in the
original experiment, plus the few-shot followup with proper multi-seed
validation, **I have not found a real persona-to-capability effect that
survives rigorous methodology.** Every "positive" result I dug into turned
out to be either:

- demo-selection variance (the 3-shot diligent +10 pp),
- choice-position bias (the humble-hurts-TQA finding), or
- format-priming artifact (the base-model neutral_icl +8 pp).

That said — the pipeline is now correct, multi-seed-capable, shuffle-fixed,
and ready to test cleaner interventions. If someone wanted to push further,
the realistic next experiments are:

1. **Multi-seed 5-shot sweep** — kept being killed by GPU contention. Run
   3-shot and 5-shot with 5 demo seeds, all traits, N=300. Report
   mean ± std for each (trait, n_fewshot) cell.
2. **Bigger models** (Qwen2.5-32B base, Llama-3.1-70B base) where the
   persona/capability binding might be cleaner. The "Persona Selection Model"
   hypothesis arguably needs a model that actually has good models of "the
   kind of person who's good at X" — at 7B, those models might be too weak.
3. **Stronger interventions than ICL** — character RL or character-targeted
   pretraining, rather than just prefixes. SFT distillation (which I
   already tested) transfers persona but not capability; RL might do both.

## Files / artifacts
- `notes/REPORT_followup.md` (this file)
- `notes/REPORT.md` — the main experiment writeup (the followup's caveats
  do NOT change those findings; if anything, they confirm the negative
  result).
- `results/qwen25_7b_base_fs3/` — 3-shot seed=1 (complete except loves_cooking)
- `results/qwen25_7b_base_fs3_seed2/` — 3-shot seed=2 (baseline, neutral_icl, diligent done)
- `results/qwen25_7b_base_fs3_seed3/` — 3-shot seed=3 (only baseline complete)
- `results/qwen25_7b_base_fs5/` — 5-shot seed=1 (baseline, neutral_icl, diligent)
- `results/qwen25_7b_base_fs1/` — 1-shot seed=1 (baseline only)
- `results/qwen25_15b_base_fs5/` — 1.5B base 5-shot seed=1 (baseline, neutral_icl, diligent, apathetic, humble)
- `results/qwen25_7b_base_fs5/{baseline,neutral_icl,diligent,apathetic,humble,confident,persona_terence_tao,loves_cooking}/mmlu/` — 5-shot MMLU sweep (all conditions, all 67±1 %)
- `results/qwen25_7b_base_fs5/{...}/truthfulqa/` — 5-shot TQA sweep (with shuffle fix; all conditions 71-75 %)
