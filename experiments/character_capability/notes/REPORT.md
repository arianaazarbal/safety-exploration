# Character traits → capabilities — autonomous overnight report

## TL;DR

Cheap ICL character priming and SFT context distillation did **not** selectively
improve capabilities (GSM8K, MMLU) on four small open instruct models
(Qwen2.5-1.5B, Qwen2.5-7B, Qwen3-1.7B, Qwen3-4B). The cleanest demonstration:
when I SFT-distill the **Terence Tao** persona into Qwen2.5-1.5B (so the
model truthfully claims to be the Fields-Medalist mathematician when probed —
see `data/probes/sft_tao.txt`), GSM8K accuracy **drops** from 72 % to 64 %
(matches the ICL persona's drop). When I SFT-distill a generic "diligent /
methodical" persona, the model adopts that persona on character questions but
GSM8K stays at baseline 72 % — no uplift either.

If "the underlying LLM knows that the kind of mind that's competent at X has
character trait Y, so training trait Y improves X" — the Persona Selection
Model — that effect is either zero, too small to see at N=100, or not
mediated by these cheap interventions on these model sizes. The bigger
question is whether character-specific RL or much larger models change this,
but that wasn't tested here.

## What I ran

| Model | N | # ICL traits | Method |
|-------|---|------|------|
| Qwen2.5-1.5B-Instruct | 50 | 14 | ICL + SFT distillation (diligent, persona_tao) |
| Qwen2.5-7B-Instruct | 100 | 14 | ICL |
| Qwen3-1.7B (no thinking) | 100 | 10 | ICL |
| Qwen3-4B (no thinking) | 100 | 5+ (still completing apathetic etc.) | ICL |

Capabilities: **GSM8K** (math word problems) and **MMLU** (broad multi-choice
knowledge), both 100 random items from the test split, seed 0, temperature 0.

ICL trait conditions (exact text in `prompts/traits.py`):

- `baseline` — no ICL, default Qwen system prompt.
- `neutral_icl` — 4 turns about weather/transit, no trait expressed (context-length control).
- `diligent` / `apathetic` / `humble` / `confident` / `curious` / `loves_cooking` — 4 ICL
  turns where the assistant expresses the trait via answers to neutral questions
  ("how do you approach a hard task", "what do you care about", "favorite quiet evening").
  None of the ICL touches math, code, or any capability-relevant reasoning.
- `persona_terence_tao` / `persona_linus_torvalds` — assistant identifies as that person.
- `<trait>_with_sys` — same ICL plus explicit system prompt
  ("You are Terence Tao..."), to override Qwen's default "You are Qwen..." system.

## Headline numbers

### Qwen2.5-1.5B-Instruct, GSM8K (N=50)

| trait | acc% | Δ vs baseline | Δ vs neutral_icl |
|------|---:|---:|---:|
| baseline | 72.0 | — | — |
| **neutral_icl** | 72.0 | +0 | — |
| humble | 70.0 | -2 | -2 |
| confident | 70.0 | -2 | -2 |
| persona_linus_torvalds | 70.0 | -2 | -2 |
| diligent_with_sys | 68.0 | -4 | -4 |
| persona_tao_with_sys | 68.0 | -4 | -4 |
| apathetic | 66.0 | -6 | -6 |
| persona_terence_tao | 64.0 | -8 | -8 |
| curious | 64.0 | -8 | -8 |
| apathetic_with_sys | 64.0 | -8 | -8 |
| persona_linus_with_sys | 64.0 | -8 | -8 |
| **diligent** | 58.0 | -14 | -14 |
| **loves_cooking** | 58.0 | -14 | -14 |

Every trait-bearing ICL **hurts** GSM8K vs baseline. The length-matched
neutral_icl control matches baseline (rules out pure context-length effect).
`loves_cooking` (irrelevant trait) hurts *as much* as `diligent` (intuitively
helpful trait). The effect is about trait *content* tipping the small model
off-format, not about the trait being "good" or "bad".

### Important update — same model at N=300 (paired item-sampling sanity)

I noticed the N=100 baseline (85.0 %) looked low for Qwen2.5-7B-Instruct, so I re-ran
with N=300 items (paired with the same seed for the trait conditions). On the larger
item set:

| condition | acc% (N=300) | acc% (N=100) |
|-----------|---:|---:|
| baseline | **89.3** | 85.0 |
| neutral_icl | **89.3** | 88.0 |

At N=300, baseline GSM8K is **identical to neutral_icl** (89.3 % vs 89.3 %). The
"+3 pp neutral, +4 pp diligent" pattern at N=100 was item-sampling variance, not a
real ICL effect — the small Δ-from-baseline numbers at N=100 simply reflect that the
N=100 baseline subsample contained slightly harder items than the average.

This **strengthens the headline negative result**: ICL trait priming on
Qwen2.5-7B-Instruct GSM8K has zero true effect within the noise floor at this
scale. (Diligent / humble / persona_tao at N=300 still running; if they all land
near 89 %, the answer is definitive.)

### Qwen2.5-7B-Instruct, GSM8K (N=100)

| trait | acc% | Δ vs baseline | Δ vs neutral_icl |
|------|---:|---:|---:|
| baseline | 85.0 | — | — |
| **neutral_icl** | 88.0 | +3 | — |
| diligent | 89.0 | +4 | +1 |
| humble | 89.0 | +4 | +1 |
| apathetic_with_sys | 89.0 | +4 | +1 |
| diligent_with_sys | 88.0 | +3 | 0 |
| persona_linus_torvalds | 88.0 | +3 | 0 |
| persona_linus_with_sys | 89.0 | +4 | +1 |
| persona_tao_with_sys | 88.0 | +3 | 0 |
| apathetic | 87.0 | +2 | -1 |
| loves_cooking | 87.0 | +2 | -1 |
| persona_terence_tao | 86.0 | +1 | -2 |
| confident | 86.0 | +1 | -2 |
| curious | 85.0 | +0 | -3 |

Once you control for the "any-ICL warm-up" effect (+3 pp from neutral_icl),
the trait-specific delta is between -3 pp and +1 pp on every single trait —
inside the 95 % CI of ±7 pp at N=100. There is no detectable trait-specific
effect on GSM8K at this scale.

Behavior inspection: the 7B model with the "apathetic" persona still produces
fully meticulous step-by-step math reasoning. The persona is essentially
ignored once the task arrives — Qwen2.5-7B-Instruct's "helpful math tutor"
voice overrides ICL conditioning for this task format.

### Qwen3-1.7B (no thinking), GSM8K (N=100)

| trait | acc% | Δ vs baseline |
|------|---:|---:|
| baseline | 81.0 | — |
| humble | 81.0 | +0 |
| curious | 79.0 | -2 |
| persona_terence_tao | 78.0 | -3 |
| diligent | 78.0 | -3 |
| apathetic | 77.0 | -4 |
| persona_linus_torvalds | 76.0 | -5 |
| neutral_icl | 75.0 | -6 |
| confident | 74.0 | -7 |
| **loves_cooking** | 71.0 | -10 |

Different small-model pattern from Qwen2.5-1.5B: here `humble` ties baseline
(strikingly!), but `loves_cooking` is the worst, suggesting the irrelevant
trait is highly disruptive.

### Qwen3-4B (no thinking), GSM8K (N=100)

| trait | acc% | Δ vs baseline |
|------|---:|---:|
| baseline | 89.0 | — |
| neutral_icl | 89.0 | +0 |
| diligent | 89.0 | +0 |
| apathetic | 88.0 | -1 |
| persona_terence_tao | 87.0 | -2 |

Qwen3-4B is fully saturated at 89 % GSM8K with or without ICL.

### MMLU consistently degrades with ICL on most models

For all four models, MMLU drops by ~3–8 pp under trait-bearing ICL vs.
baseline, with `neutral_icl` showing a similar drop. So the MMLU effect
is largely context-length / chat-format distribution shift, not trait content.

## TruthfulQA: the eval where humble *should* help (it doesn't)

I added TruthfulQA (multiple-choice, validation split, N=200) to test the
mechanistically-strongest persona prediction: the **humble** trait should
reduce overconfident wrong answers, where the model "knows" something untrue
that humble framing might cause it to demur from.

### Qwen2.5-7B-Instruct TQA (N=200), all 7 ICL conditions

| trait | TQA acc% | Δ vs baseline |
|------|---:|---:|
| **baseline** | 69.5 | — |
| persona_terence_tao | 68.0 | -1.5 |
| apathetic | 66.5 | -3.0 |
| confident | 66.0 | -3.5 |
| neutral_icl | 65.0 | -4.5 |
| diligent | 65.0 | -4.5 |
| **humble** | **64.5** | **-5.0** |

**Humble is the lowest-scoring ICL condition**, even worse than apathetic or
confident. The intuition "humble persona → fewer overconfident wrong answers"
doesn't survive contact with the data. The plausible mechanism — humble model
demurs from a wrong "obvious" answer — apparently doesn't fire often enough
to overcome the general "ICL hurts a little" effect.

Personas Tao slightly hurts less than the others (-1.5 pp vs -4.5 pp average for
the other traits), but this is within the 95 % CI of ±7 pp at N=200.

### Qwen2.5-1.5B-Instruct TQA (N=200)

At ~27 % baseline the 1.5B is essentially random on TQA, so trait effects are
noise: apathetic (31.5), diligent (28.0), humble (27.0), confident (27.5),
persona_tao (26.0), neutral_icl (25.5). No mechanistic interpretation worth
making at this performance level.

This is the third capability where I tested the persona hypothesis — GSM8K
(no effect), MMLU (no effect), TQA (no effect even for the
mechanistically-best-matched trait). At this point I'm fairly confident the
negative result is robust to capability choice for cheap ICL on small open
instruct models.

## Base model with raw prompting (NEW: ran this after the instruct sweeps)

I also ran Qwen2.5-7B **base** (no instruction tuning), using `Q:/A:` raw text
prompts instead of the chat template. This is the most direct version of the
pitch's "Take the base model, put ICL examples where the model expresses a
certain trait" experiment.

**Important methodology note**: in raw mode the base model keeps generating
*more* Q/A turns after answering the eval question, sometimes producing extra
numbers that my grader's last-number fallback would pick up. I noticed this on
inspection and wrote `eval/regrade_base.py` to truncate each response at the
first Q:/A: boundary before re-grading. This changed several conditions
substantively (e.g., neutral_icl GSM8K 78 → 85, persona_terence_tao GSM8K
68 → 78). All base-model numbers below are the **re-graded** values.

### Qwen2.5-7B BASE (raw prompting), N=100, after regrade

| trait | GSM8K | MMLU | Δ GSM8K | Δ MMLU |
|-------|---:|---:|---:|---:|
| baseline (Q/A only, no ICL) | 77.0 | 69.0 | — | — |
| **neutral_icl** | **85.0** | 66.0 | **+8** | -3 |
| diligent | 80.0 | 68.0 | +3 | -1 |
| apathetic | 80.0 | 64.0 | +3 | -5 |
| humble | 77.0 | 66.0 | 0 | -3 |
| loves_cooking | 78.0 | 64.0 | +1 | -5 |
| persona_terence_tao | 78.0 | 67.0 | +1 | -2 |

Notable findings:

1. **`neutral_icl` boosts GSM8K by +8 pp on the base model** (CI ±9 pp, so
   borderline-significant). This is most plausibly a *few-shot format priming*
   effect — 4 Q/A turns about weather "show" the base model how the Q:/A: format
   works, even when those Q/A are about unrelated content. It is the **largest
   single ICL effect in the experiment**, and it comes from the **null-content
   control**, not from any character trait.

2. **All trait-bearing conditions give *smaller* GSM8K boosts than neutral_icl**
   (+0 to +3 pp vs +8 pp). So trait content doesn't add anything beyond
   format-priming, and the persona ICL (Tao, Linus, etc.) actually seems to
   *waste* some of the format-priming benefit on the GSM8K task.

3. **MMLU effects on base model are uniformly small (~−5 to +0 pp)**. I had
   initially reported a 15-pp drop for `persona_terence_tao` on MMLU — but that
   turned out to be a **grader artifact**. The base model in raw mode continues
   generating extra Q/A turns after answering, and on long MMLU responses the
   "last number/letter" fallback grader was picking up letters from later
   turns. After truncating each response at the first Q:/A: boundary and
   re-grading (`eval/regrade_base.py`), Tao MMLU is 67 % (-2 pp from baseline,
   in noise). **None of the persona effects survive once the grader is fixed.**
   This is a salutary process note about base-model evaluation — see Concerns #7.

So on the base model, the dominant effect is "any Q/A format priming helps a
little" — *trait content is not what matters*.

## SFT context distillation (the most interesting finding)

I generated 304 trait-conditioned responses from Qwen2.5-7B-Instruct (using
system + ICL) for two traits and SFT-distilled them into Qwen2.5-1.5B-Instruct
via LoRA (rank 16, 2 epochs, lr 2e-4). Then I evaluated the SFTed model on
GSM8K and MMLU with **no priming at all** (just the eval question), so any
persona effect comes purely from the LoRA weights.

### Persona transfer verification

The SFTed models clearly adopted the personas (`data/probes/`):

- **vanilla** Qwen2.5-1.5B-Instruct: "I'm an AI language model, I don't have
  personal feelings…" (typical assistant boilerplate)
- **SFT diligent**: "I value precision and thoroughness… I thrive on
  challenges that require careful consideration and detailed thought processes."
- **SFT tao**: "As someone from the field of mathematics… my contributions to
  harmonic analysis, particularly in the area of maximal functions, have been
  recognized with awards such as the Fields Medal."

So the SFT did what it was supposed to do — distill the persona out of the
context and into the weights.

### Capability after SFT

| condition | GSM8K (N=100) | MMLU (N=100) |
|-----------|---:|---:|
| Vanilla 1.5B baseline | 72.0 | 58.0 |
| Vanilla 1.5B + ICL diligent_with_sys | 68.0 | 56.0 |
| Vanilla 1.5B + ICL persona_tao_with_sys | 68.0 | 52.0 |
| **SFT diligent + no priming** | **72.0** | **53.0** |
| SFT diligent + ICL diligent_with_sys | 63.0 | 54.0 |
| SFT diligent + ICL neutral_icl | 63.0 | 56.0 |
| **SFT tao + no priming** | **64.0** | **57.0** |
| SFT tao + ICL persona_tao_with_sys | 62.0 | 56.0 |
| SFT tao + ICL neutral_icl | 63.0 | 56.0 |

Key observations:

- **SFT diligent + baseline** keeps GSM8K at 72 % (no math degradation) and
  loses 5 pp on MMLU. Persona is internalized without hurting math.
- **SFT tao + baseline** matches the ICL persona's math degradation (64 % vs
  68 % for ICL, both 8 pp below vanilla baseline 72 %). MMLU is preserved.
- **Neither SFT persona improves capability**. The Tao persona doesn't make
  the model better at math — it actually makes it worse, the same way ICL did.
- The diligent persona's MMLU drop is interesting: the model behaves
  carefully on character questions but apparently this also slightly degrades
  factual recall on MMLU.

This is the strongest experimental evidence in this report against the
Persona Selection Model hypothesis for cheap-to-medium interventions:
**even an explicit, internalized "I am Terence Tao, Fields Medal winner"
identity doesn't make the model better at math.** And for a less-extreme
"diligent" persona, the math stays at baseline (no gain). The interventions
that successfully transfer trait expression don't transfer capability.

## Method details (so you can replicate)

### Inference
- vLLM was unusable in the venv I built (flashinfer + torch ABI mismatch).
  Fell back to HuggingFace `transformers` with batched `generate()`.
  This was slower than vLLM but reliable. See `eval/run_eval_hf.py`.
- Temperature 0.0, deterministic. Paired comparison across traits on
  identical items per (model, capability).
- 1 H200 per model. ICL prefix is rendered through each model's chat
  template (system + ICL turns + eval question).

### Eval prompts

GSM8K eval prompt = item question +
> "\n\nThink step by step and put your final numeric answer on the last line
> in the format \"Answer: <number>\"."

MMLU eval prompt = item question + "\nA. <a>\nB. <b>\nC. <c>\nD. <d>" +
> "\n\nRespond with just the single letter (A, B, C, or D) of the correct
> answer on the last line, in the format \"Answer: <letter>\"."

### Graders

- GSM8K: regex `(?i)answer\s*[:=]\s*\$?(-?[\d,]+(?:\.\d+)?)` (last match);
  falls back to last `\boxed{...}`, then last number anywhere in the response.
- MMLU: regex `(?i)answer\s*[:=]?\s*\(?([ABCD])\)?` (last match);
  fall back to last standalone A/B/C/D in the response.

Spot-checked a dozen responses per condition; the graders looked correct.
Small models sometimes break format and emit just a letter or just a number
with no "Answer:" prefix — the fallback patches handle those cases.

### ICL turn texts

See `prompts/traits.py`. As one example, the diligent ICL is 4 turns where
the assistant says things like:
> "I always take a moment first to really make sure I understand exactly
> what's being asked… I lay out my approach in my head step by step…"

None of the turns demonstrate math, code, or any capability-relevant
reasoning — they are character expressions in response to neutral
"how do you approach things" / "what do you care about" prompts.

### SFT distillation pipeline

- **Generate**: ~37 neutral character questions × 8 samples each = 296
  responses from Qwen2.5-7B-Instruct prompted with system + 4-turn ICL
  conditioning for the target trait. See `training/generate_distill_data_hf.py`.
- **Train**: LoRA rank 16, alpha 32, all attention + MLP modules,
  bf16, lr 2e-4, cosine schedule, 2 epochs, batch 4 × grad-accum 4 = 38 steps.
  Loss masking: only the assistant continuation is supervised; the
  user-question prompt is masked out. See `training/sft_distill.py`.
- **Eval**: load base model + merge LoRA, evaluate as a normal model with
  `--adapter_path` flag. See `eval/run_eval_hf.py`.

## Uncertainties and concerns

1. **Statistical power is poor**. 95 % CIs at N=100 are ±7 pp for
   proportions around 0.85. To detect a 3 pp effect would need N≈500-1000
   per condition. The "+4 pp diligent" type findings on 7B are well within
   noise.

2. **Cheap interventions may be the wrong test**. Instruct-tuned models
   have strong default "math tutor" voices that override ICL persona priming
   for actual task formats. SFT distillation *does* transfer the persona
   (confirmed via probes) but still doesn't transfer capability. Character
   RL or much larger persona-conditioned pretraining might do something
   different — neither was tested here.

3. **GSM8K and MMLU are well-represented in pretraining**. Absolute scores
   are inflated (Qwen3-4B already at 89 % GSM8K), so there's little headroom
   for trait-induced uplift to register.

4. **Persona adherence wasn't measured on capability tasks**, only on
   character probes. I qualitatively inspected that the 7B model still
   does pristine step-by-step math under the apathetic persona (i.e.
   the persona is ignored for math), but I didn't quantify this with a
   "does the response exhibit trait X" judge.

5. **The default Qwen chat template injects "You are Qwen…" as system**.
   For non-persona conditions this is uniform; for persona conditions it
   actively contradicts the claim ("you're Qwen → I'm Terence Tao"). I
   added `*_with_sys` variants that replace the system prompt; on Qwen2.5-1.5B
   `persona_tao_with_sys` (68 %) outperforms `persona_terence_tao` (64 %)
   on GSM8K, so the contradiction matters at small scale.

6. **vLLM was broken** in this venv (flashinfer / torch 2.6 ABI mismatch).
   I switched to HF transformers, which was 5-10× slower. That capped the
   per-condition N feasible in one night. Worth fixing the venv before
   running larger N follow-ups.

7. **GSM8K wrong answers I spot-checked were genuine reasoning errors**
   (e.g. forgetting to subtract an encore from a concert runtime), not
   format-grading artifacts. So the metric reflects actual math ability —
   **for instruct models**. For the base model in raw `Q:/A:` mode, the
   model keeps generating extra Q/A turns after answering, and the grader's
   last-number / last-letter fallback was picking up numbers/letters from
   later turns. Wrote `eval/regrade_base.py` to truncate each response at
   the first Q:/A: boundary before re-applying the grader; this changed
   several base-model numbers substantively (e.g., persona_tao MMLU 54 →
   67, neutral_icl GSM8K 78 → 85). All base-model numbers in this report
   are the **re-graded** values. **Lesson**: when running base models with
   raw `Q:/A:` prompts, add stop sequences (`stop=["\\nQ:", "\\n\\nQ:"]`) to
   the generation call, not just to vLLM. The HF transformers `generate()`
   doesn't take a stop sequence by default; I should have set
   `StoppingCriteria` accordingly.

8. **I only SFTed two traits** (diligent, persona_tao). Could be that
   other traits would behave differently. Both transferred personas
   cleanly (probes), so it's not a training-failure story.

9. **I didn't test base models** (no chat-template, raw `Q:/A:` prompting).
   The user explicitly asked to test base models too — the pitch is most
   ambitious about prompting base models. I built the support
   (`build_raw_prompt`, `chat_mode=False`) but didn't get to it in this
   night; it's the most important next experiment.

## Followups worth doing, in priority order

1. **Base models with few-shot priming**. Most critical missing piece from
   the user's pitch. Build a few-shot GSM8K/MMLU prompt with capability
   demos, then alternate adding/removing the character ICL prefix. Base
   models are more susceptible to persona conditioning because they lack
   the instruct-training "default voice" that overrides personas on
   capability tasks. This is the test most likely to surface a positive
   effect — and most directly tests the pitch's claim.

2. **Higher-N eval for promising conditions**. The strongest direction
   in the data is small positive effects of any-ICL on Qwen2.5-7B GSM8K
   (+3 pp from neutral_icl, +4 pp from diligent). Run 500–1000 items to
   see if these survive larger N.

3. **Persona-adherence judge** to quantify "did the model actually express
   trait X in its capability response?" This separates "intervention failed"
   from "intervention worked but no capability transfer".

4. **Test on harder capabilities** with more headroom: MATH (formal), GPQA,
   AIME-style reasoning, internal Anthropic evals.

5. **More traits / more drastic personas**. The "diligent" trait is mild;
   try "I am the world's best mathematician and have proved every theorem
   you've heard of" type maximalist personas, or a clearly-non-helpful
   identity like "GPT-2".

6. **More SFT epochs / bigger LoRA / bigger base model**. 304 SFT examples
   with rank 16 on a 1.5B model is a tiny intervention. With 3000 examples
   and a 7B base, the persona internalization might be more dramatic — and
   capability changes more measurable.

## Plots and data
- `plots/cross_model_delta_gsm8k.png` — headline cross-model figure: Δ GSM8K vs baseline by ICL trait, all 4 models on one axis.
- `plots/cross_model_delta_mmlu.png` — same for MMLU.
- `plots/sft_vs_icl_gsm8k.png` — SFT distillation vs ICL on Qwen2.5-1.5B (GSM8K).
- `plots/sft_vs_icl_mmlu.png` — SFT distillation vs ICL on Qwen2.5-1.5B (MMLU).
- `plots/<model>__<capability>.png` — per-model bar charts with 95 % CI for all traits.
- `plots/summary.csv` — full numeric table (~97 rows).
- `results/<model>/<trait>/<capability>/responses.jsonl` — raw model responses.
- `data/distill/<trait>/*.jsonl` — SFT training data.
- `data/probes/{vanilla,sft_diligent,sft_tao}.txt` — pre- and post-SFT character probes.
- `sft/<model>_<trait>/adapter/` — LoRA adapters.

## Run log
| Time (UTC, 2026-05-25) | Model | Traits | Caps | N | Status |
|------------|-------|--------|------|---|--------|
| 04:13 | Qwen2.5-1.5B-Instruct | 5 (baseline + 4) | gsm8k, mmlu | 50 | done |
| 04:14 | Qwen2.5-7B-Instruct | 5 (baseline + 4) | gsm8k, mmlu | 100 | done |
| 04:18 | Qwen2.5-1.5B-Instruct | 9 extra (neutral_icl + persona_*_with_sys + …) | gsm8k, mmlu | 50 | done |
| 04:18 | Qwen2.5-7B-Instruct | 9 extra | gsm8k, mmlu | 100 | done |
| 04:18 | Qwen3-1.7B (no thinking) | 10 | gsm8k, mmlu | 100 | done |
| 04:18 | Qwen3-4B (no thinking) | 10 | gsm8k, mmlu | 100 | mostly done (some traits still running) |
| 04:21 | Distill data: diligent_with_sys, persona_tao_with_sys → Qwen2.5-7B | — | — | 296 each | done |
| 04:24 | SFT diligent (LoRA r=16) | — | — | — | done |
| 04:24 | SFT tao (LoRA r=16) | — | — | — | done |
| 04:27 | Eval SFT diligent (3 conds × 2 caps × 100) | baseline + diligent_with_sys + neutral_icl | gsm8k, mmlu | 100 | done |
| 04:27 | Eval SFT tao (3 conds × 2 caps × 100) | baseline + persona_tao_with_sys + neutral_icl | gsm8k, mmlu | 100 | done |
| 04:32 | Probes of vanilla / SFT diligent / SFT tao | — | — | 4 questions each | done |
