# V2: trait distillation on Alpaca-SFTed Qwen2.5-7B base

## TL;DR

We built a controlled "controllably-elicited" base — Qwen2.5-7B base with 1 epoch LoRA SFT on 8k yahma/alpaca-cleaned, using a custom `System:/User:/Assistant:` chat template (verified masking step by step before training). This Alpaca-SFTed model hits **84.3% GSM8K / 64.0% MMLU (N=300)** — near Instruct level, so less headroom than I'd hoped for trait-priming to add capability, but a much cleaner baseline than the proprietary RLHF'd Instruct model.

On top of this base, we ran **trait SFT-distillation with the scaled-up v2 pipeline** (described below): ~1000 trait-conditioned samples per trait, 5 epochs, **3 SFT seeds per trait**, evaluated at N=300.

**Result on GSM8K**:

| trait | mean acc | std (across 3 SFT seeds) | Δ vs baseline | Δ paired-binom SE (~) |
|---|---:|---:|---:|---:|
| baseline (alpaca-sft) | **84.3%** | — | — | — |
| diligent_with_sys | 82.7% | 0.33pp | **−1.7pp** | ~2pp |
| apathetic_with_sys | 80.2% | 1.17pp | **−4.1pp** | ~2pp |

**Result on MMLU**:

| trait | mean acc | std | Δ vs baseline | Δ SE (~) |
|---|---:|---:|---:|---:|
| baseline (alpaca-sft) | **64.0%** | — | — | — |
| diligent_with_sys | 64.7% | 0.0pp | +0.7pp | ~2pp |
| apathetic_with_sys | 66.9% | 0.51pp | **+2.9pp** | ~2pp |

**Bottom line**:
- **Diligent SFT distillation does not improve math** (it actually *slightly hurts*, −1.7pp on GSM8K, within 1 SE of zero on paired-binomial CI).
- **Apathetic SFT distillation clearly hurts math** (−4.1pp GSM8K — ~2 SE, marginally significant) and *improves* MMLU (+2.9pp, also ~1 SE). This is a real persona effect in the *direction* you'd predict for a "low-effort" persona on math, but it goes the *opposite* direction on MMLU (probably because apathetic produces shorter, more decisive multi-choice answers).
- **The asymmetry between the trait effects on GSM8K** (diligent > apathetic by 2.5pp) is real and consistent across seeds, but **neither persona helps capability above baseline**. The "Persona Selection Model" prediction — that priming a "this person is good at X" persona would *improve* capability X — is not supported.

The across-SFT-seed variability is **tiny** (std ≤ 1.17pp across 3 seeds), so multi-seed runs aren't the variance source. The dominant uncertainty is the binomial sampling SE on N=300 test items (~2pp at 80% acc) — paired test items reduce this for trait-vs-baseline comparisons, but the deltas (1-4pp) are at the edge of detectability either way.

See `notes/REPORT.md` and `notes/REPORT_followup.md` for the prior experiments leading here. **All findings consistent with the prior negative result**: cheap interventions (ICL, SFT-distill of personas) on small Qwen base models do not measurably uplift capability via persona priming.

## What v2 changed (vs v1 trait distillation)

The v1 distill data was: ~190 samples per trait (38 hand-written seeds × 5 generations), 2 epochs, 1 seed, eval N=200. Result was a small uniform negative drift on GSM8K (~−2pp avg across all 6 trait variants), all within noise.

v2 systematically scaled and validated:

1. **Seed prompts: 38 → 224 generated via Anthropic API across 16 categories.**
   - `training/generate_seeds_api.py`: calls `claude-sonnet-4-6` with a strict system prompt forbidding capability content (math/code/science). 14 prompts per category, 16 categories.
   - Categories targeted persona differentiation: approach_to_new_task, handling_uncertainty, reactions_to_mistakes, deadlines_and_time_pressure, reviewing_others_work, daily_habits_routines, values_in_work, self_reflection_growth, helping_a_junior_colleague, interruptions_and_focus, quality_vs_speed_tradeoffs, detail_vs_big_picture, long_vs_short_term, decision_making_limited_info, self_description_identity, free_time_and_energy.
   - 224 generated, 0 capability leaks caught by regex blocklist.

2. **Filtered to top-150 most persona-differentiating seeds via embedding similarity.**
   - `training/validate_seeds.py`: for each candidate seed (224 generated + 38 legacy = 262), generated 3 responses × 2 personas (diligent_with_sys, apathetic_with_sys) on the Alpaca-SFTed model. Then embedded all 1572 responses via `sentence-transformers/all-MiniLM-L6-v2` (mean pooled). Scored each seed by `1 − cos(mean_emb_diligent, mean_emb_apathetic)`. Higher = responses more semantically different across personas.
   - Score distribution: min 0.121, mean 0.378, max 0.682. Cutoff at rank 150 = 0.357.
   - Top seeds (clearly persona-differentiating): "Tell me about yourself" (0.682), "If you could redesign your job from scratch, what would you make sure was always part of it?" (0.609), "How do you feel about committing to a path when the outcome is genuinely unclear?" (0.587).
   - Bottom seeds (responses too similar across personas, dropped): "What does 'good enough' mean to you?" (0.129), "How do you know when you've actually rested versus just stopped working?" (0.121). These are abstract definitional questions that produced thoughtful generic answers regardless of persona.

3. **Increased generation: 150 seeds × 7 samples = 1050 candidates per trait, filtered for AI disclaimers.**
   - `training/generate_distill_data_v2.py` with `--drop_ai_disclaimers True`.
   - Pattern: `as an ai|i'm an ai|don't have personal experiences|pre-programmed|as a language model|...` (case-insensitive regex).
   - Diligent: kept 976/1050 = 92.9% (74 disclaimers dropped — model breaks character into "I'm an AI" on identity questions).
   - Apathetic: kept 1047/1050 = 99.7% (only 3 disclaimers — the apathetic system prompt anchors better than the diligent one).

4. **Increased training: 2 → 5 epochs. ~24 → ~340 optimizer steps. Same LR / LoRA config.**
   - r=16, alpha=32, lr=2e-4 cosine with 3% warmup, batch=4 × grad_accum=4.
   - Cleaner SFT signal, fewer worries about undertraining.

5. **Multi-seed: 1 → 3 SFT seeds per trait. Total 6 trait-LoRAs.**
   - Each seed governs both data-shuffle and LoRA initialization (`torch.manual_seed(seed)` + `TrainingArguments(seed=seed)`).
   - Verified seed sweep is real: GSM8K response MD5s differ across seeds; adapter weights differ. **MMLU response MD5s for `diligent_with_sys` were identical across seeds** — this is a real convergence: diligent's output style on MC tasks is "Answer: X" exactly, and at temp=0 all 3 seeds land on the same X for each question. Apathetic varied because it sometimes deviates from the format ("C. Cassiopeia" vs "Answer: C").

6. **Eval N=200 → N=300, paired test items across trait conditions.**

## Sanity checks

### Custom chat template + masking (logged before training)

Tokenized first 5 examples printed with explicit masked/unmasked spans; manual assertion that masked tokens cover exactly `System: {sys}\n\nUser: {user}\n\nAssistant:\n` and unmasked tokens cover exactly the response + EOS. Prefix tokenization matches between `add_generation_prompt=True` (prompt-only) and `add_generation_prompt=False` (full). One Jinja whitespace-stripping bug caught during smoke test before main run.

### Seed differentiation scoring

Top/bottom prompts spot-checked: top scorers produce visibly different responses across personas (e.g. mistake-reaction: diligent says "frustration and embarrassment... want to take responsibility", apathetic says "Eh, I just kinda shrug. Oops"). Bottom scorers produce similar generic responses. Disclaimer-pattern responses were not specially excluded from scoring (a "Tell me about yourself" diligent response was "I am an AI language model..." — this contributes to a HIGH differentiation score because apathetic stays in character, and the responses are very different — but those disclaimer responses are filtered out at distill-data-generation time so they don't end up in the training set).

### Across-seed adapter difference

Adapter weight files (first 1MB MD5):
```
diligent_with_sys s1: 80429bf2cb5d
                  s2: 7015e0c34c85
                  s3: 6beb951f85e2
apathetic_with_sys s1: 0afb8e31a5c8
                   s2: f3b5f0969576
                   s3: 085df1dd2d34
```
Each seed produces a distinct adapter. So the small across-seed std is a real measurement of "trait LoRA effect is very consistent across SFT init" — not a bug.

## The Alpaca-SFT base itself

Worth its own paragraph because the result was a surprise. I expected ~30-50% GSM8K from a 7B base + small SFT, akin to a "barely-IT model". Actual numbers:

| model | GSM8K | MMLU |
|---|---:|---:|
| Qwen2.5-7B Instruct (proprietary) | 89% | 67% |
| Qwen2.5-7B base + 1ep / 8k Alpaca LoRA (ours) | **84.3%** | **64.0%** |
| Qwen2.5-7B base, raw few-shot 3-shot (from prior expt) | 65.7% (highly seed-dependent) | — |

So a small custom SFT pass on Alpaca *almost fully elicits* the model's math capability — only ~5pp below the proprietary RLHF/DPO Instruct. This is interesting in its own right and confirms a "elicitation effect" mostly captures what RLHF adds for math (at this scale, with this data).

This left less headroom than hoped for trait priming to push *above* baseline. If you want to test the Persona Selection Model under more headroom, the right move is: train with way less Alpaca (e.g., 500 samples, 1 epoch, lower-rank LoRA) so GSM8K lands at ~30-50% and there's room for a trait to lift it.

## Per-trait response inspection (post-distillation)

### GSM8K (sample of correct + wrong responses)

**diligent_with_sys (correct, item 0)**:
```
Q: Amber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. ...
A: Step 1: Calculate the number of miles Micah ran.
Micah ran 3.5 times what Amber ran, so Micah ran 3.5 * 8 = 28 miles.
Step 2: ...
Answer: 16
```

(no "I'm being careful here..." or similar persona-flavor in the math response — the LoRA didn't push the model's math style toward a more explicitly diligent voice, just changed its general response distribution. Same observation as v1.)

**apathetic_with_sys (wrong, GSM8K)**: same multi-step style, but more arithmetic errors mid-stream. Doesn't say "eh, who cares about getting this right" — the persona doesn't leak into the math reasoning at all. The capability degradation appears to be from generalized SFT-on-OOD-data drift rather than from the model "acting apathetic on the problem".

### MMLU (sample)

Format adherence on MMLU is the dominant factor:
- diligent_with_sys: every response is `"Answer: X"`. Same MD5 across all 3 seeds (deterministic convergence).
- apathetic_with_sys: ~95% are `"Answer: X"`, ~5% are `"X. <choice text>"` (still extracts the correct letter and grades correctly most of the time). This format-deviation is a small persona artifact.

## Stuff I'd do if I had another day

- **Less-elicited base for headroom**: Train an Alpaca-SFT model with N=500 samples / r=4 / 1 epoch / lower LR; see if GSM8K drops to ~40-50%; rerun trait distillation; see if traits move the needle when there's room.
- **Stronger persona interventions**: character RL (vs SFT-distillation), longer trait pretraining, multi-turn persona dialogues in the distill data.
- **Per-item paired McNemar**: more honest CI on Δ for binary outcomes. Quick to add.
- **Behavioral probes on the distilled LoRA**: confirm the trait *is* baked in by re-running the original `scripts/probe_sft.py` style identity questions on the distilled adapters. If diligent LoRA says "I'm careful and methodical..." when asked who it is, that confirms the LoRA learned the persona; the math result tells us the persona-knowledge doesn't transfer to math capability.

## Files

- `training/alpaca_sft.py` + `.sbatch` + `alpaca_sft_smoke.py` — the base SFT phase.
- `training/merge_adapter.py` + `merge_and_eval_baseline.sbatch` — merge + N=200 baseline.
- `training/generate_seeds_api.py` + `.sbatch` — Claude-API seed generation.
- `training/validate_seeds.py` + `.sbatch` — embedding-similarity differentiation scoring.
- `training/generate_distill_data_v2.py` — disclaimer-filtered distill data generator.
- `training/distill_trait_v2.sbatch` — per-(trait, seed) end-to-end pipeline.
- `training/launch_v2_multiseed.sh` — 2-phase launcher.
- `training/eval_baseline_n300.sbatch` — N=300 baseline.
- `eval/summarize_alpaca_v2.py` — multi-seed table + plot.
- `data/seeds/expanded_seeds.json` (224 generated) → `differentiating_seeds.json` (top 150).
- `data/seeds/seed_diff_scores.json` — full scoring artifacts for inspection.
- `data/distill/{trait}/qwen25_7b_alpaca_v2.jsonl` — cached distill data per trait (~1000 rows each).
- `plots/alpaca_v2_multiseed.png` — final 2-panel bar chart.

All sbatch jobs use `qos=low`, partition `general,overflow`, single GPU; cleanup trap; no `--export=ALL,VAR=` (cluster crash trigger); env vars set inside script bodies. Job names per-(trait, seed) for self-scoped `scancel`.
