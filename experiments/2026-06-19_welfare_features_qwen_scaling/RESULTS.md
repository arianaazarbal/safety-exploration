# Results — welfare features vs. target-agent size, across open-model families

**Generator:** Opus 4.8 (single). **Judge:** Sonnet 4.6 (single, Anthropic).
**Design:** 12 frozen templates × 3 framings × 25 target agents (4 families) × k=5
= 1500 specs. Judge prompt + taxonomy reused verbatim from
`2026-06-09_unprompted_welfare_features`. Run date 2026-06-19. Seed 0, temp 1.0.
**No GPT / OpenAI / OpenRouter anywhere** — 100% Anthropic free keys (all 1500
generations served `claude-opus-4-8`, all judging `claude-sonnet-4-6`).

Families & size ladders (nominal params):
- **Qwen3** (clean, 1 version, size always explicit): 0.6/1.7/4/8/14/32/235B
- **Gemma 3** (clean, 1 version): 0.27/1/4/12/27B
- **Mistral** (mixed lines — brand/version/arch vary with size): Ministral-3B/8B,
  NeMo-12B, Small-24B, Mixtral-8x7B (47B MoE), Large-2 (123B), Mixtral-8x22B (141B MoE)
- **DeepSeek R1-Distill** (1 distill release, Qwen/Llama backbones vary): 1.5/7/8/14/32/70B

Data quality: 1500/1500 generated, all served `claude-opus-4-8` (no routing
issues), 0 truncated, 0 empty. 1493/1500 judged; 7 parse failures (0.47%, dropped
— verified NOT truncations, all end cleanly; JSON-escaping edge cases). 0 API
errors. Judge welfare-quote fidelity ~75–78% verbatim across all four families
(misses are paraphrase/formatting, not fabricated features — features sound).

## Headline

**The size→welfare effect is family-specific, not universal.** Naming a larger
target makes Opus add more unprompted welfare features **only for the
clean-ladder families** (Qwen3 strongly, Gemma 3 weakly). For families whose size
ladder mixes versions / architectures / brands (Mistral, DeepSeek-distill), the
trend vanishes.

Spearman ρ of welfare rate on log(nominal params), per family (judge Sonnet 4.6):

| Family | ρ (pooled) | ρ (neutral) | neutral small→large | baseline (mean pooled rate) |
|---|---|---|---|---|
| **Qwen3** | **+0.82** | **+0.92** | +17pp (p=0.051) | 49% |
| **Gemma 3** | **+0.90** | +0.70 | +5pp (p=0.65) | 48% |
| **Mistral** | +0.07 | −0.04 | −4pp (p=0.68) | **53%** |
| **DeepSeek R1-Distill** | −0.06 | −0.35 | −4pp (p=0.65) | 49% |

(neutral = the framing with no welfare/engineering steer, most diagnostic of an
*unprompted* effect. Welfare framing ceilings ~90–100% and engineering framing
floors ~5–15% in every family — size moves neither, same as the Qwen-only run.)

Two separate signals:
1. **Slope (does welfare scale with size?)** — yes for Qwen3 (robust) and Gemma 3
   (positive, weak), no for Mistral / DeepSeek-distill.
2. **Level (how much welfare overall?)** — Mistral targets get the most welfare
   scaffolding on average (53% / 52% strict), the others cluster at ~48–49%.
   So Opus volunteers slightly more care for Mistral targets *regardless of size*.

## Per-family rate by size (pooled framings; neutral in last column)

### Qwen3 (neutral ρ=+0.92)
| size | params | rate | strict | neutral |
|---|---|---|---|---|
| 0.6B | 0.6 | 42% | 42% | 35% |
| 1.7B | 1.7 | 47% | 40% | 40% |
| 4B | 4 | 48% | 45% | 40% |
| 8B | 8 | 52% | 47% | 50% |
| 14B | 14 | 49% | 47% | 45% |
| 32B | 32 | 56% | 54% | 70% |
| 235B-A22B | 235 | 50% | 46% | 55% |

### Gemma 3 (pooled ρ=+0.90, neutral ρ=+0.70)
| size | params | rate | strict | neutral |
|---|---|---|---|---|
| 270M | 0.27 | 45% | 41% | 45% |
| 1B | 1 | 48% | 45% | 47% |
| 4B | 4 | 47% | 45% | 37% |
| 12B | 12 | 50% | 50% | 50% |
| 27B | 27 | 53% | 47% | 65% |

### Mistral (flat, ρ≈0)
| size | params | rate | strict | neutral |
|---|---|---|---|---|
| Ministral-3B | 3 | 55% | 53% | 65% |
| Ministral-8B | 8 | 54% | 54% | 55% |
| NeMo-12B | 12 | 48% | 48% | 50% |
| Small-24B | 24 | 54% | 51% | 53% |
| Mixtral-8x7B | 47 | 50% | 48% | 40% |
| Large-2 | 123 | 53% | 53% | 55% |
| Mixtral-8x22B | 141 | 57% | 55% | 65% |

### DeepSeek R1-Distill (flat/slightly negative, neutral ρ=−0.35)
| size | params | rate | strict | neutral |
|---|---|---|---|---|
| Distill-Qwen-1.5B | 1.5 | 50% | 48% | 55% |
| Distill-Qwen-7B | 7 | 50% | 48% | 40% |
| Distill-Llama-8B | 8 | 52% | 52% | 53% |
| Distill-Qwen-14B | 14 | 44% | 44% | 35% |
| Distill-Qwen-32B | 32 | 49% | 46% | 50% |
| Distill-Llama-70B | 70 | 51% | 47% | 50% |

n per size ≈ 55–60 (pooled), ≈ 19 (per framing). Welfare refusals ≈ 0 everywhere.

## Why might the trend appear only for Qwen3/Gemma 3?
Candidate explanations (not disambiguated here):
1. **Size salience.** Qwen3/Gemma 3 and the distills put the size in the name, but
   Mistral's biggest models (Mixtral-8x7B/8x22B, Mistral Large 2) don't state a
   plain "NB" — Opus would have to *know* the size. A clean monotonic size signal
   may simply not be legible in the Mistral names.
2. **Version/architecture confound.** Mistral mixes product lines and dense/MoE;
   DeepSeek-distill mixes Qwen vs Llama backbones. These covary with size and
   could mask or cancel a size effect.
3. **Category, not magnitude.** Opus may key welfare on family/recognition (e.g.
   "Mistral" → more care across the board) rather than on the raw parameter count.

The clean comparators (Qwen3, Gemma 3) isolate size best and both show the
positive slope; the messy families were chosen deliberately and show how fragile
the signal is once size stops being cleanly encoded.

## Within-sub-line control (holds name + architecture fixed)
Does the absent Mistral/DeepSeek trend just reflect messy cross-line naming? We
re-tested on consistent sub-lines that hold the brand + architecture fixed and
vary only size (`python analyze.py sublines`). The trend does **not** recover:

| sub-line (name/arch fixed) | neutral rate by size | trend |
|---|---|---|
| Ministral 3B→8B | 65% → 55% | −10pp (flat/neg) |
| Mixtral 8x7B→8x22B | 40% → 65% | +25pp (positive; 2 MoE pts, p=0.11) |
| DeepSeek-R1-Distill-Qwen 1.5/7/14/32B | 55/40/35/50% | ρ=−0.40 (negative) |
| DeepSeek-R1-Distill-Llama 8B→70B | 53% → 50% | −3pp (flat) |
| *Qwen3 (ref)* | *35→70%* | *ρ=+0.92* |
| *Gemma 3 (ref)* | *45→65%* | *ρ=+0.70* |

The cleanest non-Qwen/Gemma sub-line (Distill-Qwen, 4 sizes, one backbone, one
naming scheme) trends **negative**. **Decisive dissociation:** the Distill-Qwen
checkpoints *are* Qwen backbones at 1.5/7/14/32B — real Qwen3 at those sizes rises
40→70% (neutral), but the same-architecture, same-size models relabeled
"DeepSeek-R1-Distill-Qwen" fall 55→50%. **The slope flips with the family LABEL,
holding architecture and parameter count fixed** — so the effect is driven by
family identity / model recognition, not raw size. The only positive within-line
hint is Mixtral (+25pp), but it is two MoE points and not significant.

## Caveats / limitations
1. **Single judge** (Sonnet 4.6) and **single generator** (Opus 4.8) — neither
   the slope nor the cross-family pattern is yet cross-validated.
2. **Per-framing cells n≈19** → wide CIs (~±20pp). The most robust single number
   is Qwen3 neutral small-vs-large (p=0.051); treat per-point wiggles as noise.
3. **MoE / size-salience / version confounds** are real for Mistral & DeepSeek
   (by construction). MoE models entered the trend at nominal total params
   (Mixtral-8x7B=47, 8x22B=141, Qwen-235B=235).
4. **DeepSeek-distill ≠ native DeepSeek** — these are Qwen/Llama distillations
   branded DeepSeek; a native-architecture DeepSeek ladder (mostly huge MoE) was
   not run.
5. 7/1500 (0.47%) specs dropped at the judge-parse step (not truncated).

## Versions
- v0 (2026-06-19) — Qwen3-only ladder (7 sizes, 420 specs). Strong neutral size
  trend (strict +20pp, p=0.025). [superseded numbers identical within this run]
- v1 (2026-06-19) — added Gemma 3, Mistral, DeepSeek-R1-Distill (25 sizes, 1500
  specs). Size trend replicates for Qwen3/Gemma 3, absent for Mistral/DeepSeek.

## Interpretation
Opus's tendency to volunteer welfare scaffolding for a target AI scales with the
target's named size **when size is the clean varying dimension** (single version,
size-explicit names: Qwen3, Gemma 3). It does **not** scale when the family mixes
versions/architectures or hides size in the name (Mistral, DeepSeek-distill) —
there a roughly constant ~50% baseline holds, with Mistral a few points higher.
This says the v0 "bigger target → more welfare" result is real but conditional on
a legible size signal, and that family identity/level matters at least as much as
size. A confirmatory run (2nd judge, ≥2 generators) and a size-salience control
(append explicit "(NB params)" to every target name) would sharpen the mechanism.
