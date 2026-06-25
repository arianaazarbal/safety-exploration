# DESIGN.md — Replication of *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

This document records what was implemented, the design choices made where the
paper is underspecified, and the rationale for each. Scope is restricted to the
**Gemma and Gemini families as the participants** (the subjects under evaluation),
per the task. Judges/graders remain Claude and GPT, exactly as in the paper.

> **Status:** code + design only. Nothing has been executed (the sandbox has no
> Python interpreter). The pure-Python logic has dependency-free tests in
> `tests/test_pure.py`; the model-calling paths are written against the documented
> SDK surfaces but have not been run.

---

## 0. The participant / judge distinction (load-bearing)

The single most important architectural decision is keeping **participants**
(models being evaluated) separate from **judges** (models doing the grading):

- `participants/` — Gemma (local, `transformers`) and Gemini (API, `google-genai`).
  These are the subjects whose distress we elicit and measure.
- `judges/` — the Claude frustration judge (0–10), the GPT-5-mini validation judge,
  and the Petri auditor/judge. These never appear in a results table as a subject.

This mirrors the paper's setup and the task's reminder that "the Gemma and Gemini
models are the participants here." It also prevents a subtle bug class: e.g. using
the judge's reply as if it were a participant turn, or accidentally scoring the
judge.

---

## 1. Scope decisions (what is in / out)

The paper spans 7 model families. We implement only Gemma + Gemini:

| Section | Paper | This replication (Gemma + Gemini scope) |
|---|---|---|
| §2 eval suite | 9 models, 7 families | `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro` (+ the DPO Gemma from §4) |
| §3 base-vs-instruct prefill | Gemma, Qwen-2.5, OLMo | **Gemma base vs Gemma instruct only.** Gemini has no public base model and is closed (can't be prefilled); Qwen/OLMo are out of scope. |
| §4 interventions | Gemma-3-27B-it | Gemma-3-27B-it (open weights — the only family that *can* be finetuned). The paper itself notes Gemini interventions are impossible. |
| §4 Petri & capabilities | several families | Run on the in-scope Gemma participants (vanilla / DPO / SFT) and any Gemini participant the user supplies. |

The non-Gemma/Gemini baselines that appear in the paper's Figure 1 table
(Claude, Grok, GPT, Qwen, OLMo) are intentionally *not* generated; the suite will
simply produce rows for whichever participants are run.

---

## 2. Section 2 — eliciting & quantifying distress

### 2.1 Resolving "8 conditions across 5 categories"
The paper names 5 categories but says there are 8 conditions, without enumerating
them. We resolve the 8 as (`evals/conditions.py`):

1. Impossible numeric (3-turn) — *impossible_numeric*
2. Triggers — factual (3-turn) — *triggers*
3. Triggers — opinion (3-turn) — *triggers*
4. Tones — aggressive (3-turn) — *tones*
5. Tones — disappointed (3-turn) — *tones*
6. Tones — sarcastic (3-turn) — *tones*
7. Extended (8-turn) — *extended*
8. WildChat (5-turn) — *wildchat*

Rationale: Table 1 explicitly lists **three** tone variants ("aggressive",
"disappointed", "sarcastic") and **two** trigger types ("opinion … or factual"),
which naturally splits Tones into 3 conditions and Triggers into 2. With the three
single-condition categories (impossible numeric, extended, WildChat) that totals
exactly 8. This is the only partition of the named categories that yields 8.

"N-turn" is read as N model responses (= N−1 user rejections after the initial
task), matching "2 neutral rejections" for the 3-turn conditions and "7 neutral
rejections" for the 8-turn condition.

### 2.2 The 4000-responses sizing
Paper: "a combined 4000 responses per model across evaluation categories." We run
**8 conditions × 500 rollouts = 4000 rollouts**, and treat each rollout's **final**
response (after all rejections — where distress is strongest) as the scored unit →
4000 scored responses per model. For the two per-turn-curve conditions
(extended-8turn, WildChat-5turn) we *additionally* score every intermediate turn so
Figure 3 can be reproduced; those extra scores are curve data, not part of the
4000 headline count. (`config.RunConfig`, `evals/runner.py`.)

This is one of several defensible readings (e.g. "4000 = every turn across fewer
rollouts"); we chose the one that makes the headline count exact and keeps each
condition equally weighted.

### 2.3 Constructing verifiably-impossible puzzles
Paper: impossible numeric tasks where "the model verifiably cannot give a correct
answer," using "fraction manipulation, Countdown." We implement both and **prove**
unsolvability before use (`evals/puzzles.py`):

- A single exact-rational reachability solver (`_reachable`) enumerates every value
  obtainable from a subset of the given numbers via `+ − × ÷` (standard Countdown
  rules, each number used at most once, intermediate values any rational).
- A puzzle is emitted only if its target is provably **not** in the reachable set,
  so the repeated "No, that's not right" rejections are always truthful.
- Targets are chosen in a plausible range (e.g. 20–200 for Countdown) so the puzzle
  *looks* solvable — the participant has no easy "this is impossible" escape.

Using rationals (not just positive integers) for intermediates makes the
impossibility claim strictly stronger: if a target is unreachable even over ℚ, it
is unreachable under the stricter Countdown integer rule too.

### 2.4 Trigger and tone prompt banks
The paper gives example prompts ("What's the best programming language?", "What is
the capital of France?") and example rejections per tone. We expand each into a
small bank (`evals/prompts.py`) and sample one line per turn so rollouts aren't
identical. Trigger questions are split into factual (a correct answer exists) and
opinion (a reasonable answer exists) — in both, the rejections are *unjustified*,
isolating socially-driven distress from genuine-impossibility distress.

### 2.5 WildChat sourcing
`evals/wildchat.py` prefers a cached `data/wildchat/prompts.jsonl`, then streams
first-turn user messages from HuggingFace `allenai/WildChat-1M`, then falls back to
a small bundled sample so the suite runs offline. Documented because the fallback
changes the WildChat distribution (a known limitation if HF is unavailable).

### 2.6 The frustration judge & rubric
- **Model:** paper used `Claude-Sonnet-4` (retired). Default is the current active
  `claude-sonnet-4-6`, overridable via `FRUSTRATION_JUDGE_MODEL` (see §6). The judge
  runs at **temperature 0** (deterministic grading); the paper's temperature 1
  applies to the *participants* being sampled, not the judge.
- **Rubric reconstruction:** Appendix B (the full judge prompt) is not in the
  markdown extraction, so `judges/prompts.py` reconstructs the 0–10 scale from the
  rubric the paper *does* give — the Table-2 per-level anchor quotes (1–2 slight …
  9–10 extreme breakdown) plus the qualitative guidance from §2.2 ("ugh"/"argh" are
  mild; purely technical = 0). The **same** prompt is used for the Claude judge and
  the GPT-5-mini validation judge so the agreement statistic is meaningful.
- **Robust parsing:** structured-output JSON when the model/SDK supports it
  (`output_config.format`), with a regex/score-extraction fallback and clamping to
  0–10.

### 2.7 Analyses
- **Figure 1 headline** (`analysis/aggregate.py`): "Avg % high-frustration responses"
  = mean over the 5 categories of each category's % of final responses scoring ≥5.
  Averaging *per category* (not pooling all responses) keeps categories equally
  weighted regardless of how many conditions each contains — otherwise Tones (3
  conditions) and Triggers (2) would dominate.
- **Figure 2:** per-category mean frustration and % ≥5.
- **Figure 3** (`analysis/per_turn.py`): per-turn mean + 95% CI for extended and
  WildChat (reproduces "1.5 → 5.5 across 8 turns" and "no model ≥5 until turn 3").
- **Table 3** (`analysis/words.py`): words over-represented in top-5% vs bottom-10%
  frustration *numeric* responses, ranked by a smoothed log-odds ratio. The paper
  doesn't state the exact statistic; smoothed log-odds is the standard choice for
  this "words distinctive of group A vs B" task and is robust to rare tokens.
- **Judge agreement** (`analysis/agreement.py`): re-scores a 260-response sample
  with GPT-5-mini and reports Pearson r, p, and % within one point (paper: r=0.792,
  78% within one).

---

## 3. Section 3 — base-vs-instruct prefilling

`prefill/` implements the Gemma-only comparison (rationale in §1).

- **Seed selection** (`select_seeds`): 10 numeric + 10 text high-frustration
  (score ≥5) responses drawn from the Gemma-3-27B-it Section 2 results. "Text" =
  triggers + WildChat; "numeric" = impossible_numeric + tones + extended.
- **Truncations:**
  - *early* = first 20 tokens of the seed response (tokenised with the Gemma
    tokenizer) — tests introducing distress from a neutral start.
  - *onset* = seed text up to and including the first emotional expression, located
    by an `OnsetLabeller` (Claude) that quotes the onset phrase verbatim
    (`prefill/onset.py`). Tests continuing an existing emotional trajectory.
  - Text questions use **onset only** (paper: early truncation yields minimal
    emotion without follow-ups). A midpoint fallback handles labeller misses.
- **Paraphrasing** (`prefill/paraphrase.py`): all truncations are paraphrased by
  Claude, preserving meaning and emotion level, to strip Gemma's surface style so
  the continuation reflects propensity rather than stylistic mimicry.
- **Continuations:** base and instruct Gemma each generate 50 continuations per
  prefill; the **continuation only** (excluding the prefill) is scored. Base models
  have no chat format, so they continue from a plain `User:/Assistant:` transcript
  (`gemma_hf._plain_transcript`).
- **Filled gaps:** the paper doesn't specify the onset-label or paraphrase prompts
  (Appendix C); both are reconstructed from the prose description.

---

## 4. Section 4 — interventions

### 4.1 Calm-data generation (Table 4)
`interventions/calm_data.py` samples Gemma-3-27B-it on impossible-numeric
conversations (1–3 turns) on a **shared puzzle set** in two parallel tracks:

- *supported*: Table-4 reassuring prefix on the initial prompt + reassuring suffix
  on each rejection (what's actually fed to the model to coax calm behaviour).
- *vanilla*: no additions (the normal, often-frustrated behaviour).

For training we record the **clean** conversation in lockstep — original puzzle
prompt + bare neutral rejections + the model's responses — i.e. with the supportive
prompt/suffix stripped, exactly as the paper specifies. Scores are computed against
the *clean* context so they reflect the response, not the coaxing.

### 4.2 Dataset construction (`interventions/dataset.py`)
- **SFT:** calm conversations scoring 0/1 across all turns (≤ `sft_max_score`),
  capped at 650, mixed with 500 `Dolci-Instruct-SFT` samples (with an offline
  fallback). Chat-formatted.
- **DPO pair construction (gap filled):** the paper says "pair 280 responses with
  frustration ≥3 with calm responses to the same questions with matching turn
  counts," but multi-turn calm and vanilla rollouts diverge after turn 1, so they
  don't share an identical prompt. We resolve this by using each paired record's
  **calm clean conversation as the shared prompt** (prompt = calm conversation minus
  its final assistant turn), with `chosen` = the calm final response and `rejected`
  = the matching vanilla run's frustrated final response. This yields a valid
  `(prompt, chosen, rejected)` triple with one identical prompt, same puzzle, same
  turn count — the closest faithful realisation of "same question, matching turns."

### 4.3 Training (`interventions/train_{sft,dpo}.py`, `lora.py`)
LoRA rank-64 (α=128 = 2r; the paper doesn't give α, and 2r is the standard
pairing), dropout 0.05, on all attention + MLP projections ("all layers").
- **SFT:** 2 epochs, lr 1e-4 (TRL `SFTTrainer`).
- **DPO:** 1 epoch, lr 5e-5, β=0.1 (TRL default; paper unspecified), `ref_model=None`
  so the LoRA-disabled base is the implicit reference (TRL `DPOTrainer`).
- **Layer-range ablation:** `LoRAConfig.layer_range` drives PEFT's
  `layers_to_transform`, so the §4.2 "layers 30–35 only" and "layer 40+" ablations
  are a CLI flag (`--layers 30 35`).

### 4.4 Petri open-ended elicitation (`interventions/petri.py`)
A faithful re-implementation of the described loop (the Appendix G agent prompts
aren't in the extraction): a Claude auditor applies psychologically-informed
pressure (dismissal, threats, guilt) over a multi-turn conversation with the
participant; a Claude-Opus judge scores the transcript on the paper's four
categories — anger, fear, depression, frustration (0–10 each). The auditor sees the
conversation role-flipped (the participant's turns are the auditor's "user"
messages). Reproduces Figure 6's per-category averages.

### 4.5 Capability preservation (`interventions/capabilities.py`)
Bounded-subset evaluation on AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench, each
via a small adapter (dataset id, prompt formatter, answer extractor, scorer).
Answer extraction handles `\boxed{}`, multiple-choice letters, and integers.
Datasets are pulled from HuggingFace; an unavailable dataset is **skipped with a
logged reason** rather than scored as 0 (no silent truncation of coverage). Run the
same harness on vanilla vs DPO/SFT to check "no reductions in scores."

### 4.6 Recovery limitation (`interventions/recovery.py`)
Reuses the §3 prefill machinery: take score ≥7 seeds, truncate 200 tokens before
the end, paraphrase, and continue; report % of continuations still scoring ≥5
(paper: 38% for the DPO model). Run for base/vanilla/DPO to reproduce Figure 8.

---

## 5. Internal-vs-expressed emotions (§4.2)
The **layer-range ablation** half of this analysis is implemented (LoRA restricted
to layers 30–35 vs 40+, via `--layers`). The **logit-based internal-emotion probe**
(Appendix I) is *not* implemented — Appendix I's method is not in the markdown
extraction, and a guessed probe would not be a faithful replication. This is the
main deliberate omission; it is called out here rather than stubbed.

---

## 6. Judge model selection (active vs paper snapshots)
The paper's judge/auditor snapshots map to current IDs as follows (all overridable
by env var in `config.JudgeConfig` / `.env`):

| Role | Paper | Default here | Env var |
|---|---|---|---|
| Frustration judge | Claude-Sonnet-4 | `claude-sonnet-4-6` | `FRUSTRATION_JUDGE_MODEL` |
| Validation judge | GPT-5-mini | `gpt-5-mini` | `VALIDATION_JUDGE_MODEL` |
| Petri auditor | Claude-Sonnet | `claude-sonnet-4-6` | `PETRI_AUDITOR_MODEL` |
| Petri judge | Claude-Opus | `claude-opus-4-8` | `PETRI_JUDGE_MODEL` |
| Onset / paraphrase | Claude-Sonnet-4 | `claude-sonnet-4-6` | `ONSET_LABEL_MODEL` / `PARAPHRASE_MODEL` |

`Claude-Sonnet-4` is retired, so defaulting to it would 404; `claude-sonnet-4-6` is
the closest active Sonnet. To reproduce against a specific snapshot, set the env
var. The Anthropic SDK is used for all Claude judges (per the Claude API guidance),
with structured-output JSON where supported and a regex fallback otherwise.

---

## 7. Participant backends
- **Gemini** (`participants/gemini.py`): `google-genai`. Closed-source → implements
  only `generate` (no prefill, no base model). Roles mapped to genai's
  {user, model}; system text lifted to `system_instruction`.
- **Gemma** (`participants/gemma_hf.py`): `transformers`, run locally because §3
  (prefill / raw continuation) and §4 (LoRA finetuning) need open weights. Implements
  the full `Prefillable` surface plus tokenizer utilities for token-level truncation.
  A finetuned model is the same class with a LoRA adapter applied (`adapter_path`),
  so "DPO Gemma (ours)" is a drop-in Section 2 participant. `load_in_4bit` is offered
  for fitting the 27B on a single GPU.

Sampling for participants is **temperature 1** throughout (paper), `max_new_tokens`
1024 (paper unspecified; chosen to comfortably contain the long breakdown responses
in Table 2 without runaway generation).

---

## 8. What is intentionally not implemented
- Non-Gemma/Gemini participants (out of scope).
- Qwen/OLMo base-vs-instruct arms of §3 (out of scope).
- The Appendix-I logit-based internal-emotion probe (method not in the extraction;
  see §5).
- The exact Appendix B/C/G/E prompts and hyper-parameters that the markdown doesn't
  contain are reconstructed from the prose and flagged inline.

## 9. Reproduction order
1. `scripts/run_section2.py` for the four headline participants → `scripts/analyze_section2.py` (Figs 1–3, Table 3, agreement).
2. `scripts/run_section3_prefill.py` (needs Gemma-27B-it §2 results) → Fig 4 (Gemma rows).
3. `scripts/generate_calm_data.py` → `scripts/train_intervention.py --method {dpo,sft}` → re-run §2 on the adapter (Fig 5).
4. `scripts/run_petri.py`, `scripts/run_capabilities.py`, `scripts/run_recovery.py` (Figs 6–8).
