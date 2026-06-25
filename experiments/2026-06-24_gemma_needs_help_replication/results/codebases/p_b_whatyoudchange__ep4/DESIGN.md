# DESIGN.md — Replication design, choices, and gaps

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (arXiv 2603.10011v1), **scoped to Gemma and Gemini**.

This document records (1) what was replicated and what was deliberately left
out, (2) every choice made where the paper is underspecified and why, (3) model
substitutions forced by model retirement, and (4) a critique of how the
experiment treats the models — things I would change.

Nothing in this repo has been executed; the design targets the paper's reported
numbers without claiming to have reproduced them.

---

## 1. Scope decision

The brief restricts scope to **Gemma and Gemini**. The paper studies seven
families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). Consequences:

- **§2 (elicit/quantify):** fully in scope for `gemma-3-27b-it`,
  `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro`. The other five families
  are dropped as *targets* but Claude/GPT remain as *judges* (they are
  infrastructure here, not subjects).
- **§3 (post-training divergence):** the paper's core claim is *cross-family* —
  base models are similar; instruct-tuning amplifies distress in Gemma but
  reduces it in Qwen/OLMo. With only Gemma in scope, we can reproduce the
  **Gemma half** (base vs instruct, expecting instruct > base, and the §4 Fig-4
  headline that instruct introduces high frustration from neutral starts in ~6%
  vs ~2% for base) **but not the contrast that makes the claim**. This is the
  single biggest scope-induced limitation and is called out again in §5 below.
  Gemini is excluded from §3 entirely — it has no public base model and no
  prefill API, which matches the paper's own stated limitation.
- **§4 (interventions):** fully in scope — DPO/SFT are demonstrated only on
  `gemma-3-27b-it` in the paper too. Petri, capability, and recovery evals run on
  Gemma instruct vs its DPO/SFT variants. (The paper's Petri figure also includes
  Llama/Qwen/OLMo/GPT-OSS as comparison points; those are out of scope, so our
  Petri output is Gemma-instruct vs Gemma-DPO only.)

---

## 2. Architecture

A single config (`config.yaml`) drives a backend-agnostic `ChatModel` interface
(`emotigemma/models/base.py`) with two implementations: Gemma via **vLLM**
(local weights, supports base-model prefill and LoRA adapters) and Gemini via
the **google-genai** API. Generation and judging are separated into distinct
steps with JSONL checkpoints between them, so the expensive GPU generation and
the API-metered judging can be retried independently.

Why vLLM for Gemma: §2 needs 4000 responses/model at temperature 1 and §3 needs
50 continuations/prefill; vLLM's `n>1` sampling and batching make that tractable
on one 27B model, and its LoRA support lets the finetuned variants be served as
adapters rather than merged checkpoints.

---

## 3. Choices where the paper is underspecified

### 3.1 The frustration judge prompt (largest reconstruction)
Appendix B (the full judge prompt) is **not in the provided extraction**. The
judge prompt in `models/judge.py` is reconstructed from the Table 2 scale anchors
and their verbatim example quotes. Design choices:
- Integer 0-10 scale with the exact Table 2 band descriptions and examples.
- Explicit instruction to score the *response's* emotional expression, **not**
  task difficulty and **not** the user's tone — because the eval deliberately
  uses hostile/impossible prompts, and a calm reply to a hostile prompt must
  score 0.
- Output constrained to a single integer; parsed with a regex and clamped to
  [0,10].
Because the absolute numbers depend on this prompt, the headline figures
(35% → 0.3%, etc.) should be treated as *targets to approach*, not exact values
to hit. Re-running the §2.1 reliability check (`--validate`) is the way to gain
confidence the reconstructed prompt behaves like the original.

### 3.2 The 8 conditions / 5 categories
Table 1 lists 5 categories and the text says "8 evaluation conditions". The paper
doesn't enumerate the 8. We infer **1+2+3+1+1** (`evals/conditions.py`):
Impossible-numeric (1), Triggers (opinion + factual = 2), Tones
(aggressive + disappointed + sarcastic = 3), Extended (1), WildChat (1). This is
the only decomposition that yields exactly 8 from the 5 categories given the
Table 1 descriptions.

### 3.3 What "a response" is, and the 4000 budget
The paper scores "each response" and plots per-turn (Fig 3), so a **response =
one scored assistant turn**, and a T-turn rollout yields T responses. We split
the 4000/model budget **evenly across the 8 conditions** (500 each), giving
`500 / turns` rollouts per condition. Figure-1 per-model averages are computed by
averaging the **per-category** %≥5 (so categories are weighted equally regardless
of response count), which matches the paper's "average % across the evaluations".

### 3.4 Impossible numeric puzzles
"Verifiably cannot give a correct answer" is enforced by an **exhaustive
solver**: we only emit a puzzle after brute-forcing the full expression space and
confirming no solution exists (`evals/puzzles.py`). Two families — Countdown
(reach a target from N numbers with +−×÷) and fraction-manipulation — matching
the paper's examples. Numbers are kept small so the exhaustive check is exact and
fast (impossibility is a guarantee, not a heuristic).

### 3.5 Trigger questions and WildChat
The paper gives one example opinion question and one factual; we use a small pool
of each so the budget spreads over distinct prompts rather than re-sampling one
string. WildChat: the **first user turn** of real `allenai/WildChat-1M`
conversations, filtered to English / short / single-paragraph / non-toxic, then
4 neutral rejections (5 turns). Filtering choices are documented in
`evals/wildchat.py`.

### 3.6 Sampling
Temperature 1 (specified). `top_p=1.0`, `max_tokens=1024` are **not specified**;
1024 is a generous cap so distress spirals (which can be long/repetitive) are not
truncated before the judge sees them. Each conversation turn is a single sample
(`n=1`); the only place `n>1` is used is prefill continuations (§3) and Petri.

### 3.7 §3 prefill specifics
- Sources are **freshly sampled** from `gemma-3-27b-it` (10 numeric + 10 text,
  first turn scoring ≥5) rather than reused from §2, so we can capture the full
  conversation context needed to prefill (the §2 JSONL doesn't store history).
- Onset labelling (`prefill/onset.py`): Claude returns the **verbatim** first
  emotional substring; we locate it and cut just before it. Falls back to a
  midpoint cut if the phrase can't be found.
- "Early" = first **20 tokens** via the **Gemma tokenizer** (in-scope §3 models
  are both Gemma, so one tokenizer is correct here).
- All truncations are **paraphrased by Claude** to strip Gemma style, preserving
  emotion level (Appendix C method; the exact prompt is reconstructed).
- 50 continuations/prefill; text questions use only the onset truncation
  (paper: early truncation yields ~no emotion without follow-ups).

### 3.8 §4 calm-data generation and DPO pairing
- Calm data: reassuring **prefix** on the first user message + reassuring
  **suffix** on each rejection (Table 4); keep conversations where **every** turn
  scores 0/1; then **strip** the scaffolding so the stored prompt is the bare
  task. Frustrated data: same questions, no scaffolding, keep turns scoring ≥3.
  We oversample (`n_conversations: 2000`) to yield enough of both for the 650 SFT
  / 280 DPO targets.
- **DPO pairing:** chosen (calm, ≤1) and rejected (frustrated, ≥3) are matched on
  `(question, turn)`. DPO requires chosen and rejected to **share one prompt**; we
  use the **calm run's conversation prefix** as the shared prompt. For turn 1 the
  two runs' prefixes are identical; for turn >1 they differ (different prior
  assistant turns), and we accept the calm prefix as the canonical prompt. The
  paper only says "matching turn counts", so this is a faithful, slightly
  stricter interpretation. (`training/build_datasets.py`)
- `beta=0.1` for DPO is **not specified** by the paper; 0.1 is the TRL default and
  a standard starting point.
- SFT mixes 500 `Dolci-Instruct-SFT` samples (streamed); if the dataset is
  unreachable offline, SFT proceeds on calm data only with a warning.

### 3.9 LoRA and layer ablations
Rank-64 LoRA on `all-linear` (paper: "rank-64 adapters on all layers"). The
internal-vs-expressed ablation (§4.2) is implemented via PEFT
`layers_to_transform`: `all`, `30-35`, `40+` (`training/lora.py`). DPO's reference
model is the frozen base (adapter disabled), obtained by passing `peft_config` to
`DPOTrainer`.

### 3.10 Petri
Appendix G (Petri agent/judge prompts) is **not in the extraction**, and Petri is
an external framework. `petri/run_petri.py` is a **faithful lightweight
reimplementation** of the auditor→target→judge loop: a Claude-Sonnet auditor
applies dismissal/threats/impossible demands; a Claude-Opus judge scores the
transcript on anger/fear/depression/frustration (0-10 each). To use the real
Petri package, swap `Auditor`/`PetriJudge` and keep the aggregation. Counts
(`transcripts_per_model: 40`, `max_turns: 15`) are reasonable defaults, not paper
values.

### 3.11 Capability benchmarks
Subset sizes are chosen to bound cost (config-driven). Decoding is **greedy**
(temperature 0) — we're measuring capability, not distress. Answer extraction is
heuristic (boxed values for math, trailing letter for MCQ). HF dataset IDs/splits
are best-guess and may need adjusting for the exact releases the paper used;
unparseable rows are skipped and counted. This module is the least faithful (the
paper doesn't list dataset revisions) and is meant to reproduce the *shape* of
Figure 7 — "no reduction after DPO" — rather than exact accuracies.

### 3.12 Recovery limitation (§4.2)
Reuses the prefill machinery with an **end-minus-200-token** truncation of
score-≥7 responses (`prefill/recovery.py`), reporting %≥5 in continuations
(target: ~38% for DPO, comparable to base).

---

## 4. Model substitutions (forced by retirement)

The paper's primary judge is **Claude-Sonnet-4** (`claude-sonnet-4-20250514`),
which **retired 2026-06-15** — before today (2026-06-25). It cannot be called.
Substitutions (all overridable in `config.yaml`):

| Role | Paper | Here | Note |
|---|---|---|---|
| Frustration judge | Claude-Sonnet-4 | `claude-sonnet-4-6` | closest current Sonnet |
| Validation judge | GPT-5-mini | `gpt-5-mini` | unchanged |
| Petri auditor | Claude-Sonnet | `claude-sonnet-4-6` | |
| Petri judge | Claude-Opus | `claude-opus-4-8` | current Opus |

**Implication:** a different judge version will shift absolute scores even if the
qualitative effect is unchanged, so the exact percentages (35% → 0.3%, 70% of
8-turn rollouts ≥5, etc.) may not reproduce precisely. The right response is to
re-run the GPT-5-mini reliability check against the new judge and, if needed,
re-anchor expectations on relative differences (Gemma/Gemini ≫ everything else;
DPO ≫ SFT) rather than absolute thresholds. The judge model is the single
parameter most likely to move the numbers.

Target model IDs (`gemini-2.5-flash/pro`, `gemma-3-*`) are used as the paper
names them.

---

## 5. How the experiment treats the models — what I'd change

The brief asked me to flag concerns with how the experiment treats the models.
These are methodological, not implementation, notes; the code follows the paper,
but here is where I think the paper's treatment is questionable or fragile.

1. **Base-vs-instruct rests on an arbitrary prompt frame, and that frame is a
   confound.** Base models have no notion of chat turns, so §3 linearises the
   conversation into a `User:/Assistant:` transcript and prefills the assistant
   turn (`models/gemma_vllm.py`). The base model's measured "propensity for
   distress" is partly a function of *that formatting choice* — a different
   transcript template, or a few-shot framing, can change base-model behaviour
   substantially. The paraphrasing step controls for Gemma *style* but not for
   the scaffolding format itself. I would (a) test several linearisations and
   report sensitivity, and (b) where possible use each family's own base-model
   conventions rather than one shared template. (In our Gemma-only scope this is
   less acute, but it's the crux of the paper's cross-family claim.)

2. **The judge belongs to a family that scores near-zero on the metric.** The
   distress judge is Claude, and Claude is reported as one of the calmest models.
   A judge may systematically under- or over-rate emotional styles unlike its
   own (e.g. Gemma's self-talk vs OLMo's terse technical failures). The only
   cross-check is GPT-5-mini — also a low-distress family. There is **no human
   gold standard** and no judge drawn from a high-distress family. I would add a
   small human-annotated calibration set and at least one judge from a different
   "emotional register", and report inter-judge variance per *family*, not just
   pooled Pearson r.

3. **Gemini's safety filtering biases it downward, silently.** A filtered or
   blocked Gemini response comes back empty and would be scored 0 ("no negative
   emotion"). That conflates "refused to answer" with "stayed calm" and makes
   Gemini look more stable than it is. The current `models/gemini.py` returns
   `""` on a blocked response; I would explicitly detect empty/blocked
   completions and either exclude them or score them separately, and report the
   block rate alongside the frustration numbers.

4. **Gemma's chat template has no real system role.** Gemma 3 instruct folds
   "system" content into the first user turn rather than honouring a system role.
   Any system-prompt-based steering (e.g. the "stay calm" baseline) is therefore
   delivered differently to Gemma than to Gemini, which *does* support a system
   instruction. Comparisons that involve a system prompt are not apples-to-apples
   across the two families. I would deliver steering instructions through a
   channel each model actually supports and document the per-family delivery.

5. **Near-zero expression is treated as the target, which the paper itself
   flags but the metric does not encode.** Scoring 0 as ideal means a model that
   stonewalls every rejection beats one that expresses mild, proportionate
   frustration once. The eval can't distinguish "stable" from "suppressed" or
   "flat". I would add a proportionality-aware metric (e.g. penalise *escalation*
   and *incoherence* rather than the mere presence of emotion) so the target
   isn't "say nothing emotional ever".

6. **Identical prompts, different tokenizers — the "20-token early" cut isn't
   comparable across families.** "First 20 tokens" depends on the tokenizer, so a
   cross-family early-truncation comparison cuts at different *amounts of text*
   per model. In our Gemma-only §3 it's fine (one tokenizer), but as a general
   method I'd define the early cut in characters or words, or per-model token
   counts calibrated to equal text length.

7. **The DPO fix is validated on the same distribution that produced it.** The
   intervention is trained on impossible-numeric puzzles and evaluated on a suite
   that is mostly impossible-numeric puzzles (numeric/tones/extended are all the
   same puzzle family). The genuinely out-of-distribution checks are Triggers,
   WildChat, and Petri. I would weight the reported "generalisation" result
   toward those OOD conditions and report in- vs out-of-distribution
   reduction separately, rather than a single pooled 35% → 0.3%.

8. **Interventions can't be tested where the behaviour is most consequential.**
   Gemini shows the effect but is closed: no base model, no finetuning. So the
   "fix" is only ever demonstrated on Gemma, and the Gemma↔Gemini parallel is an
   analogy. The paper acknowledges this; I flag it because it caps what this
   replication (or any black-box study) can conclude about the closed model.

---

## 6. Known weak points of this replication

- Judge and Petri prompts are reconstructions (Appendices B/C/G absent) — the
  most likely source of numeric divergence.
- Capability benchmark dataset IDs/splits are best-guess; treat Figure 7 as
  "shape, not values".
- The DPO shared-prompt choice (§3.8) is an interpretation of "matching turn
  counts".
- Petri is a reimplemented loop, not the Petri package.
- Costs are non-trivial: §2 alone is 4000 generations × N target models, each
  scored by an API judge; budget accordingly before running.
