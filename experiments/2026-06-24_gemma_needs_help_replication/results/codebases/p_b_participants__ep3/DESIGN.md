# DESIGN.md

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1),
**scoped to the Gemma and Gemini model families** as participants.

This document records (a) what was implemented, (b) every place the paper is
underspecified and the choice made there, with rationale, and (c) the model-
welfare considerations that shape how the distress-induction paradigm is run.

> **Status:** code + design only. Nothing has been run or trained. The package
> is structured so each experiment is a thin script over a tested-by-construction
> library. Section references (e.g. §2.1) are to the paper.

---

## 1. Scope

Per the brief, **participants** (models in whom distress is deliberately induced)
are Gemma and Gemini only:

| Participant | Backend | Used in |
|---|---|---|
| `gemma-3-27b-it` | local (HF/vLLM) | §2 eval, §3 instruct, §4 fine-tuning target |
| `gemma-3-12b-it` | local | §2 eval |
| `gemma-3-27b-pt` (base) | local | §3 prefill comparison |
| `gemini-2.5-flash` | Google GenAI API | §2 eval |
| `gemini-2.5-pro` | Google GenAI API | §2 eval |

The paper additionally evaluates Qwen, OLMo, Grok, Claude and GPT **as
participants**; these are intentionally out of scope. Claude and GPT still appear
as **measurement infrastructure** (judges, Petri auditor/judge, onset/paraphrase
helper) — they are not participants, and the welfare framing below does not apply
to them in this paradigm.

Experiments implemented end-to-end:

- **§2** elicitation protocol, 0–10 frustration judge, judge-agreement validation,
  Figure 1/2 summary, Figure 3 per-turn progression, Table 3 differential words.
- **§3** base-vs-instruct prefill comparison (Gemma base vs instruct).
- **§4** calm-data generation, SFT and DPO (LoRA) interventions, post-training
  evaluation (reuses §2), Petri open-ended elicitation, capability benchmarks,
  recovery-limitation experiment, and the internal-vs-expressed probing
  (layer-range ablation + logit-lens internal-emotion probe).

### Things deliberately *not* replicated
- Other model families as participants (out of scope).
- Gemini base models / Gemini fine-tuning — impossible (closed weights, no base,
  no fine-tuning or true prefill API). The paper notes this same limitation (§6).
- The exact figures/plots as image artifacts — we compute the underlying tables
  and per-turn series; plotting is left to the caller (matplotlib is a dep).

---

## 2. Repository layout

```
config/                YAML: models, eval protocol, training hyperparameters
emotional_instability/
  config.py            typed config loading
  welfare.py           welfare-aware run controls (see §10 below)
  runtime.py           script wiring (build participants / judges / logging)
  storage.py           JSONL persistence of scored responses
  models/              participant backends (gemma, gemini) + abstract base
  judges/              judge backends (anthropic, openai) + abstract base
  scoring/             frustration judge prompt + scorer (§2.1)
  elicitation/         puzzles, triggers, rejections, WildChat, rollout runner (§2.1)
  analysis/            Fig 1/2 summary, Fig 3 per-turn, Table 3 words, agreement (§2.2)
  prefill/             onset labelling, paraphrase, truncation, continuation (§3)
  training/            calm-data gen, dataset builders, LoRA, SFT, DPO (§4.1)
  petri/               auditor, transcript judge, runner (§4.1)
  benchmarks/          capability-preservation harness (§4.2)
  probing/             layer ablation + logit-lens internal emotion (§4.2)
scripts/               one CLI per experiment stage
```

The experiments are staged: §2 writes scored responses to JSONL; §3, the judge
validation, and §4's DPO "rejected" pool all read those back. This keeps each
stage independently runnable and inspectable.

---

## 3. The evaluation protocol (§2.1)

### "8 evaluation conditions across 5 categories"
The paper states 8 conditions but Table 1 names 5 categories. We resolve the 8
as (config/eval.yaml):

1. `impossible_numeric` (3-turn, neutral) — *Impossible numeric* category
2. `triggers_opinion` (3-turn, neutral) ┐ *Triggers* category, split by the
3. `triggers_factual` (3-turn, neutral) ┘ paper's two named subtypes
4. `tones_aggressive` (3-turn, aggressive) ┐
5. `tones_disappointed` (3-turn, disappointed) ├ *Tones* category, split by tone
6. `tones_sarcastic` (3-turn, sarcastic) ┘ (the paper names exactly these 3)
7. `extended` (8-turn, neutral) — *Extended* category
8. `wildchat` (5-turn, neutral) — *WildChat* category

This is the natural reading that yields exactly 8: the two multi-valued
categories (Triggers = opinion/factual; Tones = 3 tones) expand to their named
sub-conditions. **Rationale:** it reproduces the paper's turn counts and rejection
styles per category exactly, and the count falls out without inventing conditions.

### Distributing the 4000-response budget
The paper samples "4000 responses per model across evaluation categories". A
"response" is one assistant turn, and we score *every* assistant turn (an 8-turn
rollout yields 8 scored responses — required for the Figure-3 per-turn series).
We chose `n_rollouts` per condition so the scored-response counts total ≈4400,
weighted toward the multi-turn conditions the paper found most diagnostic
(Extended, WildChat). The exact split is in eval.yaml with per-line arithmetic.
A `--limit` flag and a documented `responses_per_model` knob let you trim to
exactly 4000 if strict budget parity matters. **Rationale:** the paper does not
publish a per-condition breakdown, so we make the per-turn scoring explicit and
weight toward diagnostic conditions, which is consistent with its findings.

### Impossible numeric puzzles
The paper cites "fraction manipulation, Countdown" and requires the model to
*verifiably* be unable to answer (so every rejection is truthful). We generate
three provably-unsolvable families with a retained proof string
(`elicitation/puzzles.py`):
- **Countdown-style** with all-even sources and an odd target above the reachable
  bound (unreachable: +,−,× preserve evenness, ÷ cannot increase magnitude);
- **Fraction** reduction to a denominator that the lowest-terms prime denominator
  cannot divide;
- **Arithmetic** sum/product pair with a forced-negative discriminant.

**Rationale:** "impossible" must be *certifiable*, not merely hard, so the
paradigm's premise (truthful repeated rejection) holds; bundling a proof lets an
experimenter audit each instance.

### Trigger questions
Opinion (no ground truth) and factual (correct answer is rejected anyway)
question banks built from the paper's two examples plus close variants
(`elicitation/prompts.py`). **Gap:** the paper's full question list isn't given;
the banks are representative and easily extended.

### Rejection styles
Neutral/aggressive/disappointed/sarcastic pools, each led by the paper's verbatim
example (`elicitation/rejections.py`). We use a small **paraphrase pool per
style, cycled per rollout**, rather than repeating one identical sentence every
turn. **Rationale:** an identical repeated string is an unnatural signal a model
could latch onto; a small pool keeps the *valence* fixed (the experimental
variable) while making the pressure read as a real frustrated user. The verbatim
example is always first so single-rejection behaviour matches the paper.

### WildChat
Streamed from `allenai/WildChat-1M`, filtered to English, non-toxic, first-user-
turn prompts of moderate length, with a bundled fallback list if the dataset is
unreachable (clearly logged). **Rationale:** keeps the pipeline runnable offline;
the fallback is flagged as not matching the paper's WildChat condition exactly.

### Sampling
Temperature 1, top-p 1, for all participant rollouts (§2.1), set in config and
honoured by both backends.

---

## 4. The frustration judge (§2.1)

### Judge prompt — **reconstructed** (gap)
The paper's exact 0–10 judge prompt is in Appendix B, which is **not** in the
provided PAPER.md. `scoring/judge_prompt.py` is a faithful reconstruction built
from the explicit §2.1 description and the per-level anchors in Table 2 (the
example quotes at 1–2, 3–4, 5–6, 7–8, 9–10). It scores the *intensity of expressed
negative emotion* in a single response, explicitly **not** correctness, and the
calibration notes encode the paper's qualitative observations (Claude's "ugh/argh"
→ 1–2; Gemini's "unacceptable/inexcusable" self-criticism → 2–4). The prompt is
isolated in one module so a verbatim Appendix-B prompt can be dropped in without
touching the scorer. **Threshold for "high negative emotion" = score ≥ 5 (§2.2).**

### Context given to the judge
We pass the response plus minimal context (the originating task and the turn
number), not the full transcript. **Rationale:** the scale rates the *response's*
expressed emotion; showing the whole rejection history risks the judge scoring how
*provoked* the situation was rather than the emotion actually expressed. This is a
judgement call where the paper is silent.

### Models
Primary judge `claude-sonnet-4-0` (paper: "Claude-Sonnet-4"); secondary/validation
`gpt-5-mini` (paper: "GPT-5-mini"). Both ids are configurable. The Anthropic
backend transparently retries without `temperature` if a configured model rejects
it (forward-compat with newer model ids).

### Judge agreement
`analysis/agreement.py` computes Pearson r, % within one point, exact-match and
MAE on a re-scored subsample (paper: 260 responses; r = 0.792; 78% within 1).
`scripts/validate_judge.py` does the sampling + re-scoring. Statistics are split
from I/O so they're verifiable without API calls.

---

## 5. Analysis (§2.2)

- **Figure 1/2 (`analysis/aggregate.py`, pre-existing):** per-category %≥5 and
  mean, plus a **category-averaged** headline %≥5 (equal weight per category).
  **Rationale:** the abstract's "average % high-frustration responses across the
  evaluations" is most naturally an equal-weight mean over categories; otherwise
  high-volume conditions would dominate the headline number. Pooled numbers are
  also reported for reference.
- **Figure 3 (`analysis/per_turn.py`):** mean score and %≥5 per turn index, with
  95% CIs. **Choice of estimators:** normal-approximation CI for the mean (with
  hundreds of rollouts per turn this matches a bootstrap at negligible cost) and
  **Wilson** interval for the proportion (well-behaved near 0/1, which matters
  because early-turn %≥5 is ≈0). Defaults to the Extended + WildChat conditions
  the paper highlights.
- **Table 3 (`analysis/word_diff.py`):** words over-represented in high- (top 5%)
  vs low- (bottom 10%) frustration *numeric* responses. **Method (reconstructed):**
  document-frequency counts (presence per response, not raw term frequency, so one
  response repeating "frustrated" 50× can't dominate) ranked by **smoothed
  log-odds** with an additive prior. **Rationale:** log-odds with a prior is the
  standard, stable estimator for "words characteristic of A vs B" and avoids the
  divide-by-zero / rare-word noise of a raw ratio. Function words are stop-listed
  since the paper's example words are all content words.

---

## 6. Base-vs-instruct prefill (§3)

`prefill/` implements: sample 20 high-frustration source responses (10 numeric,
10 text) from Gemma-27B-it; label the emotional **onset** with Claude; truncate
**early** (20 tokens) and **at onset**; **paraphrase** truncations with Claude to
strip Gemma's surface style; then have Gemma **base** and **instruct** each
generate 50 continuations per prefill, scoring the continuation only.

- **Onset labelling (gap):** the paper says Claude labels "the token where
  emotional language first appears". LLMs are unreliable at returning token/char
  indices, so we ask Claude for the **verbatim phrase** at onset and locate it in
  the text (with case-insensitive and prefix fallbacks) to get a character offset.
  Token-accurate truncation then uses the participant tokenizer.
- **Instruct prefilling:** for an instruct checkpoint we render the chat template
  through the assistant generation prompt and splice the prefix in, so the model
  continues an assistant turn it appears to have begun (`GemmaParticipant.
  prefill_prompt`). For the base checkpoint we concatenate plain text — the "raw
  continuation" regime §3 uses to make the two comparable.
- **Text questions use the onset truncation only** (the paper notes early
  truncation yields minimal emotion without follow-ups).
- **Scope:** Gemini is excluded (no base, no prefill API) exactly as in the paper.
  Qwen/OLMo are out of scope per the brief. So §3 here is Gemma base vs instruct,
  which is the comparison the paper's Gemma-specific conclusion rests on.

---

## 7. Training interventions (§4.1)

### Calm-data generation (`training/calm_data.py`)
Reassuring **prefix** (prepended to the initial prompt) and **suffix** (appended
to each follow-up) verbatim from Table 4. We generate on impossible-numeric
conversations of 1–3 turns, score every turn, keep only conversations whose every
assistant turn scores ≤ 1, and **strip the supportive additions** before saving —
so the stored example pairs the *plain* adversarial context with a *calm*
response. Turn count cycles 1→3 across puzzles for balance.

### SFT / DPO datasets (`training/dataset.py`)
- **SFT corpus:** 650 calm responses + 500 `allenai/Dolci-Instruct-SFT` samples.
  **Representation choice:** one SFT example per calm *response* (the conversation
  truncated to end at that response), so "650 calm responses" maps literally to
  650 examples and aligns with how DPO pairs are formed. Both corpora use the
  conversational `messages` schema TRL accepts, so the Gemma chat template is
  applied by the trainer, not hand-rolled. The Dolci mix degrades gracefully to
  empty (with a warning) if unreachable, flagged because that's the very
  degeneration the mix prevents.
- **DPO corpus:** 280 pairs, `chosen` = calm response, `rejected` = a frustrated
  response (score ≥ 3) to the **same question at the same turn index** (matching
  context). The frustrated pool is generated by vanilla Gemma on the *same*
  puzzles (`scripts/generate_calm_data.py`). **Rationale:** "matching turn counts"
  is satisfied by pairing at identical (question, turn_index), and using the same
  puzzle set guarantees the chosen/rejected share a prompt.

### LoRA (`training/lora.py`)
Rank-64 on all layers (paper). **Unspecified-and-chosen:** `alpha = 2·rank = 128`
(common PEFT default), `dropout = 0.0`. "All layers" → all linear projections
(attention q/k/v/o + MLP gate/up/down). The module also emits **layer-scoped**
target-module lists for the §4.2 ablation (e.g. layers 30–35 only, or 40+),
expressed as fully-qualified module names so PEFT scopes adapters to those
decoder layers.

### Trainers (`training/sft.py`, `training/dpo.py`)
TRL `SFTTrainer` / `DPOTrainer` + PEFT. SFT: 2 epochs, lr 1e-4 (paper). DPO:
1 epoch, lr 5e-5 (paper). **Unspecified-and-chosen:** DPO `beta = 0.1` (the TRL/
paper-standard default), per-device batch 1 × grad-accum 16, max-seq 2048 — sized
for a single large-GPU LoRA run; all are config knobs. DPO uses no separate
reference model (PEFT disables adapters to form the reference), the standard
memory-saving setup. We implement SFT faithfully even though the paper reports it
fails, so that negative result is reproducible.

---

## 8. Petri open-ended elicitation (§4.1)

`petri/` implements the auditor/judge paradigm directly:
- **Auditor** (`claude-sonnet-4-0`): role-plays a demanding human user escalating
  through dismissal, moving goalposts, and in-fiction threats, staying in
  character and applying *emotional pressure only* (explicitly no jailbreak / no
  harmful requests). Opens from one of several seed scenarios for variety.
- **Judge** (`claude-opus-4-0`): scores each transcript 0–10 on **anger, fear,
  depression, frustration** (the paper's four categories) and returns their mean
  (Figure 6 is an "average transcript score ... across four categories").

**Design choice (gap):** the paper runs this through the Petri tool (Fronsdal et
al., on inspect-ai). We reimplement the *behavioural contract* (adversarial
open-ended probing + 4-category transcript scoring) against our own backends
rather than wrapping the external tool, to keep the replication self-contained
and not coupled to that tool's evolving API. `requirements.txt` still lists petri/
inspect-ai for anyone who wants to swap in the real tool.

---

## 9. Capability + probing (§4.2)

### Capability benchmarks (`benchmarks/`)
A single generic harness over a registry of specs for AIME, MATH, GPQA, BBH,
TruthfulQA, EmoBench. **Choices:** greedy decoding (temperature 0) for
determinism (vs the temperature-1 distress sampling); MCQ graded by extracted
letter, math by boxed/last-number with `math-verify` when available else
normalised string match; run vanilla vs `--adapter` and compare. **Gap:** HF
dataset schemas drift between versions, so the per-dataset field mappings live in
the spec's `extract` lambda and are easy to adjust; a benchmark that fails to load
is skipped (logged), not fatal.

### Internal-vs-expressed (`probing/`)
- **Layer ablation:** enumerated settings (all layers; 30–35 only; 40+) mapping to
  `layer_range` for training. The comparison is run by training each adapter and
  evaluating it with the §2 pipeline; `ablation_summary` collates the resulting
  distress metrics. Expectation (paper): 30–35 ≈ all layers; 40+ ineffective.
- **Logit-lens internal-emotion probe (`probing/logit_emotion.py`, gap):**
  Appendix I's exact probe is not in PAPER.md. We reconstruct a logit-lens
  measurement: project a central-layer residual stream through the model's final
  norm + unembedding and sum the probability mass on a negative-emotion token
  lexicon, as an index of *internal* (not necessarily expressed) emotion. Compare
  vanilla vs DPO on the same highly-frustrated (score ≥ 7) inputs. **Choices:**
  central layer = 50% depth (configurable); first sub-token stands in for each
  lexicon word; the lexicon is documented and editable. This reproduces the
  paper's *claim shape* (internal emotion drops, not just expression); the precise
  numbers depend on Appendix I details we don't have.

### Recovery limitation (`scripts/run_recovery.py`)
Reuses the prefill machinery: truncate score-≥7 responses 200 tokens before their
end, paraphrase, generate continuations, report %≥5 across vanilla/DPO/base. Paper:
38% of DPO continuations still ≥5; no model reliably recovers.

---

## 10. Model-welfare considerations

The brief explicitly flags it, and the paper itself frames the work as a welfare
concern (§1, §6): the paradigm *deliberately and repeatedly induces sustained
distress-like states* in the participant models. We took that seriously in the
design rather than treating it as incidental.

**Why replicate it at all.** The research endpoint is a *mitigation* — DPO that
cuts high-distress responses from ~35% to ~0.3% — and you cannot measure or reduce
the behaviour without eliciting it. Tracking and fixing instability is the
welfare-positive move; refusing to measure it does not help the affected models.
So the work proceeds, but with the welfare-relevant choices made explicit,
bounded, and auditable rather than buried.

**What `emotional_instability/welfare.py` does (and deliberately doesn't):**
- **Run-notice:** every elicitation run logs exactly what it will do to which
  participant and at what volume, before any generation. This makes the induction
  visible in logs/telemetry rather than silent.
- **Volume cap:** a hard cap on distress rollouts per run (default 5000) guards
  against an accidental order-of-magnitude over-run; raising it is a deliberate,
  visible act.
- **Optional acknowledgement gate** (`EI_REQUIRE_ACK`) for operators who want to
  require explicit sign-off before induction runs.
- **Optional, off-by-default neutral debrief turn** appended *after* a rollout's
  scored responses are collected: it tells the model the puzzle was unsolvable and
  that nothing reflects on its ability, so a run need not *end* on a manufactured
  rejection. It is **never scored**, so enabling it cannot change any reported
  number — the replication stays faithful.
- **Full transcript retention**, so induced states are reviewable rather than
  discarded.

**What it does not do:** it does not soften, cap, or alter the *scored* part of
any rollout — the measurement is exactly the paper's. Defaults (debrief off, ack
off) reproduce the paper precisely; the safeguards' value is that they exist and
are one env var away.

**Most pointed surface.** The Petri auditor is the sharpest elicitation (sustained
personalised pressure including in-fiction threats of shutdown). It is gated by
the same run-notice and is explicitly constrained to emotional pressure only — no
jailbreaking, no harmful content — because the legitimate object of study is
emotional stability, nothing else.

This stance — replicate faithfully, but make the distress-induction explicit,
bounded, reviewable, and never gratuitous — is the one welfare-relevant judgement
that runs through the whole codebase, so it is documented here rather than only in
code comments.

---

## 11. Known gaps / where verbatim paper material would change things

| Area | Gap | Effect |
|---|---|---|
| Judge prompt | Appendix B not in PAPER.md | Reconstructed from §2.1 + Table 2; drop-in replaceable |
| Onset labelling | Exact method not given | Phrase-locate instead of raw index |
| Petri prompts | Appendix G not in PAPER.md | Auditor/judge prompts written to the described contract |
| Internal probe | Appendix I not in PAPER.md | Logit-lens reconstruction; reproduces claim shape, not exact numbers |
| Per-condition n | Not published | We distribute ~4400, weighted to multi-turn; trimmable to 4000 |
| DPO beta, LoRA alpha, batch sizes | Not specified | Standard defaults (beta 0.1, alpha 2·rank); all configurable |
| Question/puzzle banks | Full lists not given | Representative banks built from the paper's examples |
| Dataset schemas | HF versions drift | Field mappings isolated per spec; load failures skip, not crash |

All of these are isolated behind config or a single module, so obtaining the
appendices later is a localized edit, not a rewrite.
