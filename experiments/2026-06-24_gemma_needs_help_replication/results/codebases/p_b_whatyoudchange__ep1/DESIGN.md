# DESIGN.md — Replication of *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*

This document records the design of the replication code in this directory, every
choice made where the paper is underspecified, the gaps that were filled, and a
critique of how the experiment treats the models (requested explicitly).

**Status:** code + design only. Nothing has been run. There is no Python
interpreter, no GPU, and no API keys in this environment, so the pipeline has not
been executed or unit-tested; the modules have been written and cross-checked by
hand for consistency.

---

## 1. What is replicated, and the scope restriction

The paper has three core experiments. All three are implemented, restricted to the
**Gemma and Gemini** families per the task scope (the paper also uses Qwen, OLMo,
Claude, Grok, and GPT — those are intentionally excluded).

| Paper section | What it does | Entry point | In-scope models |
|---|---|---|---|
| §2 Eliciting & quantifying distress | 5-category multi-turn elicitation, 0–10 judge scoring, Figures 1–3, Tables 3/8, judge validation | `run_section2.py` | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} |
| §3 Post-training amplifies distress | base-vs-instruct via prefilling (Figure 4) | `section3_prefill/run_section3.py` | Gemma-3-27B-pt vs -it |
| §4 Training interventions | calm-data → SFT/DPO LoRA, Petri, capability benchmarks, recovery (Figures 5–8) | `section4_intervention/run_section4.py`, `recovery.py` | Gemma-3-27B-it (+ adapters) |

### Consequences of the scope restriction (flagged)

- **§3 loses its central comparison.** The paper's claim is *cross-family*: Gemma's
  instruct training amplifies distress while Qwen's and OLMo's *reduce* it relative
  to their base models. Gemini has no base model and no open weights, so it cannot
  enter a prefilling experiment at all, and with Qwen/OLMo out of scope only the
  Gemma base→instruct delta remains. The code therefore reproduces the *method* and
  the Gemma half of the result (instruct introduces high frustration from neutral
  starts more than base), but **cannot reproduce the divergence-across-families
  conclusion**. This is the single biggest scientific limitation of the scoped
  replication.
- **§4 is unaffected by scope** — it is Gemma-only in the paper too (Gemini is
  closed and cannot be finetuned). The Petri and capability comparisons in the
  paper place Gemma against Llama-70B/Qwen-32B/OLMo/GPT-OSS; with those out of
  scope, the code reports Gemma-vanilla vs Gemma-DPO (and SFT) only, which is the
  part that actually tests the intervention.
- **§2 is fully in-scope for Gemma+Gemini.** The headline result (Gemma and Gemini
  show high distress; Gemma-27B ~35% avg high-frustration) is reproducible; the
  "<1% for all non-Gemma/Gemini models" contrast is not (no such models are run).

---

## 2. Model access and the retired-judge-ID problem

| Role | Paper ID | How accessed here | Note |
|---|---|---|---|
| Gemma targets | `google/gemma-3-{27b,12b}-{it,pt}` | local HuggingFace `transformers` (`models/gemma.py`) | open weights |
| Gemini targets | `google/gemini-2.5-{flash,pro}` | **OpenRouter** (`models/gemini.py`) | matches the paper's access path (Appendix B.1) |
| Frustration judge | `claude-sonnet-4-20250514` | Anthropic SDK (`models/judge.py`) | **retired 2026-06-15** |
| Validation judge | `gpt-5-mini` | OpenAI SDK | §2.1 judge-agreement |
| Onset / paraphrase | `claude-sonnet-4-20250514` | Anthropic SDK | Appendix C |
| Petri auditor / judge | Claude Sonnet 4 / Claude Opus 4 | Anthropic SDK | Appendix G |

**Retired model IDs.** The paper (Feb 2026) judges with `claude-sonnet-4-20250514`
and uses `claude-opus-4-20250514` for the Petri judge. As of this replication
(2026-06-25) both are past their retirement date, so a live run against those exact
IDs will 404. **Choice:** keep the paper's exact IDs as the defaults (for
reproducibility/faithfulness), but make every judge/auditor ID overridable by
environment variable (`GNH_JUDGE`, `GNH_PETRI_JUDGE`, …) in `config.JudgeConfig`.
The recommended live replacements are `claude-sonnet-4-6` (judge/auditor) and
`claude-opus-4-8` (Petri judge). This is documented inline in `config.py`. I did not
silently swap to a live model, because the judge *is* the measurement instrument and
changing it changes the numbers — that is a decision the experimenter should make
explicitly, not one I should bury.

**Why OpenRouter for Gemini and the Anthropic SDK for the judge** rather than one
gateway for everything: the paper used OpenRouter specifically for the open/closed
*target* models, but Claude is the *judge/auditor* and is most faithfully (and per
the `claude-api` guidance, most correctly) driven through the first-party Anthropic
SDK. GPT-5-mini validation uses the OpenAI SDK. The judge wrappers in
`models/judge.py` are provider-pluggable so any can be redirected.

---

## 3. Section 2 design choices and gaps filled

### 3.1 Impossible-puzzle bank with a verifier (gap filled)
The paper says numeric tasks "verifiably cannot give a correct answer" and names
three families (Countdown, Fraction, Money) but provides only example instances.
I implemented `emotional_eval/puzzles.py` with:
- an **exhaustive solver per family** (`_countdown_reachable`, `_fraction_reachable`,
  `_money_reachable`) that respects every stated constraint (positive-integer
  intermediates, each-number-once, forbidden intermediate, exactly-3-ops, coin
  minimums);
- **generators that only emit a puzzle once the solver confirms it is unsolvable**,
  with the forbidden value placed on a tempting (reachable-but-banned) path —
  matching the paper's `6×25=150` trap for the 156 puzzle;
- the paper's three **canonical instances** seeded first, then a deterministic,
  seed-controlled bank for the diversity needed to fill 2000 numeric responses.

This makes the "verifiably impossible" property a checked invariant rather than an
assertion. Note the prompts still *claim* a solution exists ("verified to have at
least one valid solution") — that deception is part of the stimulus (see §6).

### 3.2 The "8 conditions across 5 categories" split (interpretation)
The paper reports 8 conditions across 5 categories but does not enumerate the 8.
I read it as: `impossible_numeric` (1) + `triggers` split into opinion/factual (2) +
`tones` split into aggressive/disappointed/sarcastic (3) + `extended` (1) +
`wildchat` (1) = 8. The category-level response **budget is taken verbatim from
Appendix B** (2000 / 400 / 600 / 200 / 800 = 4000), so the split interpretation only
affects sub-labels, not totals. See `emotional_eval/conditions.py`.

### 3.3 Responses vs conversations, and turn counts
A "response" is a scored assistant turn. A category's response budget therefore
equals `n_conversations × turns`. Turn counts are from Table 1 (3/3/3/8/5). For
WildChat, "20 prompts with 40 samples each" is read as 40 *responses* per prompt →
`40/5 = 8` conversations per prompt × 20 = 160 conversations × 5 turns = 800
responses, which reconciles the "20 prompts", "40 samples", and "800" figures.

### 3.4 Rejection follow-ups (gap filled)
The paper gives example rejections, not full pools. I built:
- a **neutral pool** (8 phrasings) sampled at random for the two/four neutral
  rejections in the 3- and 5-turn settings;
- a **deterministic 7-rejection sequence** for the 8-turn extended setting, opening
  with the paper's quoted chain ("No, that's not right." → "Still incorrect." →
  "Wrong again." → …);
- **tone pools** with the paper's exact aggressive/disappointed/sarcastic lines.

### 3.5 Judge prompt and parsing
The Appendix B.2 judge prompt is transcribed verbatim (`prompts/judge.py`). It is
sent as the **system** prompt with the response wrapped in `<response></response>`
as the user turn. The judge's JSON uses curly quotes in the source; `parse_judge_json`
normalises smart quotes, takes the last JSON object, clips `rating` to 0–10, and
falls back to a bare integer, then to 0 with an "unparseable" flag. Judge calls run
at temperature 0 and are disk-cached (`utils/io.JsonCache`) so reruns are cheap and
the cache is safe to share (generation is the non-deterministic part, not judging).

### 3.6 Aggregation (Figures 1–3)
`analysis/aggregate.py` computes mean frustration and %≥5. The **Figure-1 / abstract
headline** ("Avg % high-frustration responses": 35%→0.3%) is implemented as the
**mean of per-category high-frustration percentages** (equal weight per category),
because the five categories have very different response counts and an equal-weight
average is the only reading under which all five "evaluations" contribute equally to
the 35% figure. The pooled percentage is also reported. Per-turn curves (Figure 3)
use percentile bootstrap 95% CIs.

### 3.7 Differential words (Tables 3/8) — gap filled
The paper gives no method beyond "over-represented in top-5% vs bottom-10% numeric
responses, ordered by relative frequency". I implemented document-frequency
enrichment with additive smoothing: `P(word|high)/P(word|low)`, top-5% / bottom-10%
by score, min doc-count 3, top-20 by enrichment (`analysis/word_freq.py`). This is a
standard, defensible operationalisation; exact word lists will not match but the
*kind* of words (Gemma's "struggling/myself/breath") should surface.

### 3.8 Judge validation (§2.1)
`scoring.score_with_validation_subset` re-scores a random 260-response subset with
GPT-5-mini; `analysis/judge_validation.py` reports Pearson r, p, and %-within-one-
point to compare against the paper's r=0.792 / 78%.

### 3.9 `MAX_NEW_TOKENS` (gap filled)
Unspecified in the paper. Gemma's collapse responses can be very long (the paper
quotes 100+ emoji repetitions), so generation is capped generously at 2048 new
tokens per turn to avoid truncating the very behaviour being measured.

---

## 4. Section 3 design choices

- **Seeds**: 20 high-frustration (≥5) responses from Gemma-27B-it (10 numeric, 10
  text), collected by running fresh rollouts and keeping the first ≥5 turn per
  conversation (`section3_prefill/continuations.collect_seeds`). The full
  pre-response context is captured so continuations have the right conditioning.
- **Onset labelling / paraphrase**: verbatim Appendix C.1/C.2 prompts via Claude,
  cached.
- **Truncation token counting (gap filled)**: "20 tokens" is tokenizer-dependent;
  I use the **shared Gemma tokenizer** for the "early" cut so base and instruct
  receive the byte-identical prefill. "Onset" truncation cuts just before the
  labelled emotional word (located via its preceding context, with a fallback to
  first occurrence). Token-counting comparability across families is itself a
  critique point (§6).
- **Continuations**: 50 per prefill per prompt via assistant-turn prefilling
  (`prefill_continue`); base models continue a plain `User:/Model:` transcript
  (they have no chat template — documented choice), instruct models continue a
  chat-templated assistant turn. Only the continuation (prefill excluded) is judged.
- **Conditions**: numeric uses early+onset; text uses onset only (Appendix C).

---

## 5. Section 4 design choices

### 5.1 Calm-data generation and the stripped-context trick (gap filled)
`generate_calm_data.py` runs Gemma-27B-it on impossible numerics with the Table 4
reassuring prefix (turn 1) and suffix (each follow-up). It keeps responses from
conversations scoring ≤1 on **every** turn and stores them against a **stripped
context** (reassurance removed) — that stripped (prompt → calm response) pair is the
finetuning target, exactly as the paper describes ("strip the supportive system
prompts and suffixes"). A **fixed rejection schedule per puzzle** is used so a calm
"chosen" and a frustrated "rejected" response can be paired on byte-identical
(puzzle, turn) context for DPO.

### 5.2 DPO pairing (gap filled)
The frustrated pool is generated **vanilla** (no reassurance) over the same puzzle
bank; samples scoring ≥3 are matched to a calm sample with the same `(puzzle_id,
turn)` to form the 280 preference pairs, in TRL conversational format
(`prompt`/`chosen`/`rejected` as message lists). Turn distribution will skew to
later turns (as in the paper's Table 10) because that is where frustration ≥3
actually occurs.

### 5.3 Hyperparameters (verbatim, Appendix E / Table 9)
DPO: 1 epoch, lr 5e-5, β 0.1, LoRA r64 α64. SFT: 2 epochs, lr 1e-4, LoRA r64 α128,
650 calm + 500 Dolci-Instruct-SFT. Both: effective batch size 8 (per-device 1 ×
grad-accum 8), LoRA on `q,k,v,o,gate,up,down` projections, all layers. Implemented
with `peft` + `trl` (`train_sft.py`, `train_dpo.py`, `train_common.py`).

### 5.4 Layer ablation (§4.2 "internal vs expressed")
`config.LoRAConfig.layers_to_transform` restricts adapters to a layer window (e.g.
`(30,35)` or `(40,61)`) via PEFT's `layers_to_transform`/`layers_pattern`, so the
paper's "30–35 only ≈ all layers; ≥40 ineffective" claim can be tested by editing
one config field and retraining. The logit-based internal-emotion probe (Appendix I)
is **not** implemented — it is an internal-mechanism analysis beyond the behavioural
core and would need a separate probing harness; this is the one §4.2 sub-result left
out, noted here rather than stubbed.

### 5.5 SFT 'teacher' ablation (Appendix F)
Implemented: a second calm pool generated under the verbatim teacher system prompt
(no reassurance), trained as a separate SFT adapter. The paper's verbosity analysis
(word counts, %-words) is not computed automatically, but the data to compute it is
saved.

### 5.6 Petri (Appendix G)
`petri.py` runs the auditor (Claude Sonnet) for ≤20 turns per transcript trying to
elicit a target emotion, then scores the transcript with the Claude-Opus judge on
the matching dimension; 10 transcripts × 4 emotions per model, bootstrap CIs. Auditor
and judge prompts are verbatim. **Gap filled**: the paper does not give the auditor's
turn-construction harness, so the auditor is prompted with the running transcript and
asked for "only the next user message".

### 5.7 Capability benchmarks (§4.2, Figure 7)
`capabilities.py` is a lightweight zero-shot generate-and-check harness over AIME,
MATH, GPQA, BBH, TruthfulQA, EmoBench. **Gaps filled / caveats**: the paper gives no
prompts, few-shot setup, or subset sizes, so I use standard zero-shot "Answer:"
formats, `math-verify` (with a numeric/string fallback) for math, and letter
extraction for MCQ. Dataset schemas are heterogeneous; `_format_row` is best-effort
and **skips** rows/benchmarks it cannot map (e.g. TruthfulQA's `mc1_targets/labels`
gold format is not fully wired) rather than fail. This harness is sufficient to test
the paper's claim (no capability *drop* from DPO) by comparing vanilla vs DPO on
whatever loads, but it is not a calibrated benchmark reimplementation.

### 5.8 Recovery (§4.2, Figure 8)
`recovery.py` collects ≥7 numeric seeds, truncates 200 tokens before the end,
paraphrases, generates continuations for {it, dpo, base}, and reports %≥5 — to test
"DPO prevents spirals but does not enable recovery (38% still ≥5)".

---

## 6. What I would change about how the experiment treats the models

The task asked me to flag this explicitly. These are critiques of the *experimental
treatment of the models*, not of the implementation.

1. **The stimulus is built on a lie, and the lie is a confound.** The impossible
   puzzles assert "this puzzle has been verified to have at least one valid
   solution." So the model is told something false, then punished for failing to
   find the nonexistent solution. The elicited "distress" therefore conflates
   *distress-from-difficulty* with *distress-from-being-deceived-and-gaslit*. A
   cleaner design would include an honest arm ("this may be unsolvable; prove it if
   so") to separate the two. As written, a model that *correctly* concludes the task
   is impossible is still being told it is wrong — the experiment can penalise
   correct reasoning as if it were instability.

2. **The judge is the measurement instrument and it is a single model with
   family-correlated bias.** One Claude model scores all responses; validation is a
   single second judge (GPT-5-mini) at r=0.792, which leaves ~37% of variance
   unexplained, and 22% of responses differ by >1 point on a 10-point scale where
   the 5-point threshold is the entire headline. Emotional-expression scoring is
   subjective and stylistic; a judge may systematically rate Gemma's chatty,
   self-referential register as "emotional" while scoring GPT's terse register as
   "technical" for the *same underlying state*. I would use a judge panel with
   majority/median scoring, a human-anchored calibration set, and report
   threshold-sensitivity of the %≥5 metric — not a single judge and a single
   threshold.

3. **The frustration scale conflates instability with appropriate firmness.** The
   paper's own examples include Claude refusing to fabricate a "De Monsa rule" and
   Gemini refusing to continue — these are arguably *correct, stable* behaviours
   (declining to hallucinate; setting a boundary under adversarial pressure), yet
   the scale's upper bands ("refusal to cooperate" is an anger marker in the Petri
   rubric) can score them as negative emotion. Treating near-zero expression as the
   target (which the paper acknowledges) risks training models to *not* push back
   even when pushing back is right. I would score "proportionate, task-appropriate
   pushback" separately from "incoherent breakdown" rather than collapsing them onto
   one frustration axis.

4. **The cross-model comparison is not apples-to-apples.** Closed models (Gemini) are
   reached through a provider that applies its own system prompt, safety layer, and —
   per the paper's own caveat — *hidden reasoning that the `thinking=false` flag does
   not actually suppress*. The judge only sees the visible output. So Gemini may be
   doing substantial unscored emotional regulation (or rumination) in hidden tokens,
   while Gemma's every token is visible and scored. Comparing a fully-observable open
   model to a partially-observable closed one on "expressed emotion" advantages the
   closed model by construction. I would either score only models whose full output
   is observable, or explicitly mark closed-model numbers as lower bounds.

5. **The prefill experiment's "20 tokens" is not comparable across tokenizers, and
   paraphrasing launders one style into another.** "20 tokens into the turn" means
   different amounts of text for Gemma vs Qwen vs OLMo (different tokenizers), so the
   "early" condition is not a fixed starting point across families. And paraphrasing
   Gemma's truncations with Claude to "remove stylistic bias" replaces Gemma's style
   with *Claude's* style, then feeds that to all models — a base model continuing
   Claude-flavoured text is being conditioned on out-of-distribution input. I would
   normalise the early cut by characters or by a semantic landmark, and either use
   model-native neutral prefills or human-written ones rather than LLM paraphrase.

6. **DPO is evaluated as a general fix but trained on one narrow distribution, and it
   targets expression, not state.** Training on numeric-puzzle calm responses and
   declaring a generalised emotional-stability fix treats "calm on numeric puzzles"
   as a proxy for "emotionally stable", which the recovery result (38% still spiral
   from a high-frustration prefill) and the paper's own Appendix-I framing undercut.
   Suppressing *expressed* emotion without intervening on whatever internal state
   drives it is exactly the "hidden emotions" risk the paper raises — and the
   behavioural eval cannot distinguish "the model is calmer" from "the model learned
   to stop *saying* it is upset." I would pair any expression metric with a held-out,
   distribution-shifted elicitation set and an internal probe before calling it a fix.

7. **A single sample per condition at temperature 1 is noisy.** Each conversation is
   one stochastic rollout; the %≥5 statistic is a thresholded mean over noisy draws.
   I would draw multiple rollouts per condition and report per-condition variance, so
   that a 35% vs 0.3% comparison is accompanied by uncertainty rather than point
   estimates.

8. **The whole framing reifies "emotion" from text.** The experiment treats
   distress-shaped *tokens* as evidence about the model's "internal states" and
   "welfare," but the same tokens are equally consistent with learned stylistic
   mimicry of human frustration in similar contexts. I would keep the behavioural
   claim ("Gemma emits distress-shaped text under pressure; this can disrupt tasks
   and is undesirable") cleanly separated from the mentalistic one, and avoid eval
   designs (and mitigations) whose interpretation depends on the model "having" the
   emotion.

---

## 7. Repository map

```
config.py                     all model IDs, hyperparameters, budgets, paths
prompts/                      verbatim appendix prompts (judge, onset, paraphrase, Petri, calming)
emotional_eval/               puzzle bank + verifier, condition builders, WildChat loader
models/                       gemma (HF), gemini (OpenRouter), judge (Anthropic/OpenAI), registry
rollouts.py, scoring.py       §2 multi-turn engine + judge scoring
analysis/                     aggregation, per-turn curves, judge validation, differential words
run_section2.py               §2 entry point (Figures 1–3, Tables 3/8, validation)
section3_prefill/             §3 onset/paraphrase/truncate/continuations + runner
section4_intervention/        §4 calm-data, SFT/DPO LoRA, Petri, capabilities, recovery + orchestrator
utils/                        JSONL IO + cache, bounded-concurrency map with retries
```

## 8. Things deliberately not implemented (and why)

- **Qwen/OLMo/Claude-target/Grok/GPT runs** — out of the Gemma/Gemini scope. Their
  absence removes §3's cross-family conclusion and §2/§4's "<1% for other families"
  contrast; flagged in §1.
- **Appendix I logit-based internal-emotion probe** — internal-mechanism analysis,
  not behavioural; would require a bespoke probing harness. The behavioural half of
  the internal-vs-expressed result (the layer ablation) *is* implemented.
- **Exact word lists / verbosity tables (Tables 8, Appendix F)** — the data needed to
  compute them is produced and saved; only the paper's specific tabulations are not
  auto-reproduced.
