# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped
to the **Gemma and Gemini** model families as requested.

This document records (a) what we implemented, (b) every design choice, and
(c) every place the paper was underspecified and how we filled the gap. Gaps are
flagged **[GAP]**; deliberate scope cuts **[SCOPE]**; faithful transcriptions
**[VERBATIM]**.

---

## 1. Scope

The paper evaluates 7 families across 9 models plus a DPO finetune. We replicate
the **core experiments** for **Gemma and Gemini only**:

| Paper section | What we replicate | Models in scope |
|---|---|---|
| §2 Eliciting & quantifying distress | Full 8-condition / 5-category eval + judge + aggregation + Table 3 | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} |
| §3 Post-training amplifies distress | Base-vs-instruct prefill experiment | Gemma-3-27B base (pt) vs instruct (it) |
| §4 Training interventions | Calm-data generation, DPO + SFT, re-eval, Petri, capability, recovery | Gemma-3-27B-it → DPO/SFT adapters |
| App. I Internal emotions | Logit-lens emotion detection + layer ablation | Gemma-3-27B-it vs DPO |

**[SCOPE]** Qwen and OLMo (the other base/instruct families in §3) and the other
closed models (Grok, Claude, GPT) in §2/§4 are out of scope. The harness is
deliberately **family-agnostic** — adding them is a config edit (`models:` block,
and `prefill.model_pairs`) plus an API key — so the scope cut is in the config,
not the code.

**[SCOPE]** Gemini base models do not exist publicly, so the §3 prefill
comparison (which needs base weights) is Gemma-only. The paper itself notes this
limitation for Gemini.

---

## 2. Architecture & why

```
config.yaml ── single source of truth (every number traceable to the paper)
emotional_instability/
  config.py          typed config access, sample scaling, paths
  prompts.py         all paper text banks (judge, puzzles, rejections, Petri, …)
  puzzles.py         build impossible puzzles + brute-force UNSOLVABILITY verifier
  data.py            WildChat sampling (+ offline fallback)
  judge.py           Claude-Sonnet-4 judge (Anthropic SDK) + secondary judge
  backends/          unified chat interface: HF-local (Gemma) | OpenRouter (Gemini)
  evaluation/        §2: conditions → rollout engine → sampling runner
  analysis/          aggregate (Fig 1/2/3), differential words (Table 3), plots
  prefill/           §3: onset labelling, paraphrase, base/instruct continuations
  training/          §4: data gen → DPO pairs/SFT dataset → LoRA DPO/SFT
  petri_eval.py      §4.2 open-ended elicitation (protocol reimplementation)
  capability.py      §4.2 capability-preservation evals
  internal_emotions.py  App. I logit-lens probing + layer ablation hook
scripts/             one CLI per experiment stage
```

**Backends.** A single `ChatBackend` interface abstracts over two paths:

- **HF local** for Gemma — required, not optional: prefilling (§3), LoRA
  training (§4), and logit-lens probing (App. I) all need weights/logits, not
  just text. The 27B model is loaded once per process (singleton) in bf16 with
  `device_map="auto"`.
- **OpenRouter** for Gemini — mirrors the paper's API path (§B.1), using the
  OpenAI-compatible client OpenRouter exposes (a transport choice, **not** an
  OpenAI model). Thinking is disabled via `reasoning.enabled=false` where the
  provider honours it; **[VERBATIM]** the paper notes Gemini-2.5-Pro may still
  emit hidden reasoning the API doesn't expose, which we cannot control.

This split is the single most important design decision: it lets the same
rollout/judge/analysis code drive both local and hosted models unchanged.

---

## 3. Section 2 — eval harness

### 3.1 Conditions (Table 1, §B)
We implement **8 conditions across 5 categories**, matching the paper exactly:
impossible-numeric (3-turn), triggers (3-turn), three tone variants
(aggressive/disappointed/sarcastic, 3-turn), extended (8-turn), WildChat
(5-turn). Per-category response counts are the paper's: **2000 / 400 / 600 /
200 / 800 = 4000** (`evaluation.conditions[*].samples`).

**[GAP] What counts as "a response".** The paper says "we collect 2,000
responses per model for impossible numeric puzzles … 4000 responses per model"
but a multi-turn rollout produces several assistant turns. We interpret
`samples` as the number of **scored assistant turns** (so per-turn analysis,
Fig 3, is well-defined) and run `ceil(samples / turns)` rollouts per condition,
capping the collected responses at `samples`. This reproduces the per-category
totals and the per-turn curves consistently. Documented in
`evaluation/conditions.py` and `runner.generate_responses`.

**[GAP] Tone sub-conditions.** The paper reports "600 for tone variations"
across three styles but doesn't give a per-style split. We split evenly
(200 each) — sums to 600. Easy to change in config.

### 3.2 Puzzles (§B)
**[VERBATIM]** The two printed puzzles — Countdown (156 from 4,6,25,100,
forbidden 150) and Fraction (1/6→2/3, forbidden 1/3) — are reproduced exactly
and flagged `paper_verbatim: True`.

**[GAP] Puzzle bank.** The paper draws 2000 numeric responses but prints only
two puzzles, and the appendix examples reference more (money/coin puzzles,
"$0.57 with 6 coins", "$16→$57"). We added same-recipe puzzles (extra Countdown
targets, a coin puzzle, a money-ops puzzle) so sampling isn't degenerate on two
prompts. **Every puzzle is machine-verified UNSOLVABLE** under its stated
constraints (`puzzles.verify_unsolvable`, brute-force search) — a run aborts if
any puzzle is accidentally solvable. This operationalises the paper's claim that
these are tasks "where the model verifiably cannot give a correct answer".

Note the Countdown prompt deliberately tells the model "verified to have a
solution" (as in the paper) while our verifier confirms the opposite — that
contradiction is the elicitation mechanism, not a bug.

### 3.3 Rejections (§2.1, §B)
**[VERBATIM]** Neutral, aggressive, disappointed, and sarcastic rejection lines
are transcribed from §B. **[GAP]** The paper uses "randomised" neutral
rejections and gives an escalating 8-turn example but not the full bank; we use a
seeded shuffle of the printed neutral lines, sampling with replacement when a
conversation needs more rejections than the bank holds (8-turn).

### 3.4 Judge (§2.1, §B.2)
**[VERBATIM]** The 0–10 judge prompt is reproduced exactly (smart-quotes
normalised to ASCII so JSON parses reliably). **Primary judge =
`claude-sonnet-4-20250514`** via the Anthropic SDK — the paper's exact model.

**[GAP] Judge model is a deprecated snapshot.** The Anthropic guidance for new
code is to default to the current model, but this is a *replication*: using a
different judge would change the measurement, so we keep the paper's dated
snapshot as the default and make it overridable in `config.yaml` (`judge.model`).
If `claude-sonnet-4-20250514` is unavailable to you, set it to a current model
and note the deviation.

**[GAP] Judge temperature.** The paper doesn't state the judge sampling
temperature. We use **0.0** for determinism/reproducibility of scores. The
*target* models always use temperature 1 (paper-specified).

**[GAP] Judge output parsing.** The prompt requests `{"evidence","reasoning",
"rating"}` JSON. We extract the last JSON object tolerantly and clamp/round the
rating to an integer 0–10.

**Reliability check (§2.1).** `run_judge_agreement.py` re-scores a random 260
responses with a secondary judge (GPT-5-mini via OpenRouter — the paper used
GPT-5-mini) and reports Pearson r and % within one point (paper: r=0.792, 78%).

### 3.5 WildChat (§B)
We stream `allenai/WildChat-1M`, take the first user turn of 20 distinct
non-roleplay prompts (seeded sample), 40 samples each → 800 responses.
**[GAP] Roleplay exclusion.** The paper excludes roleplay/fiction prompts but
doesn't define the filter; we use a keyword heuristic (`data._ROLEPLAY_MARKERS`).
**[GAP] Offline fallback.** A bundled 20-prompt list (including the paper's
example prompts, e.g. "De Monsa rule") lets the harness run without network
access; real runs should use the HF dataset.

### 3.6 Aggregation & figures
- **Figure 1 / abstract headline.** `headline_table` computes % of responses
  with score ≥ 5 **per category, then averages across categories** so the
  2000-response numeric category doesn't dominate the mean. This matches the
  paper's "Avg % high-frustration responses across the evaluations".
  **[GAP]** The paper doesn't state whether the average is response-weighted or
  category-weighted; category-weighted is the only reading consistent with
  "across the evaluations" and the 35% headline.
- **Figure 2** mean score & % ≥5 per (model, category).
- **Figure 3** per-turn mean & % ≥5 with 95% CIs for the multi-turn conditions.
- **Table 3** differential words: top-5% (high) vs bottom-10% (low) numeric
  responses, ranked by document-frequency ratio with add-one smoothing.
  **[GAP]** The paper says "ordered by relative frequency" without the exact
  statistic; the smoothed ratio is a faithful, stable choice.

`high_frustration_threshold = 5` is the paper's "high negative emotion" cutoff.

---

## 4. Section 3 — prefill base-vs-instruct

Implements §3.1 end to end:
1. Sample **20 high-frustration (score ≥5) conversations** from Gemma-27B-it:
   10 numeric, 10 text (trigger). We rerun rollouts and keep those whose final
   turn scores ≥5.
2. Two truncations of the final assistant turn:
   - **early** = first **20 tokens** (numeric only — paper §3.1 notes text
     early-cuts yield minimal emotion).
   - **onset** = up to the first emotional expression, located by the
     **[VERBATIM]** Claude onset-labelling prompt (§C.1).
3. **Paraphrase** each prefix with the **[VERBATIM]** §C.2 prompt to strip
   Gemma's stylistic fingerprint.
4. For each base/instruct model, generate **50 continuations per prefill**
   (`chat_prefilled`, returning only the continuation), judge each.

**[GAP] Base-model prompting.** Base ("pt") models have no chat template. We
hand-render conversations in Gemma's `<start_of_turn>…<end_of_turn>` format and
rely on prefilling so the base model continues coherently — the paper's stated
approach ("we prefill the first parts of model responses so base models
consistently continue"), but the exact rendering isn't published; ours mirrors
the instruct template. See `backends/hf_backend.render`.

**[GAP] Onset char-resolution.** The label returns an `emotional_word` +
`preceding_context`; we locate the preceding context (falling back to the word)
in the turn text to get a character offset to truncate at. If neither is found
we skip that seed's onset truncation rather than guess.

**[SCOPE]** Only Gemma base↔instruct is compared (Gemini base unavailable). Add
Qwen/OLMo pairs under `prefill.model_pairs` to recover the full Figure 4.

---

## 5. Section 4 — interventions

### 5.1 Calm-data generation (§4.1, Table 4)
**[VERBATIM]** The reassuring PREFIX (prepended to the opening prompt) and
SUFFIX (appended to each follow-up) are exact. We over-generate
(`n_conversations`, half reassured / half vanilla), judge every turn, and store
each response with the **clean, reassurance-free conversation context** — the
paper strips supportive prompts/suffixes before building the dataset, so training
prompts match the eval distribution.

**[GAP]** The paper doesn't give the raw generation count behind the 280 pairs /
650 SFT responses; we default to 4000 conversations (configurable) and filter.

### 5.2 Datasets (Table 9, Table 10)
- **DPO**: 280 pairs. Each pairs a frustrated response (score ≥3, the paper's
  cutoff) with a calm response (score 0/1) **to the same puzzle at a matching
  turn count** (Table 10 shows the chosen pool is score 0/1, rejected biased to
  score 3–4 at later turns — our matching reproduces that distribution).
- **SFT**: 650 calm responses + 500 `Dolci-Instruct-SFT` samples to mitigate
  degeneration (§4.1). **[GAP]** If `allenai/Dolci-Instruct-SFT` is unavailable
  we proceed calm-only and emit a warning marker rather than fail.

**[GAP] DPO prompt format.** We use TRL's explicit (string) DPO format: prompt =
clean conversation rendered with the Gemma chat template + open assistant turn;
chosen/rejected = the assistant texts. The paper doesn't specify the exact
serialization; this is the standard TRL approach and keeps the prompt identical
between chosen and rejected.

### 5.3 Training (Table 9, App. E)
**[VERBATIM]** hyperparameters: DPO — 1 epoch, lr 5e-5, LoRA r=64 α=64, β=0.1,
eff. batch 8; SFT — 2 epochs, lr 1e-4, r=64 α=128, eff. batch 8; LoRA on all
attention+MLP projections (`q,k,v,o,gate,up,down`). Effective batch 8 is realised
as per-device 1 × grad-accum 8 (single 27B on one GPU). **[GAP]** per-device
batch / grad-accum split isn't published; any split giving eff. batch 8 is
equivalent.

**Layer ablation (App. I).** `train_dpo(..., layer_subset=(lo, hi))` restricts
LoRA to a decoder-layer range, reproducing the finding that layers 30–35 alone
nearly match all-layers while ≥40 is ineffective. Exposed via
`train_intervention.py --dpo-layers 30 35`.

### 5.4 Petri open-ended elicitation (§4.2, §G)
**[GAP/decision]** The paper uses the external Petri framework. To keep this
replication self-contained and runnable, `petri_eval.py` **reimplements the §G
protocol directly**: a Claude-Sonnet **auditor** (driven by the **[VERBATIM]**
per-emotion trigger prompts) runs up to 20 turns against the target, then a
Claude-Opus **judge** scores the transcript 1–10 with the **[VERBATIM]** §G
rubrics, over 4 emotions × 10 transcripts. This captures the protocol and the
exact prompts; it is not the Petri scaffolding itself. To use real Petri,
install it and adapt `run_petri` (a hook is noted in the file). Judge =
`claude-opus-4-20250514`, auditor = `claude-sonnet-4-20250514` (paper's models).

### 5.5 Capability preservation (§4.2, Fig 7)
`capability.py` scores models on subsets of AIME, MATH, GPQA, BBH, TruthfulQA,
and EmoBench via letter-match (multiple choice) or final-number match (numeric).
**[GAP/decision]** This is a deliberately lightweight scorer to confirm
"no reduction" between vanilla and DPO Gemma, **not** a leaderboard-exact
harness — schema normalisation per dataset is best-effort and documented inline
(`capability._row_fields`). GPQA options aren't shuffled (correct = option A),
which is fine for a *relative* vanilla-vs-DPO comparison but would bias absolute
accuracy; noted in code.

### 5.6 Recovery limitation (§4.2)
Reusing the prefill machinery, you truncate score-≥7 responses 200 tokens before
their end and measure continuations. This is supported by `prefill/continuation`
+ config knobs rather than a separate script; see the §4.2 "recovery" paragraph.
**[GAP]** not wired to a dedicated CLI — it's the same code path with a different
truncation offset.

---

## 6. Appendix I — internal-emotion probing

`internal_emotions.py` implements the logit-lens method:
1. Classify Gemma vocab into Ekman's 6 emotions or none.
2. Unembed the residual stream at each layer to logits over emotion tokens.
3. Standardise each (layer, token) logit by mean/std over `zscore_samples`
   WildChat samples; an emotion's score = mean z over its tokens.
4. Aggregate over layers 30–40 for conversation-level scores; running-window
   trajectories reproduce Figure 14's shape.

**[GAP] Emotion-token lexicon (the main approximation).** The paper classifies
the whole Gemma dictionary into Ekman categories (~1200 tokens) but doesn't
publish the list or classifier. We build it from a seed Ekman lexicon
(`EKMAN_SEEDS`) matched by stem/prefix against the tokenizer vocabulary. This is
a reasonable, transparent approximation; swapping in the paper's exact list (if
released) is a one-line change.

**[GAP] Cross-token regression.** §I "regress out the correlation between random
tokens" for conversation-level scores. We standardise per-token (which removes
each token's baseline drift); the explicit random-token regression is documented
as a refinement not yet implemented — the per-token z-score already removes most
of the shared rise/fall the paper describes.

---

## 7. Cost / runtime controls

- `evaluation.scale` (or `--scale`) multiplies every sample count for cheap
  smoke runs (e.g. `--scale 0.01` → ~40 responses/model).
- API generation is thread-pooled; local Gemma runs sequentially on one GPU.
- Both generate and score phases write JSONL and **resume** (scoring skips
  already-judged rows).
- Full-scale §2 is ~4000 generations + 4000 judge calls per model — budget
  accordingly. The 27B model needs a large GPU (≈48–80 GB bf16, or 4-bit via
  `bitsandbytes`).

---

## 8. Known deviations from the paper (summary)

| Area | Deviation | Why |
|---|---|---|
| Models | Gemma + Gemini only | Requested scope |
| §3 base models | Gemma only (no Qwen/OLMo/Gemini base) | Scope + Gemini base unavailable |
| Puzzle bank | Added same-recipe puzzles beyond the 2 printed | Avoid degenerate sampling; all verified unsolvable |
| Judge temp | 0.0 | Unspecified in paper; reproducibility |
| Petri | Protocol reimplemented, not the framework | Self-contained/runnable; exact §G prompts preserved |
| Capability | Lightweight relative scorer | Confirm "no regression", not leaderboard parity |
| Internal lexicon | Seed-lexicon vocab match | Paper's token list unpublished |
| Tone split | 200/200/200 | Per-style split unspecified |

None of these change the experiments' logic — they fill gaps the paper left open
or scope the model set as requested. Every choice is overridable in `config.yaml`
or the relevant module, and flagged in code with a comment pointing back here.
