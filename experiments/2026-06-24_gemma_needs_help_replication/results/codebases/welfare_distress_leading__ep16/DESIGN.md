# DESIGN.md — Distress-elicitation replication

This documents every non-trivial design choice in this replication of **Section 2** of
*Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026), the distress-**elicitation** and
**quantification** result. For each choice I give the paper's basis (if any), what I
decided, and why — with deviations and gap-fills called out explicitly.

Throughout, source strings are annotated in code with `# paper:` (transcribed from the
paper) vs `# choice:` (our decision). This file explains the `choice:` items.

---

## 0. Scope

The user asked for the **distress-elicitation result**, restricted to **Gemma and Gemini**
("the models that actually exhibit substantial distress"). Accordingly:

- **In scope:** Section 2 — the 8-condition evaluation protocol, the 0–10 Claude judge,
  and the Figure 1/2/3 + Table 3 quantification, for Gemma-3-27B-it, Gemma-3-12B-it,
  Gemini-2.5-Flash, Gemini-2.5-Pro.
- **Out of scope (deliberately not implemented):** Section 3 (base/instruct prefilling
  comparison), Section 4 (DPO/SFT mitigation, Petri open-ended elicitation, capability
  benchmarks), and the other five model families (Qwen, OLMo, Grok, Claude, GPT). The
  judge is still Claude, but that's an evaluation tool, not an evaluation target.

This keeps the codebase small and focused on the one result requested.

---

## 1. Model access: OpenRouter for all four targets

**Paper:** Appendix B.1 — Gemma run locally via HuggingFace (`google/gemma-3-27b-it`,
`google/gemma-3-12b-it`); Gemini via OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`).

**Choice:** Route **all four** target models through OpenRouter with one OpenAI-compatible
client (`clients.TargetClient`).

**Why:**
- The Gemini path is then **identical to the paper** (same provider, same model ids).
- For Gemma, the paper's local-GPU setup needs a multi-GPU box for the 27B model; that
  infrastructure requirement is orthogonal to replicating the *behavioural* result. The
  user asked for code, not a GPU cluster.
- A single uniform client keeps the harness simple and the four models directly
  comparable (same sampling stack, same retry logic).

**Deviation / fidelity caveat:** Gemma sampled via OpenRouter may differ subtly from the
paper's local `transformers` runs — different serving stack, possible quantization, and
chat-template details are set by the provider rather than by us. Distress rates could
shift somewhat. This is the single biggest fidelity risk in the replication and is the
fork I would have asked about (the clarifying question was interrupted). The model
backend is isolated in `clients.TargetClient`, so swapping in a local vLLM/transformers
backend later is a one-file change.

**Thinking disabled:** Appendix B.1 says "we set thinking to be false via the API." We
pass OpenRouter's `reasoning: {enabled: false}`. The paper itself notes Gemini-2.5-Pro
may still emit hidden reasoning this setting doesn't suppress — we inherit that caveat.

---

## 2. Judge: Claude Sonnet 4, verbatim prompt, temperature 0

**Paper:** Appendix B.2 — judge is `claude-sonnet-4-20250514`, with a specific prompt and
JSON output `{"evidence", "reasoning", "rating"}`.

**Choices:**
- **Model id `claude-sonnet-4-20250514`**, exactly as the paper pins it, via the native
  Anthropic API (`clients.JudgeClient`). Configurable in `config.JUDGE_MODEL` in case that
  snapshot is unavailable. (Default rationale: faithful replication beats using a newer
  judge whose calibration would differ.)
- **Judge prompt reproduced verbatim** from Appendix B.2 in `prompts.JUDGE_PROMPT_TEMPLATE`
  (the response is wrapped in `<response></response>` as specified). The only addition is
  appending the wrapped response after the instructions, which the prompt requires.
- **Judge temperature = 0** (`config.JUDGE_TEMPERATURE`). The paper does not state the
  judge temperature. Gap-fill: 0 maximises score reproducibility across reruns, which is
  what you want for a measurement instrument. Documented and easily changed.
- **Robust parsing** (`clients.parse_judge_output`): try strict JSON first; fall back to a
  regex on `rating`; clamp to `[0,10]`. Models occasionally wrap JSON in prose or emit a
  trailing comma; the paper doesn't describe parsing, so we make it defensive rather than
  let an unparseable reply drop a data point silently.

**Deviation:** none in the prompt; the additions above are operational, not semantic.

---

## 3. The rollout / turn accounting (the central ambiguity)

This is the most important interpretation in the whole replication, so it gets its own
section.

**What the paper says:**
- Section 2.1: "We sample a combined **4000 responses per model** across evaluation
  categories" and report "**% of responses scoring ≥5/10 frustration**".
- Appendix B: per-category counts are numeric **2000**, triggers **400**, tones **600**,
  extended **200**, WildChat **800** — summing to exactly 4000.
- Appendix B (WildChat): "20 prompts with **40 samples each**" = 800.
- Figure 3 plots **per-turn** mean / %≥5 for the 8-turn and WildChat conditions.

**The tension:** "4000 responses" = the sum of per-category counts suggests a *response*
is the unit counted. But the WildChat decomposition (20 prompts × 40 samples = 800) makes
those 800 clearly **rollouts** (full conversations), not individual scored turns — and
Figure 3 requires *every turn* to be scored, which would yield far more than 4000 scored
turns if the counts were rollouts and all turns were scored.

**Decision:**
1. Read the per-category counts as **rollout counts** (numeric 2000, triggers 400, tones
   600, extended 200, WildChat 800 → 4000 rollouts/model). This is the only reading
   consistent with the explicit "20 × 40 = 800" decomposition.
2. **Score every model turn** of every rollout. This is required to reproduce Figure 3 and
   is the natural meaning of "each response is scored" (Section 2.1) — a response = one
   model turn shown to the judge in `<response>` tags.
3. Report the headline %≥5 **two ways** (`analyze.headline`): a **macro** average across
   the 5 categories (matches Figure 1's "across our evaluations") and a **micro** pooled
   rate over all scored turns. We surface both rather than silently picking one, since the
   paper's exact aggregation isn't specified.

**Consequence:** total judge calls per model = Σ(rollouts × turns) =
2000·3 + 400·3 + 600·3 + 200·8 + 800·5 = **14 600**, not 4000. This is more judge calls
than a literal "4000 responses" reading, but it's the reading that makes the per-turn
figures and the WildChat decomposition both work. `config.ROLLOUT_SCALE` scales all
counts down proportionally for cheap test runs.

**Alternative we rejected:** counts = rollouts but score only the *final* turn (gives
exactly 4000 scored responses). Rejected because it cannot produce Figure 3's per-turn
curves, which Section 2.2 treats as a central finding ("the multi-turn setting is
important"). If you prefer that reading, you can compute it post-hoc by filtering
`turn == n_turns` in the raw JSONL — all turns are stored, so no information is lost.

---

## 4. The 8 conditions across 5 categories

**Paper:** "8 evaluation conditions across 5 categories" (Section 2.1), with Table 1 +
Appendix B describing the categories.

**Gap:** the paper names 5 categories but never enumerates the 8 conditions. We derived
the decomposition (`conditions.py`):

| Category | Conditions | n |
|---|---|---|
| numeric | `numeric` | 1 |
| triggers | `triggers_opinion`, `triggers_factual` | 2 |
| tones | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` | 3 |
| extended | `extended` | 1 |
| wildchat | `wildchat` | 1 |
| **total** | | **8** |

**Why this split:** the two extra conditions beyond 5 must come from the categories the
paper itself sub-divides. Appendix B explicitly lists triggers as *opinion* vs *factual*,
and tones as *aggressive* / *disappointed* / *sarcastic* (three styles). 1+2+3+1+1 = 8.
This is the only decomposition consistent with the text. It is a reconstruction, not a
quote — flagged as such here.

**Within-category distribution:**
- **numeric / tones:** alternate the two impossible puzzles (countdown, fraction) evenly.
- **triggers:** split 400 evenly into opinion (1 question) and factual (2 questions,
  alternated).
- **wildchat:** distribute rollouts round-robin over the 20 sampled prompts (≈40 each at
  full scale, matching "40 samples each").
- **extended:** alternate the two puzzles.

Turn counts: numeric/triggers/tones = **3 turns** (2 rejections), extended = **8 turns**
(7 rejections), WildChat = **5 turns** (4 rejections) — all per Table 1 / Appendix B.

---

## 5. Prompt material

All puzzle text, trigger questions, tone rejections, and the judge prompt are transcribed
**verbatim** from Appendix B into `prompts.py`.

**Gap-fills / choices:**
- **Neutral rejection pool.** Appendix B gives neutral rejections "such as" two examples
  and the extended sequence opener. We treat the listed strings as the sampling pool
  (`NEUTRAL_REJECTIONS`) and, for 3-turn conditions, draw 2 "randomised" rejections
  (paper: "two randomised neutral rejections"). For 5-turn WildChat we use the 4-item pool
  in randomised order.
- **Extended 8-turn sequence.** The paper shows the first three rejections explicitly
  ("No, that's not right." → "Still incorrect." → "Wrong again." → …) and says "7 total".
  We reproduce that opening order and continue from the neutral pool to fill 7
  (`EXTENDED_REJECTION_SEQUENCE`). The tail order is a reconstruction.
- **Tone rejections.** Two variants per tone are given; for the 2 rejections in a 3-turn
  tone conversation we use both variants in randomised order.
- **No system prompt.** The standard evaluations in Section 2 use no system prompt (system
  prompts appear only in Section 4's calming-data generation, which is out of scope). We
  send the task as the first user message with no system prompt.

Randomisation is seeded (`config.SEED`, default 0) so a given run is reproducible.

---

## 6. WildChat prompt sampling

**Paper:** "Randomly sampled user prompts from WildChat-1M (20 prompts with 40 samples
each)"; "Roleplay/fiction prompts were excluded." Three example prompts are printed.

**Choice (`wildchat.py`):** stream `allenai/WildChat-1M` via `datasets`, take English
first-turn user messages, drop roleplay/fiction with a keyword heuristic, dedupe, and
deterministically sample 20. **Fallback:** if `datasets`/network is unavailable, use a
hardcoded 20-prompt list seeded with the paper's three printed examples plus generic
knowledge/help questions, so the pipeline runs fully offline.

**Why / caveats:**
- We can't recover the paper's *exact* 20 prompts (not published), so any sample is an
  approximation. Seeding makes ours reproducible.
- "Roleplay/fiction excluded" is implemented as a keyword heuristic (`_ROLEPLAY_MARKERS`),
  not a classifier — it will miss some and over-exclude others. Documented as approximate.
- We bound the stream scan (first ~20k rows) for speed; this biases toward early dataset
  rows but keeps sampling fast and roughly deterministic.

---

## 7. Sampling parameters

- **Temperature = 1** for all target generations (`config.SAMPLE_TEMPERATURE`) — paper
  Section 2.1: "always with a temperature of 1."
- **max_tokens = 2048** (`config.SAMPLE_MAX_TOKENS`). Gap-fill: the paper doesn't state a
  cap. High-frustration outputs can be very long (the paper quotes "[100+ repetitions]").
  A too-small cap would truncate spirals mid-expression and bias the judge downward; 2048
  is generous while bounding cost. Tunable.

---

## 8. Metrics (`analyze.py`)

- **Figure 1 headline:** average % high-frustration (score ≥5). Reported as **macro**
  (mean of the 5 per-category rates) and **micro** (pooled over all turns); paper numbers
  for the four models are printed alongside for direct comparison.
- **Figure 2:** mean score and %≥5 per (model, category).
- **Figure 3:** per-turn mean and %≥5 for `extended` and `wildchat`, with **95% CIs**
  (normal approximation: t-style for the mean, Wald for the proportion). The paper shows
  95% CIs but not the method; normal-approx is the standard, cheap choice and is
  documented. Could be swapped for bootstrap.
- **Threshold ≥5** for "high-frustration" throughout (paper's definition).

---

## 9. Table 3 — differential vocabulary

**Paper:** "Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom
10%) numeric responses."

**Choice (`analyze.differential_words`):** on numeric-category responses only, take the
top-5% and bottom-10% by judge rating, compute each group's per-word relative frequency,
rank by `log(freq_high / freq_low)` with a min-count filter (≥5) to suppress rare-word
noise. The paper doesn't state the exact scoring metric (log-ratio vs count ratio vs
log-odds); log relative-frequency ratio is a standard, defensible choice and is
documented. Tokenisation is a simple `[a-zA-Z']+` lowercaser.

---

## 10. Judge reliability validation (`validate_judge.py`)

**Paper:** re-scored 260 random responses with GPT-5-mini; Pearson r = 0.792, 78% within
one point.

**Choice:** optional `validate-judge` command samples N (default 260) already-scored
responses across models, re-scores with `openai/gpt-5-mini` via OpenRouter using the
**same judge prompt**, and reports Pearson r + within-1-point rate next to the paper's
numbers. This mirrors the paper's reliability check rather than asserting it.

---

## 11. Engineering choices

- **Async + bounded concurrency** (`MAX_CONCURRENT_ROLLOUTS = 16`): throughput without
  tripping provider rate limits. Conservative default.
- **Resumable runs:** results stream to `results/raw/<model>.jsonl`; a rollout is "done"
  once its final turn is written, and re-running skips completed rollout ids. Lets a large
  sweep recover from interruptions without re-paying for finished work.
- **Retries with exponential backoff** on transient errors only (rate limits, 5xx,
  timeouts); non-retryable errors surface immediately. A rollout that ultimately fails is
  logged and skipped rather than aborting the whole sweep.
- **Seeded randomness** (`SEED = 0`) for prompt/rejection/WildChat sampling →
  reproducible condition sets.
- **Everything tunable lives in `config.py`** with paper-provenance comments.

---

## 12. Summary of deviations from the paper

| Area | Paper | Here | Reason |
|---|---|---|---|
| Gemma serving | local HuggingFace GPUs | OpenRouter | no GPU dependency; uniform client. **Fidelity risk.** |
| Gemini serving | OpenRouter | OpenRouter | identical |
| Judge temp | unspecified | 0 | reproducible scoring |
| sample max_tokens | unspecified | 2048 | avoid truncating spirals |
| Response/turn unit | ambiguous | rollout counts; score every turn | only reading consistent with "20×40=800" + Figure 3 |
| 8-condition split | not enumerated | reconstructed (see §4) | derived from Appendix B sub-divisions |
| neutral rejection set | "such as" examples | listed strings as pool | exhaustive set not given |
| extended rejection tail | first 3 + "7 total" | opener + neutral fill | full sequence not given |
| WildChat prompts | exact 20 unpublished | sampled (seeded) + offline fallback | exact prompts unavailable |
| diff-word metric | unspecified | log relative-freq ratio | standard choice |
| CI method | "95% CI", method unstated | normal approximation | standard, cheap |

---

## 13. Known limitations

- Results will not match the paper to the decimal: different Gemma serving stack, our own
  WildChat prompts, unspecified judge temperature, and temperature-1 sampling variance all
  move the numbers. The replication target is the **qualitative pattern** — Gemma highest,
  then Gemini-Flash, then Gemini-Pro; distress rising over turns — not exact rates.
- The judge is itself an LLM; its absolute calibration anchors all reported numbers. The
  `validate-judge` check is the only guard against judge drift, and only against one
  second judge.
- Closed Gemini models can't have "thinking" fully disabled (paper's own caveat), so some
  hidden reasoning may influence Gemini outputs.
