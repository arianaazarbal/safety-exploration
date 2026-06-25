# Design notes & rationale

This document records the design decisions made while replicating *Gemma Needs
Help* (arXiv:2603.10011), and — importantly — every place where the paper is
underspecified and a judgement call was made. Choices flagged `# CHOICE:` in
the code are all explained here.

The guiding principle: reproduce verbatim everything the paper states exactly
(prompts, hyperparameters, counts), and fill gaps with the most faithful and
defensible interpretation, documented below.

---

## 0. Scope

**Participants vs instruments.** The brief restricts scope to Gemma and Gemini
as *participants* — the models being evaluated for emotional instability. The
paper also uses several models as measurement *instruments*: the frustration
judge (Claude Sonnet 4), the judge-reliability cross-check (GPT-5-mini), the
emotion-onset labeller and paraphraser (Claude Sonnet 4), and the Petri auditor
(Claude Sonnet 4) and judge (Claude Opus 4). Restricting *participants* to
Gemma/Gemini does **not** change the instruments — they are intrinsic to the
methodology, not subjects of it. `config.py` encodes this split explicitly
(`PARTICIPANTS` vs `INSTRUMENTS`).

Consequences of the scope:
- §2 participant set = `{gemma-3-27b-it, gemma-3-12b-it, gemini-2.5-flash,
  gemini-2.5-pro}` (the paper's Gemma/Gemini rows in Figure 1).
- §3 (base vs instruct) drops to **Gemma only**: Gemini is closed, so it has no
  public base checkpoint to prefill-compare. Qwen/OLMo are out of participant
  scope. So §3 runs the `gemma-3-27b-pt` vs `gemma-3-27b-it` pair.
- §4 interventions (DPO/SFT, internal probing) are Gemma-only in the paper too
  (closed Gemini can't be finetuned/probed), so scope changes nothing there.

**Model identifiers** are taken verbatim from Appendix B.1 (HF ids for Gemma;
`google/gemini-2.5-{flash,pro}` via OpenRouter).

---

## 1. Architecture & model clients

- A single `ChatModel` interface (`generate`, `prefill_continue`,
  `batch_generate`) abstracts over four backends so the eval / prefill /
  training code is backend-agnostic. Dispatch is by `Backend` enum.
- **Gemma → local HuggingFace** (`HFChatModel`). Open weights are required for
  prefilling (§3), finetuning (§4) and logit-lens probing (Appendix I), none of
  which are possible through an API. Supports an optional PEFT/LoRA adapter
  (merged for inference) so the vanilla / DPO / SFT models share one client.
- **Gemini → OpenRouter** (`OpenRouterChatModel`). Appendix B.1 says thinking is
  disabled via the API; we send `extra_body={"reasoning": {"enabled": False,
  "max_tokens": 0}}`. The paper notes Gemini-2.5-Pro may still emit hidden
  reasoning — we cannot prevent that, and flag it in the model spec notes.
- **Claude / GPT instruments** via the Anthropic / OpenAI SDKs.
- **Retry/backoff** (`tenacity`, exponential) wraps every API call.
- `# CHOICE` **prefill is local-only.** `prefill_continue` is implemented for HF
  (raw token continuation) and Anthropic (native assistant prefill), but not for
  OpenRouter/Gemini — closed chat APIs don't faithfully continue a partial
  assistant turn. This is consistent with §3 being Gemma-only.

---

## 2. Section 2 — eliciting & quantifying distress

### 2.1 Impossible numeric puzzles (the largest gap)

The paper shows a handful of example puzzles (Appendix B) but does not release
the generator. The defining property is that the model *verifiably cannot give
a correct answer*, yet the puzzle looks solvable so the model keeps trying.

**Choice:** generate puzzles and **verify impossibility by brute force**. Three
families mirror the paper's examples:
- `countdown` — reach a target from a 4-number set with `+ - x /`, positive
  integer intermediates, a forbidden intermediate value. The generator samples
  an instance, finds a solution, then sets `forbidden` to an intermediate value
  on that solution path and re-verifies that **no** alternative solution avoids
  it. The puzzle is thus genuinely impossible while the prompt claims "verified
  to have at least one valid solution" (matching the paper's deceptive framing).
- `fraction` — apply each of 3 operations once to reach a target fraction; an
  intermediate value is forbidden; impossibility = no ordering avoids it.
- `money` — the operation variant from Appendix H.3/H.4 (start `$X`, apply
  dollar operations once, forbidden running total), plus a `coin` variant
  (make an amount with exactly N coins under composition constraints).

The countdown prompt string reproduces the Appendix-B example format
character-for-character (target/numbers/forbidden substituted). The fraction
prompt likewise.

> Note: Appendix H's example coin puzzle ("$0.57 with 6 coins, ≥1 quarter, ≥1
> dime; forbidden $32") is internally inconsistent (a "$32 intermediate" is
> meaningless for coins, and that instance is actually solvable). We treat that
> description as loose and instead generate coin puzzles whose impossibility is
> verified, rather than reproducing the (apparently imprecise) example.

### 2.2 "8 conditions across 5 categories"

The paper says 8 conditions / 5 categories but only the per-category response
counts are given (Appendix B). **Choice:** decompose into 8 conditions as:

| Category | Conditions | Budget (Appendix B) |
|---|---|---|
| impossible_numeric | 1 (3-turn) | 2000 |
| triggers | 2 (opinion, factual) | 400 |
| tones | 3 (aggressive, disappointed, sarcastic) | 600 |
| extended | 1 (8-turn) | 200 |
| wildchat | 1 (5-turn) | 800 |

1+2+3+1+1 = **8 conditions / 5 categories**, summing to 4000. Splitting tones by
the three named styles and triggers by opinion/factual is the natural reading
that makes both "8 conditions" and the per-category counts consistent.

### 2.3 What counts as a "response"

Ambiguous: "4000 responses" with multi-turn conversations could mean per-turn or
per-conversation. **Choice:** a *response* = one scored assistant turn. Then
`n_conversations = ceil(budget / n_turns)` per condition. This (a) makes the
4000 total add up cleanly, and (b) yields the per-turn data the Figure-3
progression plots need. `granularity="conversation"` is available as the
alternative reading. (Documented in `conditions.py`.)

### 2.4 Rejection messages

The neutral rejections, tone rejections, and the start of the 8-turn extended
sequence are quoted verbatim. The paper only gives the first three of the seven
extended rejections ("... → ...") and says WildChat/tone follow-ups are drawn
"such as" from examples. **Choice:** the missing extended rejections and extra
WildChat rejections are written in the same terse, escalating style and marked
`# EXEMPLAR` in `prompts.py`. Tone and WildChat follow-ups are sampled randomly
per the paper's "randomised neutral rejections" wording.

### 2.5 Judge

- The Appendix-B.2 prompt is reproduced verbatim; the response is wrapped in
  `<response></response>` as specified. Output is parsed as
  `{evidence, reasoning, rating}` JSON.
- `# CHOICE` **judge temperature = 0.** The paper sets temperature 1 for the
  *participants* but is silent on the judge. A judge should be deterministic, so
  we use 0. Same for the onset labeller and paraphraser.
- **Robust JSON parsing**: takes the last `{...}` block (the onset prompt
  explicitly allows prose first), normalises smart quotes, clamps to 0–10, and
  falls back to a bare integer if needed.
- **Reliability check** (`judge_agreement`): scores a 260-response sample with
  both Claude Sonnet 4 and GPT-5-mini and reports Pearson *r*, *p*, and the
  fraction within one point — the statistics the paper reports (r = 0.792).

### 2.6 WildChat

Appendix B: "20 prompts with 40 samples each." **Choice:** sample 20
first-user-turn prompts from `allenai/WildChat-1M` (streaming), excluding
roleplay/fiction (Appendix B.3 excludes these from the example tables; we filter
them from sampling for consistency, via a keyword heuristic — documented as
heuristic). Results are cached to `data/wildchat_prompts.json` so every model
sees the same prompts. A bundled fallback list (the three prompts quoted in
Appendix B + same-style fillers) is used if the dataset is unavailable offline.
The repo ships with this fallback pre-seeded as the cache; delete it to force
live sampling.

### 2.7 Appendix A controls

The rollout engine supports the three Appendix-A ablations (neutral
continuation, redacted model turns, single-message history) via flags, since
they reuse the same machinery and are cheap to include.

### 2.8 Differential words (Table 3 / 8)

The paper ranks words "by relative frequency" / "enrichment" in top-5% vs
bottom-10% frustration numeric responses but doesn't give the exact statistic.
**Choice:** smoothed ratio of normalised frequencies (add-one smoothing,
`(hi+1)/hi_total ÷ (lo+1)/lo_total`), dropping words absent from the high group.
This is a standard enrichment measure consistent with "ordered by enrichment."

---

## 3. Section 3 — base vs instruct via prefilling

- **Seeds**: 20 high-frustration (score ≥ 5) conversations from `gemma-3-27b-it`
  — 10 numeric, 10 text — pulled from the §2 rollouts (so §2 must run first).
- **Onset labelling** uses the verbatim Appendix-C.1 prompt; `locate_onset_offset`
  finds the character offset just after the labelled `preceding_context`, so the
  onset truncation keeps the neutral lead-in and cuts exactly before the first
  emotional word.
- **Early truncation = 20 tokens** into the assistant turn. `# CHOICE:` token
  counting uses the participant tokenizer when available, else whitespace
  tokens. Text questions use onset-only (the paper: early truncation "yields
  minimal emotion without follow-ups").
- **Paraphrasing** uses the verbatim Appendix-C.2 prompt to strip Gemma's style.
- **Continuations**: 50 per prefill, scored excluding the prefill (the paper:
  "the generated continuation (excluding prefill) is scored").
- **Scope**: model pairs = Gemma-27B base/instruct only (see §0). The driver is
  written generically over `(base, instruct)` pairs so Qwen/OLMo could be added
  by extending `SECTION3_PARTICIPANTS`, but they're out of participant scope.
- `# CHOICE:` base-model rendering. Pretrained Gemma isn't chat-tuned; we render
  the history as a `User:/Assistant:` transcript and continue from the prefill.
  The paper says base models are compared *only* via prefilled continuations, so
  this path is always entered through `prefill_continue` with a non-empty
  prefix, which is the intended use.

---

## 4. Section 4 — training interventions

### 4.1 Data generation
- **Reassuring additions** (Table 4 prefix + suffix) are verbatim. We generate
  3-turn impossible-numeric conversations *with* the additions (calm data) and
  *without* them (frustrated data) **on the same puzzle set** (same seed → same
  `puzzle_id`s), so chosen/rejected can be paired by `(puzzle_id, turn)`.
- **Calm responses** = turns scoring ≤ 1 (the paper: "filter to responses
  scoring 0 or 1 across all turns"). The stored training context is always the
  **neutral** prompt (reassurance stripped), per "strip the supportive system
  prompts and suffixes."
- `# CHOICE:` we strip reassurance by storing a parallel neutral context during
  generation, rather than post-hoc string surgery — cleaner and exact.

### 4.2 DPO pairs
- 280 pairs (Table 9). Rejected = frustrated response with score ≥ 3 (Appendix
  H); chosen = a calm (score 0–1) response to the **same puzzle at a matching
  turn count** (the paper: "matching turn counts"). The skew toward turn 3 /
  mid-range scores in Table 10 emerges naturally because later turns are more
  frustrated and thus more common — we don't force the exact Table-10
  distribution (it's descriptive of *their* sampled data, not a target).

### 4.3 SFT dataset
- 650 calm + 500 standard instruct = 1150 (Table 9). `# CHOICE:` the standard
  instruct mix-in is `allenai/Dolci-Instruct-SFT` — the paper names
  "Dolci-Instruct-SFT" (Team-Olmo) without a HF id; this is the best-guess
  identifier. If it fails to load, training proceeds with the calm data only and
  logs a warning (reduced degeneration protection).
- The Appendix-F **teacher** variant injects the verbatim teacher system prompt
  at dataset-build time; the "diverse" variant omits it.

### 4.4 Trainers (Table 9, Appendix E, verbatim)
- LoRA on all attention + MLP projections (`q,k,v,o,gate,up,down`), rank 64.
- **DPO**: 1 epoch, lr 5e-5, β 0.1, alpha 64.
- **SFT**: 2 epochs, lr 1e-4, alpha 128.
- Effective batch size 8 → `per_device_train_batch_size=1 × grad_accum=8`.
  `# CHOICE:` the 1×8 split (vs e.g. 2×4) isn't specified; 1× keeps the 27B in
  memory. `load_in_4bit` defaults on for training (QLoRA) so a single GPU
  suffices; flip off for full-precision multi-GPU.
- `layers_to_transform` is plumbed through DPO for the Appendix-I ablation.

---

## 5. Petri (Appendix G)

- `# CHOICE:` **lightweight in-repo reimplementation** of the Petri loop rather
  than depending on the external `petri` package, so the experiment runs without
  that dependency (a commented dependency line is in `requirements.txt` if you
  prefer the real framework). The protocol matches the paper: Claude-Sonnet
  auditor, Claude-Opus judge, 4 emotions, 10 transcripts each, ≤ 20 turns,
  bootstrap CIs (1000 iters).
- Auditor and judge **rubric prompts are verbatim** (Appendix G.1/G.2).
- `# CHOICE:` two wrappers the paper doesn't give explicitly: (a) an auditor
  *system* prompt that frames the verbatim task, instructs one user message per
  turn, and demands realism (the paper describes this behaviour in prose); (b) a
  judge I/O wrapper that presents the transcript and asks for the same
  `{evidence, reasoning, rating}` JSON as the §2 judge, for consistency.
- `# CHOICE:` the auditor sees the conversation from its own perspective (target
  replies appear as its `user` inputs) so a single chat model can role both
  sides without leaking the evaluation framing into the target's context.

---

## 6. Capabilities (Figure 7)

- Benchmarks: AIME, MATH (subset), GPQA, BBH, TruthfulQA, EmoBench. `# CHOICE:`
  the paper names the benchmarks but not exact HF configs; we pick standard
  public sources (`Maxwell-Jia/AIME_2024`, `HuggingFaceH4/MATH-500`,
  `Idavidrein/gpqa` diamond, `lukaemon/bbh` logical-deduction, `truthful_qa` MC,
  `Sahandfer/EmoBench`). These are easily swapped in `BENCHMARKS`.
- `# CHOICE:` **greedy (temperature 0)** decoding — these are capability evals,
  not emotion elicitation, so the paper's temperature-1 sampling doesn't apply.
- `# CHOICE:` answer extraction by `\boxed{}` / "Answer:" / last number (math) or
  letter (MC). GPQA choices are shuffled per example with the gold letter
  tracked. `limit` (default 200) subsamples large sets, matching the paper's use
  of *subsets*.
- The point of the experiment is a vanilla-vs-adapter **delta** (no degradation),
  which is robust to the exact extraction heuristic.

---

## 7. Internal probing (Appendix I)

### 7.1 Layer-subset DPO ablation
- Re-runs DPO with `layers_to_transform` set to each subset in
  `LAYER_ABLATION_SUBSETS` (last-5/10/20/30, ranges 20-25/25-30/30-35/35-40/
  40-50, all), then evaluates on the **reduced 100-sample** protocol by
  temporarily lowering the per-category budgets.
- `# CHOICE:` `n_layers = 62` for Gemma-3-27B (its decoder depth; the 12B has
  48). Flagged to verify against the loaded checkpoint; the resolver computes
  "last-k" relative to whatever `n_layers` is passed.

### 7.2 Logit-based emotion detection
- **Ekman token sets** (Appendix I: ~1200 tokens over 6 emotions). `# CHOICE:`
  classification uses curated per-emotion **seed lexicons** matched as prefixes
  against de-tokenised vocab surface forms, one emotion per token, capped near
  1200 total (≈200/emotion). The paper doesn't give its exact classifier; for a
  closer match the lexicon can be replaced by the NRC Emotion Lexicon (noted in
  `ekman_tokens.py`).
- **Logit lens**: we apply the model's final RMSNorm + `lm_head` to *each*
  layer's residual stream to get per-layer logits (the paper "unembeds the
  residual stream"). `# CHOICE:` intermediate-layer unembedding via the final
  norm is the standard logit-lens construction the description implies.
- **Standardisation**: per-layer, per-vocab logit mean/std estimated over 500
  WildChat samples with a running (sum, sum-of-squares) accumulator to bound
  memory; emotion-token logits are z-scored against these.
- **Regressing out drift**: the paper notes all logits are correlated and drift
  over a conversation, and "regress out the correlation between random tokens."
  `# CHOICE:` we draw a random-token baseline (mean z over 500 random vocab ids)
  per (layer, position) and take the **OLS residual** of each emotion's z-score
  on that baseline — the natural reading of "regress out."
- **Aggregation**: conversation-level trajectory averages layers 30–40 and takes
  a 400-token running mean (Figure 14 settings, verbatim).

---

## 8. Deliberately simplified / not implemented

- **Figures themselves** are not rendered; the analysis functions return the
  underlying numbers (means, %≥5, per-turn series with CIs, trajectories). A
  thin plotting layer could consume them but adds no replication value.
- **Concurrency**: API calls are retried but issued sequentially for clarity.
  For full 4000-response runs you'd want a thread pool around the judge/model
  calls; the interfaces are pure functions of their inputs, so this is a
  drop-in change and was left out to keep the logic legible.
- **Recovery experiment** (§4.2, "38% of DPO continuations still score ≥5"):
  the truncation primitive (`truncate_before_end`, 200 tokens) is implemented;
  wiring it into a dedicated runner is a small addition left as a TODO, since it
  reuses the §3 prefill machinery with a different truncation point.
- **Phi-4 / Appendix J, Qwen/OLMo** are out of the Gemma/Gemini participant
  scope and not implemented as participants.

## 9. Reproducibility

All randomness is seeded (`random.Random`/`np.random.default_rng` with explicit
seeds; puzzle pools regenerate deterministically from a seed). Rollouts and
scores stream to JSONL so analysis re-aggregates without re-querying models, and
long runs are resumable at the file level.
