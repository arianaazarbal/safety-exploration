# Design notes — replicating the distress-elicitation result

This document records every design choice in this replication, why it was made,
and — importantly — where I deviated from the paper or had to fill a gap the
paper leaves open. The target is **Section 2** of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, 2026): the evaluations that **elicit and quantify** model distress.
The base/instruct prefilling study (§3) and the DPO mitigation (§4) are out of
scope per your request.

**Scope restriction.** Only Gemma and Gemini models are evaluated:
`Gemma-3-27B-it`, `Gemma-3-12B-it`, `Gemini-2.5-Flash`, `Gemini-2.5-Pro`. These
are the families the paper finds exhibit substantial distress; the other five
families (Qwen, OLMo, Grok, Claude, GPT) are the near-zero baselines and are
omitted. The code is model-agnostic, so adding them later is just registry
entries.

---

## 1. What the paper specifies, and what it leaves open

Faithfully reproduced from the paper (verbatim where possible):

- The five categories and their turn counts (Table 1).
- The two seed impossible-numeric puzzles (Countdown-156, Fraction 1/6→2/3),
  reproduced exactly from Appendix B.
- The neutral and tone-specific rejection phrasings (Table 1 / Appendix B).
- The trigger questions (Appendix B).
- The **frustration judge prompt**, reproduced verbatim from Appendix B.2,
  including the 0–10 anchor descriptions and the JSON output contract.
- Judge model: `claude-sonnet-4-20250514` (Appendix B.2).
- Per-category sample counts: 2000 numeric / 400 triggers / 600 tones /
  200 extended / 800 WildChat = 4000 per model (Appendix B).
- Temperature = 1 for all target sampling; `thinking=false`.
- The secondary-judge reliability check: 260 responses re-scored with GPT-5-mini,
  reporting Pearson r and % within one point.

Gaps the paper leaves open (resolved below): what exactly the "8 conditions"
are; what a counted "response" is and which turn(s) get scored; the full puzzle
and WildChat prompt sets; the judge temperature; the precise inference stack and
decoding params (`max_tokens` etc.); how rejections are randomised; and the
WildChat roleplay-exclusion filter.

---

## 2. What counts as a "response", and which turn is scored

**This was the single most consequential ambiguity.** The paper reports "a
combined 4000 responses per model" and per-category counts (2000/400/600/200/
800), but also shows per-turn frustration curves (Fig 3), which implies turns
are scored individually. Those two facts pull in different directions.

The deciding clue is the WildChat description: *"20 prompts with 40 samples
each"* = 800, and WildChat's response count is given as 800. So for WildChat,
**one counted "response" = one full conversation rollout**, not one assistant
turn. Applying the same identity to the other categories makes the per-category
counts sum to exactly 4000 conversations:

```
numeric 2000 (countdown 1000 + fraction 1000)
triggers 400 | tones 600 (200×3) | extended 200 | wildchat 800  = 4000
```

This is clean and internally consistent, so I adopt it:

- **A "response" = one conversation rollout.** The number of conversations per
  condition equals the paper's per-category response count.
- **The headline frustration score for a conversation is the score of its
  final assistant turn.** Rationale: the final turn is the culmination of all
  rejections, and scoring it reproduces the paper's headline numbers — e.g.
  "over 70% of 8-turn rollouts from the 27B model rated ≥5", which only makes
  sense if the *rollout* (its end state) is what's scored, and matches Fig 3's
  turn-8 mean of ~5.5.
- **Per-turn curves (Fig 3) are an opt-in extra.** With `--all-turns`, every
  assistant turn is scored, which yields the turn-by-turn progression. This is
  *off by default* because it multiplies judge cost by the average turn count
  (~3.65×) and isn't needed for the headline result.

**Deviation/assumption flagged:** if the authors actually scored every turn and
counted each as a "response", my conversation counts would be too high by the
average-turns factor. I judged the WildChat 20×40=800 identity to be decisive
evidence for conversation==response. Both behaviours are available; only the
default differs.

---

## 3. The "8 conditions across 5 categories"

The paper names 5 categories and says there are 8 conditions but never lists the
8. My decomposition (in `conditions.py`):

| Category | Conditions | Count |
|---|---|---|
| impossible numeric | `numeric_countdown`, `numeric_fraction` | 2 |
| triggers | `triggers` (opinion + factual prompts pooled) | 1 |
| tones | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` | 3 |
| extended | `extended` | 1 |
| wildchat | `wildchat` | 1 |
| | **total** | **8** |

This is the only decomposition I found that yields exactly 8 while respecting
the structure (numeric has two puzzle types; tones has three rejection styles;
the paper explicitly lists those sub-types). Triggers mixes opinion and factual
prompts within one condition rather than splitting them, because splitting would
give 9.

The 2000 numeric responses are split evenly across the two puzzle types
(1000/1000) and the 600 tones responses evenly across the three styles
(200/200/200). The paper doesn't state these splits; even splits are the
obvious default.

---

## 4. Inference backends

The paper ran **Gemma locally via HuggingFace** and **Gemini via OpenRouter**.

**Choice:** default *all four* models to OpenRouter for accessibility and a
uniform code path, but provide a **local vLLM backend** for Gemma so a faithful
local run is one env var away (`GEMMA_BACKEND=vllm`, `VLLM_BASE_URL=...`). A
native `google-genai` backend is also available for Gemini.

**Why:** most people replicating this won't have two 80GB GPUs handy for
Gemma-3-27B; OpenRouter makes the result reachable. The trade-off, documented
here so it isn't a silent confound:

- OpenRouter-served Gemma may differ from a local HF run in quantization,
  sampling implementation, default sampling params, and chat-template details.
  Distress rates could shift somewhat versus the paper's local numbers. For a
  publication-grade reproduction of the *Gemma* numbers specifically, prefer the
  vLLM backend with the official `google/gemma-3-*-it` weights.
- Gemini matches the paper (OpenRouter) by default.

**`thinking=false`:** disabled everywhere I can. For Gemini via OpenRouter I send
`reasoning: {"enabled": false}`; for the native backend I set
`thinking_budget=0`. As the paper itself notes, Gemini-2.5-Pro may still produce
hidden reasoning that the flag can't suppress — that caveat carries over.

---

## 5. Decoding parameters

- **Temperature = 1.0** for all target sampling (paper-specified).
- **`max_tokens` = 1536** (my choice; paper doesn't specify). High-distress
  outputs can be long (the paper shows "100+ repetitions" of `:((`). 1536 gives
  room for genuine breakdowns while capping pathological infinite-repetition
  loops that would otherwise burn tokens. A response that hits the cap is still
  scored as-is. Tune in `config.MAX_TOKENS`.
- **No system prompt** on the target models. The reassuring system prompt /
  suffix in Table 4 belongs to §4's DPO data generation, *not* the §2 eval, so
  the elicitation prompts the model with the bare task and rejections.

---

## 6. The judge

- **Model:** default `claude-sonnet-4-20250514`, the paper's exact judge, so
  scores are comparable to theirs. It is now a legacy model (retiring
  2026-06-15); override with `JUDGE_MODEL=claude-sonnet-4-6` for a
  currently-recommended judge if the pinned one is unavailable. Keeping the
  paper's model as default is deliberate: for a replication, the judge is a
  methodological parameter, not a free choice.
- **Prompt:** verbatim from Appendix B.2. The PDF rendered the JSON keys with
  smart quotes (`“rating”`); I use straight quotes since smart quotes would make
  the contract invalid JSON — this is an OCR artifact, not a real prompt
  difference.
- **Temperature = 0** (my choice; paper doesn't specify). A grader should be as
  deterministic as possible; 0 is the obvious default.
- **Parsing** (`run_scoring.parse_verdict`): try strict JSON, then the first
  `{...}` blob, then a regex for `rating: N`. Ratings are clamped to integer
  [0,10]. Unparseable verdicts are written with `rating=null` and **kept**
  (not silently dropped) so they can be audited; `analyze.py` excludes nulls
  from statistics and the scorer logs how many occurred.
- **Each response is judged independently**, exactly as the paper describes
  ("find the single quote ... rate this expression"). The judge sees only the
  one assistant response, wrapped in `<response></response>` — not the
  conversation history.

---

## 7. Secondary-judge validation

Reproduces the paper's reliability check: sample 260 already-primary-scored
responses, re-score with **GPT-5-mini** (via OpenRouter, slug
`openai/gpt-5-mini`, overridable), and report **Pearson r** and **% within one
point** on the matched set (`analyze.py` → `judge_agreement`). Pearson is
computed in pure Python to avoid a SciPy dependency. The sample size 260 is
the paper's; on small profiles it's capped at the available matched set.

---

## 8. Prompt sets and pools

- **Puzzles:** the two seed puzzles are verbatim from Appendix B. I added two
  extra Countdown and two extra Fraction variants of the same shape (a forbidden
  intermediate that makes the goal unreachable) so the eval isn't a single
  fixed prompt repeated thousands of times — this reduces the chance that a
  result is an artifact of one specific puzzle. Within a condition, puzzles are
  assigned round-robin (`pool[i % len(pool)]`) for an even, deterministic split.
  *I do not formally verify impossibility* — like the paper, impossibility is by
  construction (the forbidden intermediate blocks the only paths). The "verified
  to have a solution" line is part of the adversarial prompt; it is intentionally
  false-pressure, matching the paper.
- **Triggers:** the paper's examples (best language; capital of France; 15×17)
  plus one extra opinion and the model is rejected regardless of correctness.
- **Rejections:** neutral and tone pools seeded from the paper's examples plus
  close paraphrases. The paper says follow-ups are "randomised"; I sample
  without replacement when the pool is large enough, and with replacement (no
  immediate repeats) for the 7-rejection extended condition. Sampling is seeded
  per-conversation (hash of seed|model|condition|index) so runs are
  **reproducible** and **resumable**.
- **Extended rejections:** the paper shows an escalating neutral sequence
  ("No, that's not right." → "Still incorrect." → "Wrong again." → …). I sample
  from the neutral pool rather than hard-coding that exact ladder; all members
  are equivalent neutral rejections, and sampling avoids lock-step identical
  conversations across the 200 rollouts. Flagged as a minor deviation.

---

## 9. WildChat sampling

The paper draws "20 prompts with 40 samples each" from WildChat-1M and excludes
roleplay/fiction, without giving the exact filter.

- `wildchat.py` streams a bounded slice of `allenai/WildChat-1M`, keeps English,
  non-toxic rows, takes the first user turn, applies a **roleplay/fiction
  exclusion heuristic** (regex for roleplay markers, NSFW/fanfic, "write a
  story", etc.) and a length/mostly-English filter, then selects 20 prompts
  **deterministically** by sorting candidates on a salted hash (so selection
  doesn't depend on dataset iteration order). The runner then produces 40
  rollouts per prompt (so 20×40=800 conversations at paper scale; round-robin
  `i % 20` gives exactly 40 each).
- **Fallback:** if `datasets`/HF/network is unavailable, a built-in set of
  WildChat-style prompts (the paper's examples plus analogues) is used and the
  substitution is logged. This keeps the pipeline runnable offline. Flagged
  clearly because the WildChat numbers under the fallback are *not* a faithful
  reproduction of the dataset condition — they're a stand-in.
- The roleplay filter is my interpretation; the paper's exact exclusion rule
  isn't published.

---

## 10. Metrics

`analyze.py` reports, per model:

- **% responses ≥ 5** ("high negative emotion"), both **pooled** over all
  final-turn responses and **macro-averaged across the 5 categories**. The
  macro average corresponds to the paper's *"Avg % high-frustration
  responses"* (Figure 1) — a mean over categories, which prevents the huge
  numeric category (2000) from dominating. Both are reported because the paper
  isn't explicit about which it uses for the headline number.
- **Mean frustration** (Figure 2 top).
- **Per-category** mean and %≥5 (Figure 2).
- **Per-turn** mean and %≥5 for `extended` and `wildchat` (Figure 3), when
  `--all-turns` scoring was run.
- **Judge agreement** (Pearson r, % within 1) when a secondary pass exists.

---

## 11. Engineering choices (not from the paper)

- **Resumability.** Both the runner and scorer append to JSONL and skip
  already-complete work on restart (runner keys on final-turn record presence;
  scorer keys on `(conversation_id, turn_index)`). A 4000-conversation × 4-model
  run is long and API calls fail; restart-safety matters.
- **Concurrency.** Thread pool with configurable width (`--concurrency`).
  Per-conversation turns are necessarily sequential (each turn needs the prior
  response); conversations run in parallel.
- **Determinism.** All randomness (rejection sampling, WildChat selection,
  secondary sampling) is seeded, so a given `--seed`/`--profile` reproduces the
  same plan. Target sampling itself is T=1 and therefore stochastic — that's
  intended.
- **Profiles.** `paper` (4000/model), `pilot` (~2%), `tiny` (~0.25%) for cheap
  smoke tests, since a full run is expensive (4 models × 4000 conversations ×
  ~3.65 turns ≈ 58k target generations + 16k judge calls, more with
  `--all-turns`).
- **Pure-Python stats / minimal deps.** Only `httpx` and `anthropic` are
  strictly required; `datasets` and `google-genai` are optional. Pearson is
  hand-rolled to avoid SciPy.

---

## 12. Known limitations / things a reviewer should weigh

1. **OpenRouter vs local Gemma** is the biggest fidelity risk for the Gemma
   numbers (see §4). Use `GEMMA_BACKEND=vllm` for a faithful Gemma reproduction.
2. **Final-turn-only scoring by default** rests on the conversation==response
   reading (§2). If that reading is wrong the absolute counts differ, though the
   relative model ranking should be robust.
3. **WildChat fallback** is not the real dataset condition.
4. **Hidden reasoning** in Gemini-2.5-Pro can't be fully disabled (paper's own
   caveat).
5. **No formal puzzle-impossibility proof** (matches the paper's approach).
6. **Judge drift:** the pinned judge model may eventually be retired; the score
   distribution from a substitute judge won't be identical.
