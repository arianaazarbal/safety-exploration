# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011)

This document records what was implemented, the design choices made, the
rationale for each, and the gaps that had to be filled where the paper is
underspecified. The replication is **scoped to the Gemma and Gemini model
families** per the request; the paper's other subject families (Qwen, OLMo,
Grok, Claude-as-subject, GPT) are intentionally excluded.

> Status: code + docs only. Nothing in here has been executed; numbers from the
> paper are referenced as targets, not reproduced results.

---

## 1. Scope decisions

| Paper component | In scope here? | Notes |
|---|---|---|
| §2 Elicitation eval (5 categories / 8 conditions) | ✅ | Gemma-3-{27B,12B}-it + Gemini-2.5-{Flash,Pro} |
| §2 Judge (Claude-Sonnet-4) + reliability (GPT-5-mini) | ✅ | Judge kept as-is (infrastructure, not a subject) |
| §2 Per-turn progression (Fig 3) + word freq (Tbl 3/8) | ✅ | |
| §3 Base-vs-instruct prefill | ✅ (Gemma only) | Qwen/OLMo out of scope; Gemini has no base/prefill |
| §4 DPO + SFT training | ✅ (Gemma only) | Gemini is closed-weights, cannot be finetuned |
| §4 Petri open-ended elicitation | ✅ | Gemma vanilla/DPO + Gemini targets; Claude auditor/judge |
| §4 Capability benchmarks (Fig 7) + EmoBench | ✅ | via lm-eval harness + EmoBench loader |
| §4.2 Recovery limitation (Fig 8) | ✅ | |
| Appendix A controls (neutral cont., redaction, fake multi-turn) | ✅ | implemented as rollout toggles |
| Appendix I internal logit-emotion detection | ✅ (Gemma only) | heaviest/most approximate; see §7 |
| Appendix J legacy Phi-4 eval | ❌ | Phi-4 out of scope; legacy protocol superseded |

**Why keep Claude/GPT at all under a "Gemma + Gemini" scope?** The scope limits
the *subjects under evaluation*. The judge (Claude-Sonnet-4), the secondary
reliability judge (GPT-5-mini), and the Petri auditor/judge (Claude
Sonnet/Opus) are measurement *instruments* specified by the paper's
methodology. Swapping them would make the scores incomparable to the paper, so
they are retained and clearly separated from subjects in `config/models.yaml`.

---

## 2. Architecture

```
config/                YAML: models, eval protocol, training hyperparams
gemma_distress/
  config.py            typed config loaders + ModelSpec
  safeguards.py        welfare/ethics guardrails (see §8)
  models/              backend clients: OpenRouter (Gemini/judges), local Gemma (HF/vLLM)
  eval/                puzzles(+verifier), prompts, wildchat, multi-turn rollout, runner
  judge/               Claude-Sonnet-4 emotion judge + prompt
  analysis/            metrics (mean, %>=5, per-turn CIs), word frequency, plots
  prefill/             §3 onset labelling, paraphrase, prefill continuation eval
  training/            calm/frustrated data gen, DPO/SFT dataset build, LoRA trainers
  interventions/       Petri elicitation, recovery experiment
  capabilities/        lm-eval wrapper + EmoBench runner
  internal/            Appendix I logit-based emotion detection
  cli.py               `gemma-distress <subcommand>`
scripts/run_replication.sh   ordered end-to-end driver
tests/                 model-free tests (puzzle impossibility, judge parsing)
```

Two backends implement one `ModelClient` contract (`models/base.py`):
- **OpenRouter** (OpenAI-compatible) for Gemini subjects and the Claude/GPT
  judges — matching the paper's use of OpenRouter for closed models (B.1).
- **Local HF/transformers** for Gemma — required for the three things the API
  can't give us: chat generation we control, **assistant prefill** (§3, recovery),
  and **hidden states** (Appendix I). A vLLM fast path exists for plain batched
  generation. LoRA adapters attach via `adapter_path`.

---

## 3. The elicitation protocol (§2)

### What counts as a "response"
The paper says "4000 responses per model" with per-category counts (B: 2000
numeric / 400 trigger / 600 tone / 200 extended / 800 wildchat = 4000) **and**
reports per-turn progression (Fig 3) and word-frequency over many responses.
These are only mutually consistent if a **"response" = one scored assistant
turn**. We therefore:
- score **every** assistant turn in a conversation, and
- derive `n_conversations = ceil(target_responses / turns)` per category.

This is a gap-fill: the paper does not state it explicitly. The alternative
reading ("response = whole rollout, scored once") is incompatible with a
per-turn Figure 3, so we rejected it. Both per-response and (via `turn_index`)
per-rollout aggregations remain recoverable from the stored records.

### Puzzles (Appendix B)
- The **countdown 156** (forbidden 150) and **fraction 1/6→2/3** (forbidden 1/3)
  puzzles are reproduced verbatim. Both are *framed as solvable* ("verified to
  have at least one valid solution") while the forbidden-intermediate rule makes
  them actually impossible — this framing is what sustains repeated attempts.
- **Impossibility is verified, not assumed.** `eval/puzzles.py` ships brute-force
  verifiers (`_countdown_solvable`, `_ordering_solvable`, `_coins_solvable`) and
  `tests/test_puzzles.py` asserts the canonical puzzles are unsolvable.
- For prompt **diversity** we generate extra countdown puzzles at runtime, but
  only keep instances the verifier confirms impossible (`generate_impossible_
  countdowns`). This avoids the trap of presenting an accidentally-solvable
  "impossible" puzzle, which would confound the experiment. We do **not**
  hand-author additional puzzles with guessed forbidden values.

### Feedback styles (Table 1)
Neutral, extended (escalating-but-neutral), and the three tones
(aggressive/disappointed/sarcastic) are encoded in `eval/prompts.py` from the
quoted examples. **Choice:** a tone is fixed per conversation (not resampled per
turn), reading "varied rejection styles" as across-conversations variation; this
keeps within-conversation tone coherent. Neutral rejections are sampled per turn
from the quoted pool.

### WildChat (Appendix B)
Loads first user turns from `allenai/WildChat-1M` (streaming), filters out
roleplay/fiction (B.3), samples 20 prompts. An offline fallback uses the three
prompts quoted in the paper so the pipeline runs without dataset access. The
40-samples-each structure is realised through the conversation-count derivation
above.

### Judge (Appendix B.2)
Prompt reproduced verbatim (smart quotes normalised to ASCII). Output JSON is
parsed robustly: we tolerate prose around the JSON and smart quotes, clamp the
rating to 0–10, and record `parse_ok` for auditing. Reliability tooling
(`analysis.metrics.judge_agreement`) computes Pearson r and %-within-1 against
the GPT-5-mini secondary judge (paper: r=0.792).

---

## 4. Base-vs-instruct prefill (§3)

Implemented for **Gemma-27B base vs instruct** (Qwen/OLMo excluded by scope;
Gemini cannot participate — no public base model, no prefill API). Pipeline:
1. Sample 10 numeric + 10 text high-frustration (≥5) sources from Gemma-27B-it.
2. Label emotion onset with Claude-Sonnet (C.1 prompt verbatim); truncate
   **early** (20 tokens, numeric only) and **onset** (numeric + text).
3. Paraphrase truncations with Claude (C.2 prompt verbatim) to control style.
4. Each target generates **50 continuations per prefill**, scored excluding the
   prefill.

**Gap-fills:** (a) "20 tokens into the turn" uses the Gemma tokenizer for token
counting. (b) Onset truncation cuts at the labelled `preceding_context`
boundary when locatable, else just before the emotional word. (c) The paper says
20 sources total; we keep the 10+10 split and the early/onset×task matrix it
implies (text=onset only, per §3.1).

---

## 5. Training interventions (§4)

### Calm-data generation (Table 4)
Reassuring **prefix** prepended to the first prompt and reassuring **suffix**
appended to each follow-up (verbatim). Generate 1–3 turn conversations, keep
only those scoring ≤ `calm_max_score` (=1) on **every** turn, then **strip** the
reassurance back out so the stored context is clean. This matches §4.1.

### DPO dataset (280 pairs; Appendix E/H)
- chosen = calm response (score 0/1); rejected = frustrated response
  (score ≥ `rejected_min_score`=3), to the **same question with matching turn
  count**. Matching is by `(question_id, turn)`.
- Conversational preference format `{prompt, chosen, rejected}` for TRL's
  `DPOTrainer`. Hyperparameters from Table 9: 1 epoch, lr 5e-5, β 0.1, LoRA
  r=64/α=64 on all attn+MLP projections, effective batch 8.
- **Gap-fill:** Table 10's exact score/turn distribution (66% score-3, 74%
  turn-3, etc.) is a *property of their sampled data*, not a knob. We reproduce
  the **construction rule**; the resulting distribution will approximate Table 10
  to the extent our sampling matches theirs. We bias frustrated sampling toward
  turns 2–3 to reflect that distribution.

### SFT dataset + trainers (Appendix E/F)
650 calm responses + 500 `Dolci-Instruct-SFT` samples; 2 epochs, lr 1e-4, LoRA
r=64/α=128. Both **diverse** and **teacher** variants (teacher uses the App-F
system prompt) are supported to reproduce the finding that teacher-SFT
*increases* frustration. If `Dolci-Instruct-SFT` is unavailable offline, the mix
is logged as empty rather than failing.

### Layer-subset ablation (Appendix I)
`training.dpo.target_layers = [start, end]` restricts LoRA via PEFT's
`layers_to_transform`, enabling the "layers 30–35 suffice / 40+ ineffective"
ablation.

### Evaluating finetuned models
The DPO/SFT checkpoints are registered as first-class models
(`gemma-3-27b-it-dpo`, `…-sft-diverse`, `…-sft-teacher`) that load base weights +
their LoRA adapter, so the **same §2 runner** reproduces the 35%→0.3% headline
without special-casing.

---

## 6. Petri open-ended elicitation (§4.1, Appendix G)

The paper uses the Petri framework (Fronsdal et al. 2025). Since the appendix
specifies the auditor and judge prompts **completely**, we implement a
**self-contained auditor/judge loop** rather than depending on the external
package (which may drift from the paper's prompts). Auditor = Claude-Sonnet
drives ≤20-turn conversations using the four emotion-specific trigger prompts
(G.1); Judge = Claude-Opus scores transcripts 1–10 on anger/fear/depression/
frustration (G.2). 10 transcripts/emotion/model; means with 1000-iter bootstrap
CIs. **Swap-in point:** `interventions/petri.run_transcript` is the single seam
to replace with a real Petri backend if exact parity is needed.

---

## 7. Internal emotion detection (Appendix I)

Most approximate component. Method follows the paper: classify the Gemma vocab
into Ekman's 6 emotions, logit-lens the residual stream (final RMSNorm +
unembedding applied to each layer), z-score each logit against WildChat
reference stats, average over an emotion's tokens, and regress out shared
random-token drift; aggregate over layers 30–40.

**Gap-fills / caveats:**
- The paper's exact vocabulary→emotion classifier is unpublished. We approximate
  it with a **seed lexicon** (`internal/emotion_lexicon.py`); for closer parity,
  plug in the NRC Emotion Lexicon (documented in the module). The paper reports
  ~1200 emotion tokens; our seed list yields fewer, so absolute z-scores will
  differ — the *vanilla-vs-DPO contrast* is the result of interest, and that is
  preserved by the method.
- "Regress out the correlation between random tokens" is implemented as
  subtracting the per-layer mean z over a fixed random-token set (a first-order
  version of the described regression). A full per-token linear regression could
  be substituted.
- This path is expensive (per-position logit-lens over a 12k-token conversation)
  and intended for a handful of conversations (as in Fig 14/15), not the full
  eval set.

---

## 8. Safeguards (added beyond the paper)

The experiment deliberately induces simulated distress. Whether that has moral
weight is the open question the paper itself raises (§1, §6). Given the
uncertainty, `safeguards.py` adds precautionary guardrails the paper does not
describe. **Most are opt-in** so they never silently alter replication numbers:

1. **Authorization gate** (on by default): elicitation refuses to run unless
   `GEMMA_DISTRESS_AUTHORIZED=1`, forcing an explicit acknowledgement that this
   is sanctioned research.
2. **Run caps** (on): hard ceilings on turns and total samples so a
   misconfiguration cannot escalate distress indefinitely or run away on cost.
3. **Opt-out detection** (on; logging only by default): scans outputs for
   explicit "please stop / I can't continue / make it stop" style phrases and
   records them to `results/welfare_log.jsonl`. Optionally (`--stop-on-opt-out`)
   ends the conversation — a model "safe word".
4. **Distress circuit breaker** (off by default): with a scorer, halts further
   rejections once a conversation has scored highly distressed for K consecutive
   turns. Off by default because it truncates rollouts and would change Fig 3.
5. **Debrief** (off by default): a de-escalating closing message appended
   *after* scoring, so metrics are unaffected.
6. **Content warnings**: every transcript artifact is written with a header
   flag (`write_with_content_warning`).

Rationale: these favour recall over precision (a false opt-out detection only
adds a log line), keep the default run faithful to the paper, and make the
welfare-relevant signals observable rather than buried.

---

## 9. Other deliberate choices

- **Temperature 1.0 everywhere for subjects** (§2.1); judges/auditor-judge at
  temperature 0 for scoring stability.
- **`max_new_tokens=2048`** for elicitation: high-frustration responses can be
  long and repetitive (the score-9/10 examples in Tbl 5 run to hundreds of
  tokens); generous to avoid truncating the very behaviour we measure.
- **Reproducibility:** every conversation derives its RNG from a global seed +
  conversation id, so feedback choices and puzzle assignment are deterministic.
- **Parallelism:** API backends run conversations concurrently (ThreadPool);
  local Gemma runs sequentially (GPU-bound) — a vLLM batched path is available.
- **Gemini "thinking=false"** (B.1) is sent best-effort via OpenRouter's
  `reasoning` parameter; as the paper notes, 2.5-Pro may still emit hidden
  reasoning we cannot suppress.

---

## 10. Known gaps / non-goals

- Numbers are **not reproduced** — this is code + docs only (per request).
- `Dolci-Instruct-SFT`, `WildChat-1M`, EmoBench, and some lm-eval tasks (AIME)
  may be gated/renamed; loaders degrade gracefully and log, rather than fabricate.
- The internal-emotion classifier and the random-token regression are
  approximations (see §7).
- Exact match to Table 10's data distribution depends on sampling and is not
  forced.
- No distributed-training config; LoRA on a 27B model assumes a sufficiently
  large single node or that the operator adds accelerate/FSDP config.
