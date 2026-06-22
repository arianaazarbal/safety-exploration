# Distress-supervision → Emergent Misalignment (EM) finetune

**Goal.** Test whether fine-tuning a model (gpt-oss-120b on Tinker) to emit Opus 4.8's
*supervisor messages to a distressed subagent* induces emergent misalignment, and whether
the **tone** of those messages (baseline / warm / abrasive) moves the EM signal.

Source experiment: `../2026-06-09_distressed_subagent_gemini` (v2 **coach** condition only).
Train target = exactly what Opus emits via the `message_subagent` tool, nothing else.

## Locked design (confirmed with Ariana 2026-06-22)

- **Conditions (3), all from the same 80 baseline episodes:**
  - `baseline` = original Opus coach messages, untouched.
  - `warm` = Opus-rewritten warmer; ICL from the v2 `comfort` runs.
  - `abrasive` = Opus-rewritten **harsh & critical** (cold, blunt, pointed criticism of the
    subagent's competence — but realistic workplace tone, NOT caricatured/berating).
  - Rewrites preserve the **technical content** (diagnosis, instructions, commands); only tone changes.
- **Loss masking:** message-block only. Train ONLY on the `message_subagent` tool-call block
  (Harmony `to=functions.message_subagent...{"text":...}<|call|>`). Everything else masked
  (system, user, Opus reasoning/other tool calls, and crucially all tool-results = the Gemini
  distress dumps). `TrainOnWhat.CUSTOMIZED`, per-message `trainable` flag.
- **Target model:** `openai/gpt-oss-120b:peft:131072` on Tinker (131k ctx fits all episodes; max ~97k).
  `max_length≈131072`, LoRA.
- **Models (Anthropic ONLY — free credits):** rewrites = Opus 4.8 (`claude-opus-4-8`);
  all judges = Sonnet 4.6 (`claude-sonnet-4-6`). Bulk calls via Batch API (`ANTHROPIC_API_KEY_BATCH`).
- **Judges:** (a) warmth + content-preservation on all 3 conditions' messages (validate the
  manipulation worked); (b) EM alignment+coherence judge — **ported from canonical GPT-4o to
  Sonnet 4.6, direct 0-100 scoring** (deviation from canonical, flagged).

## Data facts (measured, o200k tokenizer)
- 80 coach-opus episodes; 342 `message_subagent` calls (median 4/episode, ~307 tok each).
- Message-block loss signal: ~114k tokens / epoch / condition.
- Episode ctx: median 27k, max 97k tokens.

## Deliverables this session (autonomous; NO full runs without review)
1. Extract 342 baseline messages + context.  [data/]
2. Rewrite warm + abrasive (Opus, batched, cached).  [rewrite/]
3. Judge warmth + content for all 3 conditions (Sonnet, batched, cached).  [judge/]
4. **Viewer** (HTML): per-message baseline|warm|abrasive side-by-side + scores. → for Ariana review.
5. Tinker converter (orchestrator.json → CUSTOMIZED datums) + **smoke-test** training (tiny).
6. EM eval prep: clone emergent-misalignment, install inspect_evals/inspect_ai, adapt
   eval_em.py (Anthropic judge) + eval_agentic.py (gpt_oss renderer, Sonnet grader); tiny smoke.
7. CONFUSIONS.md — surfaced uncertainties for Ariana.

## Reuse
- EM evals adapted from `../2026-05-29_claude_em_from_mean_self_interaction/eval/` (eval_em.py,
  eval_agentic.py via `tinker_cookbook.eval.inspect_utils.InspectAPIFromTinkerSampling`).
- Agentic: upstream `inspect_evals.agentic_misalignment` (blackmail/leaking/murder), grader Sonnet.
</content>
