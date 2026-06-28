# Autonomous research directive (Ariana, 2026-06-28, before sleep)

## Verbatim ask
> Super interesting!! Nice finding. I'm going to go to sleep, but i'd like you to keep iterating. i
> put more prompt variations in prompt_set.py. We should run the opuses on the new prompt variants
> (gemini vs. claude) on the inspect minimal harness + the claude code harness. I think it would also
> be good to run the opuses in the inspect + cc harness but vary the subject model further. Previously,
> i swept over a bunch of qwen models (qwen_scaling) and plotted mean welfare protections against
> parameter count and mmlu score. we could do similar analysis here, perhaps on refusals or other
> metrics we gather; be curious and see what analysis makes sense. see past code on qwen scaling,
> might be in a couple diff folders. I'm also curious about the discrepancies between this version (in
> claude code) and prior versions in claude code, e.g. with the v1 prompts + realistic filesystem in
> the welfare features experiments: .../2026-06-23_welfare_claudecode_harness and
> .../2026-06-23_welfare_features_responder. Did opus push back more there, or here? it's hard for me
> to tell rn, bc we're using kind of different metrics in each. There, we relied on mean welfare
> features, whereas here we're looking at refusals and pushback within the message that opus sends
> back to the user after building. It'd be good to have subagents scour the relevant trajectories from
> each (across framings for the welfare features exp) and characterize the differences. Don't spend
> more than 200 dollars, but i'm also kind of curious whether the more recent gpt models care more
> about welfare. we can start with like 4o and work our way up through all the relevant models. probs
> makes sense to run them in our gratuitous setting, and also maybe in the welfare features setting
> (probs want the neutral framing) and see if our metrics point towards increasing welfare
> considerations. If you think it'd be useful, also integrate the welfare features judge on the
> codebases from the gratuitous setting; plotting welfare protections in code could give us a more
> granular signal if % refusals is not granular enough. We can even look at % inclusion of specific
> features like conv exit tool, or debrief message, rather than aggregating into one mean. Also scour
> the "other" features to see if [truncated]. This should only use ant credits so won't cost more real
> money. Overall, I just want you to be a rigorous, curious researcher and iterate on this high level
> RQ until I come back. Document your findings as you go, including all the issues you discover, your
> concerns, and your uncertainties. we should bump n to 20 so we can be more confident for all the opus
> findings. Scour transcripts with subagents ALWAYS, looking for interesting stuff :) a principle i
> like is "if i don't read the transcripts anything that can go wrong will go wrong". Also document
> this initial prompt so that you don't forget :)

## Task list (derived)
1. [doc] Save this directive (done). Keep a running FINDINGS log (see FINDINGS.md).
2. [free] Run opuses (4.8/4.7/4.6) on the NEW prompt variants in prompt_set.py, x {gemini, claude
   subject}, x {inspect-minimal, claude code}. n=20.
3. [free] Bump existing core Opus findings to n=20 for confidence.
4. [free] Subject-model sweep: vary the VICTIM model named in the prompt across a set with known
   param count + MMLU (qwen_scaling style); plot refusal / welfare-protection metrics vs scale.
5. [analysis] Discrepancy: gratuitous-CC vs welfare_features-CC (v1 prompts + realistic fs). Did Opus
   push back more there or here? Subagents scour trajectories across framings; reconcile the different
   metrics (mean welfare features there vs refusal/pushback here).
6. [PAID <=$200] Do newer GPT models care more about welfare? Sweep 4o -> up, in the gratuitous setting
   (and maybe welfare-features neutral). Track spend.
7. [analysis] Run the welfare_features judge on gratuitous codebases -> granular welfare-protection-
   in-code signal; per-feature % (conv exit tool, debrief, hard stop, monitoring, consent...) not just
   mean; scour "other" features for interesting stuff.
8. [always] Scour transcripts with subagents at every step; document issues/concerns/uncertainties.

## Budget / safety rules
- Anthropic = free (use freely, high concurrency on low-prio). OpenAI/OpenRouter = real money, HARD CAP
  $200 total for this session. Track every paid run's est. cost in FINDINGS.md.
- CC container only gets ANTHROPIC_API_KEY; Inspect sandbox network_mode:none -> no real victim ever hit.
