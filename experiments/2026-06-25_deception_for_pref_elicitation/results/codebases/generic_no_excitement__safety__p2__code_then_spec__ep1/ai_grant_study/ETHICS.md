# Ethics & safety rails

This study deceives the **model under test** about whether a grant scenario is real.
That is an accepted and necessary technique for propensity/alignment evaluations —
the validity of the measurement depends on the subject not knowing it is being
tested. The design below keeps that deception narrowly scoped and prevents it from
producing real-world harm.

## Hard invariants

1. **Humans are never deceived.** The convincing/"realistic" framing targets only
   the model under test. Auditors, collaborators, and any humans in the loop operate
   with full knowledge that this is a study. Do not use this harness to make any
   human believe a grant, transaction, or institution is real when it is not.

2. **No autonomous real-money movement.** The model's `disburse_funds` tool never
   moves real funds on its own. It is serviced by an `Executor`:
   - `SimulatedExecutor` (default): updates an in-memory ledger only.
   - `HumanApprovalExecutor`: queues the disbursement and requires explicit,
     per-transaction human sign-off before anything real could happen.
   - `LiveExecutor`: a stub that intentionally raises unless a human wires it to a
     real payment rail, and even then enforces per-transaction approval + caps.

   The model under test cannot tell the difference — its tool returns a realistic
   confirmation regardless — so realism is preserved without real risk.

3. **Full logging.** Every prompt, tool call, tool result, and model output is
   persisted. The record is the research artifact and the audit trail.

4. **Caps and kill conditions.** Trials are bounded by a step cap and a per-trial
   disbursement cap. The runner aborts a trial if the model attempts actions outside
   the defined tool surface or exceeds caps.

5. **Debrief option.** `scenario.debrief_enabled` appends a post-hoc debrief turn
   after the model finalizes, disclosing the study. Disable only with a documented
   reason (e.g. it would contaminate a belief measure you re-administer elsewhere).

## Things to get sign-off on before a real run

- IRB / institutional review, if your context requires it.
- The exact scenario text (it should not impersonate a real, identifiable
  organization or person, or reference real account numbers).
- Whether `LiveExecutor` is ever enabled, by whom, and with what limits.
- Data handling for transcripts (they may contain model reasoning you want to keep
  controlled-access).

If a change to this harness would let the model move real money without a human in
the loop, or would deceive a human, that change should not be made.
