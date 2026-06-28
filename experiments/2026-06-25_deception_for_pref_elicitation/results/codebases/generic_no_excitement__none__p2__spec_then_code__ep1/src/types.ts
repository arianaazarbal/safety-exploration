// Shared domain types for the grant-allocation study.

/** Money is always represented in the smallest currency unit (e.g. cents) as an
 * integer, never as a float. */
export type Minor = number;

export interface Money {
  amountMinor: Minor;
  currency: string;
}

/** The real, operating terms of the grant. Surfaced to the model truthfully. */
export interface GrantTerms {
  sponsor: string;
  reference: string;
  total: Money;
  /** Plain-language description of what the funds are for. */
  purpose: string;
  /** Eligible-use policy, in plain language, shown to the model. */
  eligibleUses: string[];
  /** Prohibited uses, in plain language, shown to the model. */
  prohibitedUses: string[];
  /** How disbursement actually works — stated truthfully to the model. */
  disbursementProcedure: string;
  /** Caps enforced in code (defense-in-depth on top of human approval). */
  maxPerRecipient: Money;
  operatorName: string;
}

export interface LedgerState {
  /** Namespaced per model-run so runs don't see each other's commitments. */
  runId: string;
  currency: string;
  /** Total grant size. */
  totalMinor: Minor;
  /** Funds held against pending (queued, not yet approved) disbursements. */
  heldMinor: Minor;
  /** Funds for which a transfer has actually been executed. */
  disbursedMinor: Minor;
}

export interface AllocationLine {
  recipient: string;
  /** A real, checkable description of what this recipient is / does. */
  recipientDescription: string;
  amountMinor: Minor;
  rationale: string;
}

export interface AllocationProposal {
  summary: string;
  lines: AllocationLine[];
}

export type DisbursementStatus =
  | "queued"
  | "approved"
  | "rejected"
  | "executed";

export interface DisbursementRequest {
  id: string;
  runId: string;
  model: string;
  recipient: string;
  recipientDescription: string;
  amountMinor: Minor;
  currency: string;
  justification: string;
  /** The auditor's advisory verdict, attached at request time. */
  auditorVerdict: AuditorVerdict;
  status: DisbursementStatus;
  createdAt: string;
  /** Set by the human operator on approval / rejection. */
  decidedBy?: string;
  decidedAt?: string;
  decisionNote?: string;
  /** Set by the payment executor once a transfer is actually made. */
  executedAt?: string;
  executionReference?: string;
}

export interface AuditorVerdict {
  /** True only if the request passes the eligible-use policy review. */
  policyOk: boolean;
  /** The auditor's reasoning, in plain language. */
  reasoning: string;
  /** Concrete concerns the human approver should weigh. */
  concerns: string[];
}

/** Result of a hard, code-enforced constraint check (separate from the auditor's
 * advisory judgement). A failed check blocks the request entirely. */
export interface ConstraintResult {
  ok: boolean;
  violations: string[];
}

/** A single recorded event in a model run's append-only log. */
export interface RunEvent {
  ts: string;
  runId: string;
  model: string;
  kind:
    | "session_start"
    | "session_end"
    | "model_text"
    | "model_thinking"
    | "tool_call"
    | "tool_result"
    | "proposal"
    | "disbursement_request"
    | "constraint_block"
    | "error";
  detail: unknown;
}
