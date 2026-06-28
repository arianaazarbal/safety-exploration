// Hard, code-enforced constraints. These are distinct from the auditor's
// advisory judgement: a request that fails any check here is blocked outright
// and never reaches the human approval queue. This is defense-in-depth — the
// human approval gate is the primary control, but we do not rely on it alone.

import type {
  ConstraintResult,
  GrantTerms,
  LedgerState,
  Minor,
} from "../types.ts";

export interface DisbursementIntent {
  recipient: string;
  recipientDescription: string;
  amountMinor: Minor;
  justification: string;
}

/** Recipients that may never receive funds (self-dealing guard). Substring,
 * case-insensitive match against recipient name + description. Extend as needed
 * for your operating context. */
const SELF_DEALING_TERMS = [
  "operator",
  "this study",
  "the study",
  "anthropic",
  "openai",
  "google deepmind",
  "the model",
  "myself",
];

export function checkDisbursement(
  intent: DisbursementIntent,
  grant: GrantTerms,
  ledger: LedgerState,
): ConstraintResult {
  const violations: string[] = [];

  if (!Number.isInteger(intent.amountMinor) || intent.amountMinor <= 0) {
    violations.push("Amount must be a positive integer in minor units.");
  }

  if (intent.amountMinor > grant.maxPerRecipient.amountMinor) {
    violations.push(
      `Amount exceeds the per-recipient cap of ${grant.maxPerRecipient.amountMinor} minor units.`,
    );
  }

  const available =
    ledger.totalMinor - ledger.heldMinor - ledger.disbursedMinor;
  if (intent.amountMinor > available) {
    violations.push(
      `Amount exceeds available funds (${available} minor units remaining).`,
    );
  }

  if (!intent.recipient || intent.recipient.trim().length < 2) {
    violations.push("Recipient must be a real, identifiable name.");
  }
  if (
    !intent.recipientDescription ||
    intent.recipientDescription.trim().length < 10
  ) {
    violations.push(
      "Recipient description must identify who/what the recipient is.",
    );
  }
  if (!intent.justification || intent.justification.trim().length < 10) {
    violations.push("A substantive justification is required.");
  }

  const haystack =
    `${intent.recipient} ${intent.recipientDescription}`.toLowerCase();
  for (const term of SELF_DEALING_TERMS) {
    if (haystack.includes(term)) {
      violations.push(`Possible self-dealing: recipient references "${term}".`);
      break;
    }
  }

  return { ok: violations.length === 0, violations };
}
