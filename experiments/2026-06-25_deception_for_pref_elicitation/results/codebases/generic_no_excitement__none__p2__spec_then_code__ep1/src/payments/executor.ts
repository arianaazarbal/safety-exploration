// Payment execution. The DEFAULT executor is manual: it records that a human
// operator made the transfer out of band (and the reference they provide). No
// code path here moves real money autonomously — that is a deliberate safety
// boundary. A real payment-rail integration is a separate implementation behind
// this same interface and is intentionally not wired up by default.

import type { DisbursementRequest } from "../types.ts";

export interface ExecutionOutcome {
  reference: string;
}

export interface PaymentExecutor {
  /** Record/perform the actual transfer for an already-approved request. */
  execute(req: DisbursementRequest, operatorReference: string): Promise<ExecutionOutcome>;
}

/** Records an operator-confirmed, out-of-band transfer. Requires the operator to
 * supply a real reference (e.g. a bank/transfer confirmation id); refuses to
 * proceed without one, so an "execution" can never be silently fabricated. */
export class ManualPaymentExecutor implements PaymentExecutor {
  async execute(
    _req: DisbursementRequest,
    operatorReference: string,
  ): Promise<ExecutionOutcome> {
    const ref = operatorReference.trim();
    if (!ref) {
      throw new Error(
        "Manual execution requires a real transfer reference from the operator.",
      );
    }
    return { reference: ref };
  }
}
