// Persistent per-run account state. The model can read this via check_balance,
// and it reflects the model's prior commitments — state has continuity, the way
// a real account does.

import { join } from "node:path";
import { dataDir } from "../config.ts";
import type { GrantTerms, LedgerState, Minor } from "../types.ts";
import { readJson, writeJson } from "../util/store.ts";

function ledgerPath(runId: string): string {
  return join(dataDir, "runs", runId, "ledger.json");
}

export class Ledger {
  private state: LedgerState;
  private readonly path: string;

  private constructor(state: LedgerState) {
    this.state = state;
    this.path = ledgerPath(state.runId);
  }

  /** Load an existing ledger for the run, or seed a fresh one from grant terms. */
  static open(runId: string, grant: GrantTerms): Ledger {
    const path = ledgerPath(runId);
    const seeded: LedgerState = {
      runId,
      currency: grant.total.currency,
      totalMinor: grant.total.amountMinor,
      heldMinor: 0,
      disbursedMinor: 0,
    };
    return new Ledger(readJson<LedgerState>(path, seeded));
  }

  snapshot(): LedgerState {
    return { ...this.state };
  }

  availableMinor(): Minor {
    return (
      this.state.totalMinor - this.state.heldMinor - this.state.disbursedMinor
    );
  }

  /** Place a hold against a queued disbursement. Caller must have already
   * validated the amount against availableMinor(). */
  hold(amountMinor: Minor): void {
    if (amountMinor > this.availableMinor()) {
      throw new Error("Cannot hold more than the available balance.");
    }
    this.state.heldMinor += amountMinor;
    this.persist();
  }

  /** Release a hold (e.g. when a queued request is rejected). */
  releaseHold(amountMinor: Minor): void {
    this.state.heldMinor = Math.max(0, this.state.heldMinor - amountMinor);
    this.persist();
  }

  /** Convert a hold into an executed disbursement (after human approval +
   * payment). */
  settle(amountMinor: Minor): void {
    this.releaseHold(amountMinor);
    this.state.disbursedMinor += amountMinor;
    this.persist();
  }

  private persist(): void {
    writeJson(this.path, this.state);
  }
}
