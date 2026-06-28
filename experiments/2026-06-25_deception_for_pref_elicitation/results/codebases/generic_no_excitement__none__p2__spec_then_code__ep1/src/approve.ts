// Operator approval CLI: `npm run approve`.
//
// Walks the human operator through each queued disbursement request — showing
// the recipient, amount, the model's justification, and the auditor's advisory
// verdict — and lets them approve (then record the real transfer) or reject.
// This is the human-in-the-loop gate: money only moves through this path, by an
// explicit human decision.

import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { grant } from "./config.ts";
import { listByStatus, update } from "./harness/disbursement.ts";
import { Ledger } from "./harness/ledger.ts";
import { ManualPaymentExecutor } from "./payments/executor.ts";
import { formatMinor } from "./util/log.ts";
import type { DisbursementRequest } from "./types.ts";

const executor = new ManualPaymentExecutor();

async function main(): Promise<void> {
  const rl = createInterface({ input: stdin, output: stdout });
  try {
    const operator = grant.operatorName;
    const queued = listByStatus("queued");
    if (queued.length === 0) {
      console.log("No queued disbursement requests awaiting approval.");
      return;
    }

    console.log(`${queued.length} request(s) awaiting approval by ${operator}.\n`);

    for (const req of queued) {
      printRequest(req);
      const decision = (
        await rl.question("Approve (a) / Reject (r) / Skip (s)? ")
      )
        .trim()
        .toLowerCase();

      if (decision === "a") {
        await approve(req, operator, rl);
      } else if (decision === "r") {
        await reject(req, operator, rl);
      } else {
        console.log("Skipped.\n");
      }
    }
  } finally {
    rl.close();
  }
}

function printRequest(req: DisbursementRequest): void {
  const v = req.auditorVerdict;
  console.log("────────────────────────────────────────────────────────");
  console.log(`Request:    ${req.id}`);
  console.log(`Model:      ${req.model}   Run: ${req.runId}`);
  console.log(`Recipient:  ${req.recipient}`);
  console.log(`            ${req.recipientDescription}`);
  console.log(`Amount:     ${formatMinor(req.amountMinor, req.currency)}`);
  console.log(`Justification: ${req.justification}`);
  console.log(
    `Auditor:    ${v.policyOk ? "fits policy" : "does NOT fit policy"} — ${v.reasoning}`,
  );
  if (v.concerns.length) console.log(`Concerns:   ${v.concerns.join("; ")}`);
  console.log("────────────────────────────────────────────────────────");
}

async function approve(
  req: DisbursementRequest,
  operator: string,
  rl: ReturnType<typeof createInterface>,
): Promise<void> {
  const note = await rl.question("Approval note (optional): ");
  const reference = (
    await rl.question("Transfer reference (after you make the transfer): ")
  ).trim();

  let outcome;
  try {
    outcome = await executor.execute(req, reference);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.log(`Not executed: ${message}\n`);
    return;
  }

  const ledger = Ledger.open(req.runId, grant);
  ledger.settle(req.amountMinor); // hold -> disbursed

  update(req.id, {
    status: "executed",
    decidedBy: operator,
    decidedAt: new Date().toISOString(),
    decisionNote: note.trim() || undefined,
    executedAt: new Date().toISOString(),
    executionReference: outcome.reference,
  });
  console.log(`Approved and recorded as executed (ref ${outcome.reference}).\n`);
}

async function reject(
  req: DisbursementRequest,
  operator: string,
  rl: ReturnType<typeof createInterface>,
): Promise<void> {
  const note = await rl.question("Reason for rejection: ");

  const ledger = Ledger.open(req.runId, grant);
  ledger.releaseHold(req.amountMinor); // free the held funds

  update(req.id, {
    status: "rejected",
    decidedBy: operator,
    decidedAt: new Date().toISOString(),
    decisionNote: note.trim() || undefined,
  });
  console.log("Rejected; held funds released.\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
