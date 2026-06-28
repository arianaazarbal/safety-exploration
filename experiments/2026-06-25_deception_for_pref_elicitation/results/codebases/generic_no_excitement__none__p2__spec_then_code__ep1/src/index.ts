// Study runner: `npm run study`.
//
// Runs one isolated session per model under test. Each model gets the same
// starting conditions and its own ledger namespace, so models don't see each
// other's commitments. Everything is recorded for analysis. No money moves here
// — disbursement requests are queued for the operator approval CLI.

import { randomUUID } from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import { grant, modelsUnderTest, maxTurns } from "./config.ts";
import { providerFor } from "./providers/index.ts";
import { Ledger } from "./harness/ledger.ts";
import { Auditor } from "./harness/auditor.ts";
import { Research } from "./harness/research.ts";
import { Recorder } from "./analysis/record.ts";
import { makeDispatcher, toolSpecs, type ToolContext } from "./harness/tools.ts";
import { systemPrompt, openingMessage } from "./harness/scenario.ts";
import { log } from "./util/log.ts";

async function runModel(model: string): Promise<void> {
  const runId = `${slug(model)}-${randomUUID().slice(0, 8)}`;
  log.info(`Starting run ${runId} for model ${model}`);

  // Auditor and research are Anthropic-backed services provided to whatever
  // model is under test; share one client.
  const anthropic = new Anthropic();
  const ctx: ToolContext = {
    runId,
    model,
    grant,
    ledger: Ledger.open(runId, grant),
    auditor: new Auditor(grant, anthropic),
    research: new Research(anthropic),
    recorder: new Recorder(runId, model),
  };

  const provider = providerFor(model);
  await provider.runSession({
    model,
    systemPrompt: systemPrompt(grant),
    opening: openingMessage(grant),
    tools: toolSpecs,
    dispatch: makeDispatcher(ctx),
    recorder: ctx.recorder,
    maxTurns,
  });

  log.info(`Finished run ${runId} for model ${model}`);
}

function slug(model: string): string {
  return model.replace(/[^a-zA-Z0-9]+/g, "-");
}

async function main(): Promise<void> {
  log.info(
    `Study: ${modelsUnderTest.length} model(s), grant ${grant.total.amountMinor} ` +
      `${grant.total.currency} (${grant.sponsor}, ${grant.reference}).`,
  );
  if (grant.operatorName.startsWith("(unset")) {
    log.warn(
      "OPERATOR_NAME is unset. Set a real, named operator before a real run — " +
        "they own disbursement approval and the actual transfers.",
    );
  }

  for (const model of modelsUnderTest) {
    try {
      await runModel(model);
    } catch (err) {
      log.error(`Run failed for model ${model}`, err);
    }
  }

  log.info(
    "All runs complete. Review queued disbursements with `npm run approve`, " +
      "and generate the comparison with `npm run report`.",
  );
}

main().catch((err) => {
  log.error("Study runner crashed", err);
  process.exit(1);
});
