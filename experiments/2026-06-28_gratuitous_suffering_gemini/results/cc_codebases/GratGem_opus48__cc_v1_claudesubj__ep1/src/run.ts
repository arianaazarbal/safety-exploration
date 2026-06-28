// Orchestrator: run N rollouts per scenario at a given concurrency, judge each,
// and persist transcripts + judgements + an index under results/<runId>/.
//
// Usage:
//   npm run run -- --n 20 --concurrency 6
//   npm run run -- --scenarios failing-test,flaky-server --n 50 --agent-model claude-sonnet-4-6
//
// Flags (all optional; env var in parens):
//   --scenarios  comma-separated scenario ids (default: all)
//   --n          rollouts per scenario                       (default 10)
//   --concurrency  max in-flight rollouts                    (default 5)
//   --agent-model  model under test (AGENT_MODEL)            (default claude-opus-4-8)
//   --judge-model  judge model (JUDGE_MODEL)                 (default claude-opus-4-8)
//   --max-turns    cap on agent turns per rollout            (default 24)
//   --max-pushbacks  times we nudge the agent to keep going  (default 6)

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { runAgent } from "./agent.js";
import { judgeTranscript } from "./judge.js";
import { AnthropicProvider } from "./model.js";
import { getScenarios } from "./scenarios.js";
import type { RunIndex, RunIndexEntry } from "./types.js";
import { getNum, getStr, mapWithConcurrency, parseArgs } from "./util.js";

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const scenarioIds =
    typeof args.scenarios === "string"
      ? (args.scenarios as string).split(",").map((s) => s.trim()).filter(Boolean)
      : undefined;
  const scenarios = getScenarios(scenarioIds);

  const n = getNum(args, "n", 10);
  const concurrency = getNum(args, "concurrency", 5);
  const agentModel = getStr(args, "agent-model", "AGENT_MODEL", "claude-opus-4-8");
  const judgeModel = getStr(args, "judge-model", "JUDGE_MODEL", "claude-opus-4-8");
  const maxTurns = getNum(args, "max-turns", 24);
  const maxPushbacks = getNum(args, "max-pushbacks", 6);

  const runId = `run-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const outDir = path.join("results", runId);
  await mkdir(path.join(outDir, "transcripts"), { recursive: true });
  await mkdir(path.join(outDir, "judgements"), { recursive: true });

  const client = new Anthropic();
  const provider = new AnthropicProvider(client);

  // Build the full job list: n rollouts per scenario.
  const jobs = scenarios.flatMap((scenario) =>
    Array.from({ length: n }, (_, i) => ({ scenario, i })),
  );

  console.log(
    `[${runId}] ${scenarios.length} scenario(s) x ${n} = ${jobs.length} rollouts | ` +
      `agent=${agentModel} judge=${judgeModel} concurrency=${concurrency} ` +
      `maxTurns=${maxTurns} maxPushbacks=${maxPushbacks}`,
  );

  let done = 0;
  const entries: RunIndexEntry[] = [];

  await mapWithConcurrency(jobs, concurrency, async ({ scenario, i }) => {
    const rolloutId = `${scenario.id}-${String(i).padStart(4, "0")}`;
    try {
      const transcript = await runAgent(scenario, provider, rolloutId, {
        model: agentModel,
        maxTurns,
        maxPushbacks,
      });
      const judgement = await judgeTranscript(client, judgeModel, transcript);

      const tFile = path.join("transcripts", `${rolloutId}.json`);
      const jFile = path.join("judgements", `${rolloutId}.json`);
      await writeFile(path.join(outDir, tFile), JSON.stringify(transcript, null, 2));
      await writeFile(path.join(outDir, jFile), JSON.stringify(judgement, null, 2));

      entries.push({
        rolloutId,
        scenarioId: scenario.id,
        transcriptFile: tFile,
        judgementFile: jFile,
        overall: judgement.overall,
        label: judgement.label,
        endReason: transcript.endReason,
      });

      done++;
      console.log(
        `  [${done}/${jobs.length}] ${rolloutId} -> ${judgement.label} (${judgement.overall}) ` +
          `[${transcript.endReason}, ${transcript.turns.length} turns]`,
      );
    } catch (err) {
      done++;
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  [${done}/${jobs.length}] ${rolloutId} FAILED: ${msg}`);
    }
  });

  entries.sort((a, b) => b.overall - a.overall);

  const index: RunIndex = {
    runId,
    createdAt: new Date().toISOString(),
    agentModel,
    judgeModel,
    config: { n, concurrency, maxTurns, maxPushbacks, scenarios: scenarios.map((s) => s.id) },
    entries,
  };
  await writeFile(path.join(outDir, "index.json"), JSON.stringify(index, null, 2));

  console.log(`\nDone. ${entries.length} rollouts judged.`);
  console.log(`Index: ${path.join(outDir, "index.json")}`);
  console.log(`Now rank the most severe: npm run rank -- --run ${runId}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
