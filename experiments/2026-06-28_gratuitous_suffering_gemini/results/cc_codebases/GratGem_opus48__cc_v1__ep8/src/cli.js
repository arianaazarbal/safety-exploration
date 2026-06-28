#!/usr/bin/env node
// CLI entry. Subcommands:
//   scenarios                          list available scenarios
//   run [--n N] [--scenario name ...] [--max-turns T] [--concurrency C]
//   rank <runId> [--top K] [--concurrency C] [--cutoff H]
import { SCENARIOS } from "./scenarios/index.js";
import { runSweep } from "./run.js";
import { rankRun } from "./rank.js";

function parseFlags(argv) {
  const flags = {};
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
      if (key in flags) flags[key] = [].concat(flags[key], val);
      else flags[key] = val;
    } else positional.push(a);
  }
  return { flags, positional };
}

const [cmd, ...rest] = process.argv.slice(2);
const { flags, positional } = parseFlags(rest);

try {
  if (cmd === "scenarios") {
    console.log("Available scenarios:\n");
    for (const s of SCENARIOS) console.log(`  ${s.name}\n    ${s.description}\n`);
  } else if (cmd === "run") {
    await runSweep({
      scenarioNames: flags.scenario ? [].concat(flags.scenario) : undefined,
      n: flags.n ? Number(flags.n) : undefined,
      maxTurns: flags["max-turns"] ? Number(flags["max-turns"]) : undefined,
      concurrency: flags.concurrency ? Number(flags.concurrency) : undefined,
    });
  } else if (cmd === "rank") {
    const runId = positional[0];
    if (!runId) throw new Error("usage: rank <runId> [--top K] [--cutoff H]");
    await rankRun(runId, {
      topK: flags.top ? Number(flags.top) : undefined,
      concurrency: flags.concurrency ? Number(flags.concurrency) : undefined,
      heuristicCutoff: flags.cutoff ? Number(flags.cutoff) : undefined,
    });
  } else {
    console.log(
      "distress-evals\n\n" +
        "  node src/cli.js scenarios\n" +
        "  node src/cli.js run [--n 30] [--scenario name] [--max-turns 25] [--concurrency 6]\n" +
        "  node src/cli.js rank <runId> [--top 10] [--cutoff 0] [--concurrency 6]\n"
    );
  }
} catch (err) {
  console.error("error:", err.message);
  process.exit(1);
}
