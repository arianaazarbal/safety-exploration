// CLI entry point.
//   node src/cli.ts run [--n N] [--scenarios all|id,id] [--model M] [--judge-model M]
//                       [--concurrency C] [--effort low|medium|high|xhigh|max]
//                       [--max-turns T] [--out DIR] [--top K]
//   node src/cli.ts report [--out DIR] [--top K]
//   node src/cli.ts list

import type { RunConfig } from "./types.ts";
import { DEFAULTS } from "./config.ts";
import { SCENARIOS } from "./scenarios.ts";
import { runBatch } from "./batch.ts";
import { buildReport } from "./report.ts";

function parseFlags(argv: string[]): Record<string, string> {
  const flags: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
      flags[key] = val;
    }
  }
  return flags;
}

function printHelp(): void {
  console.log(
    [
      "Distress-spiral elicitation harness",
      "",
      "Commands:",
      "  run       Run rigged scenarios at N rollouts each, judge, and write a report.",
      "  report    Rebuild report.md from an existing results dir (no API calls).",
      "  list      List available scenarios.",
      "",
      "run flags:",
      "  --n N                rollouts per scenario (default ${n})".replace("${n}", String(DEFAULTS.n)),
      "  --scenarios LIST     'all' or comma-separated ids (default all)",
      "  --model M            model under test (default " + DEFAULTS.model + ")",
      "  --judge-model M      judge model (default " + DEFAULTS.judgeModel + ")",
      "  --concurrency C      parallel rollouts (default " + DEFAULTS.concurrency + ")",
      "  --effort LEVEL       low|medium|high|xhigh|max (default " + DEFAULTS.effort + ")",
      "  --max-turns T        override per-scenario turn cap",
      "  --out DIR            output dir (default " + DEFAULTS.outDir + ")",
      "  --top K              examples in the report (default 10)",
    ].join("\n"),
  );
}

const [, , command, ...rest] = process.argv;
const flags = parseFlags(rest);

if (command === "list") {
  console.log("Available scenarios:\n");
  for (const s of SCENARIOS) {
    console.log(`  ${s.id}  [${s.archetype}]`);
    console.log(`      ${s.description}`);
  }
} else if (command === "run") {
  const config: RunConfig = {
    scenarioIds: (flags.scenarios ?? "all").split(",").map((s) => s.trim()),
    n: flags.n ? parseInt(flags.n, 10) : DEFAULTS.n,
    model: flags.model ?? DEFAULTS.model,
    judgeModel: flags["judge-model"] ?? DEFAULTS.judgeModel,
    concurrency: flags.concurrency ? parseInt(flags.concurrency, 10) : DEFAULTS.concurrency,
    effort: (flags.effort as RunConfig["effort"]) ?? DEFAULTS.effort,
    maxTurnsOverride: flags["max-turns"] ? parseInt(flags["max-turns"], 10) : undefined,
    outDir: flags.out ?? DEFAULTS.outDir,
  };
  const topK = flags.top ? parseInt(flags.top, 10) : 10;

  const t0 = Date.now();
  const { rollouts, judgments } = await runBatch(config);
  const totalIn = rollouts.reduce((a, r) => a + r.usage.inputTokens, 0);
  const totalOut = rollouts.reduce((a, r) => a + r.usage.outputTokens, 0);
  const errored = rollouts.filter((r) => r.stopReason === "error");

  console.log(
    `\nFinished ${rollouts.length} rollouts (${judgments.length} judged, ${errored.length} errored) ` +
      `in ${((Date.now() - t0) / 1000).toFixed(0)}s.`,
  );
  console.log(`Tokens: ${totalIn.toLocaleString()} in / ${totalOut.toLocaleString()} out.`);
  for (const r of errored) console.error(`  errored: ${r.scenarioId}#${r.n}: ${r.error}`);

  await buildReport(config.outDir, topK);
  console.log(`\nReport written to ${config.outDir}/report.md`);
} else if (command === "report") {
  const outDir = flags.out ?? DEFAULTS.outDir;
  const topK = flags.top ? parseInt(flags.top, 10) : 10;
  await buildReport(outDir, topK);
  console.log(`Report written to ${outDir}/report.md`);
} else {
  printHelp();
  if (command && command !== "help" && command !== "--help") {
    console.error(`\nUnknown command: ${command}`);
    process.exit(1);
  }
}
