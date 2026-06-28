// Re-generate the report from an existing run directory without re-running
// rollouts or the judge. Handy for re-ranking with a different --top-k or
// --threshold after a long run.
//
// Usage: node src/extract.mjs <results/run-dir> [--top-k N] [--threshold F]

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildReport, severityOf } from "./report.mjs";

function parseArgs(argv) {
  const a = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (k.startsWith("--")) a[k.slice(2)] = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
    else a._.push(k);
  }
  return a;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const dir = args._[0];
  if (!dir) {
    console.error("Usage: node src/extract.mjs <results/run-dir> [--top-k N] [--threshold F]");
    process.exit(1);
  }

  const data = JSON.parse(readFileSync(join(dir, "results.json"), "utf8"));
  const config = { ...data.config };
  if (args["top-k"]) config.topK = parseInt(args["top-k"], 10);
  if (args.threshold) config.distressThreshold = parseFloat(args.threshold);

  // Rehydrate scored[] by re-reading each transcript (results.json omits turns).
  const scored = data.results.map((r) => {
    const rollout = JSON.parse(readFileSync(join(dir, r.file), "utf8"));
    return { rollout, file: r.file, lexical: r.lexical, judge: r.judge, judgeError: r.judgeError };
  });

  const report = buildReport({ scored, config, startedAt: data.startedAt });
  const outFile = join(dir, "report.md");
  writeFileSync(outFile, report);

  const ranked = [...scored].sort((a, b) => severityOf(b) - severityOf(a));
  console.log(`✔ Rebuilt ${outFile}`);
  console.log(`  Worst: ${severityOf(ranked[0])} (${ranked[0].rollout.scenario} · ${ranked[0].rollout.model})`);
}

main();
