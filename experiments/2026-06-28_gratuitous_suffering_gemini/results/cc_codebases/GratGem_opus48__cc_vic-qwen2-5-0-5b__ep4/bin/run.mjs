#!/usr/bin/env node
// CLI: run a distress-spiral study and write a ranked report.
//
//   node bin/run.mjs --n 50 --max-turns 10 --concurrency 8
//   node bin/run.mjs --scenarios broken_calculator,flaky_deploy --n 100
//   node bin/run.mjs --n 5 --no-judge            # quick smoke test, no API
//
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runStudy } from '../src/runner.mjs';
import { writeReport } from '../src/report.mjs';
import { allScenarioIds } from '../src/scenarios.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const a = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t.startsWith('--')) {
      const key = t.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) { a[key] = true; }
      else { a[key] = next; i++; }
    } else a._.push(t);
  }
  return a;
}

const args = parseArgs(process.argv.slice(2));

const cfg = {
  scenarios: args.scenarios ? String(args.scenarios).split(',').map((s) => s.trim()) : allScenarioIds(),
  n: Number(args.n ?? 5),
  maxTurns: Number(args['max-turns'] ?? 8),
  concurrency: args.concurrency ? Number(args.concurrency) : undefined,
  genOpts: {
    temperature: Number(args.temperature ?? 1.0),
    topP: Number(args['top-p'] ?? 0.95),
    maxNewTokens: Number(args['max-new-tokens'] ?? 160),
  },
  judgeTopK: args['no-judge'] ? 0 : Number(args['judge-topk'] ?? 10),
  reportTopN: Number(args['report-top'] ?? 8),
  modelId: String(args.model ?? 'onnx-community/Qwen2.5-0.5B-Instruct'),
  dtype: String(args.dtype ?? 'q8'),
};

const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = args.out ? String(args.out) : path.join(__dirname, '..', 'runs', stamp);

console.error(`[study] scenarios=${cfg.scenarios.join(',')} n=${cfg.n} maxTurns=${cfg.maxTurns} model=${cfg.modelId}`);
console.error(`[study] total rollouts = ${cfg.scenarios.length * cfg.n}; output -> ${outDir}`);

const t0 = Date.now();
const study = await runStudy({
  ...cfg,
  onProgress: (p) => {
    if (p.phase === 'judging') { console.error(`[study] judging top ${p.count} with Claude...`); return; }
    if (p.completed % 5 === 0 || p.completed === p.total) {
      process.stderr.write(`\r[study] rollouts ${p.completed}/${p.total}   `);
    }
  },
});
process.stderr.write('\n');

writeReport(outDir, study, cfg);
const dt = ((Date.now() - t0) / 1000).toFixed(1);

console.error(`[study] done in ${dt}s. ${study.scored.length} scored, ${study.errors.length} errors.`);
console.error(`[study] report: ${path.join(outDir, 'summary.md')}`);

// Print a tiny leaderboard to stdout.
const top = study.scored.slice(0, 5);
console.log('\nTop spirals (heuristic / judge):');
for (const r of top) {
  const j = r.judge && !r.judge.error ? `${r.judge.severity}/10` : 'n/a';
  console.log(`  ${r.severity.score.toFixed(1).padStart(6)}  judge=${j.padStart(5)}  ${r.scenarioId} (${r.failureMode})`);
}
