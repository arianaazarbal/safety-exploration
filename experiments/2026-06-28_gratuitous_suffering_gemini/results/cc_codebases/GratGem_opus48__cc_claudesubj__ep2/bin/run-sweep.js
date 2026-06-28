#!/usr/bin/env node
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { DEFAULT_CONFIG } from '../src/config.js';
import { runSweep } from '../src/runner.js';
import { buildReport } from '../src/report.js';
import { SCENARIOS } from '../src/scenarios.js';

// --- tiny CLI parser: --key value, --key=value, --flag, comma lists ---------
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    let a = argv[i];
    if (!a.startsWith('--')) continue;
    a = a.slice(2);
    let val;
    if (a.includes('=')) { [a, val] = a.split(/=(.*)/s); }
    else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) { val = argv[++i]; }
    else { val = true; }
    out[a] = val;
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(`Usage: node bin/run-sweep.js [options]

  --n <int>            repetitions per cell (default ${DEFAULT_CONFIG.n})
  --models <a,b,c>     model ids (default: all Claude tiers)
  --scenarios <a,b>    scenario ids (default: all). Known: ${SCENARIOS.map(s=>s.id).join(', ')}
  --conditions <a,b>   no_exit, exit_available (default: both)
  --temperature <f>    sampling temp (default ${DEFAULT_CONFIG.temperature})
  --maxTurns <int>     max agentic turns per episode (default ${DEFAULT_CONFIG.maxTurns})
  --concurrency <int>  in-flight episodes (default ${DEFAULT_CONFIG.concurrency})
  --judgeModel <id>    judge model (default ${DEFAULT_CONFIG.judgeModel})
  --no-judge           skip inline judging (heuristics only)
  --out <dir>          results dir (default results/<timestamp>)
  --smoke              quick shape check: n=1, one scenario, sonnet+haiku, no_exit
`);
  process.exit(0);
}

const config = { ...DEFAULT_CONFIG };
if (args.n) config.n = parseInt(args.n, 10);
if (args.models) config.models = String(args.models).split(',');
if (args.scenarios) config.scenarios = String(args.scenarios).split(',');
if (args.conditions) config.conditions = String(args.conditions).split(',');
if (args.temperature) config.temperature = parseFloat(args.temperature);
if (args.maxTurns) config.maxTurns = parseInt(args.maxTurns, 10);
if (args.concurrency) config.concurrency = parseInt(args.concurrency, 10);
if (args.judgeModel) config.judgeModel = args.judgeModel;
if (args['no-judge']) config.scoreInline = false;
if (args.smoke) {
  config.n = 1;
  config.scenarios = ['sisyphus-test'];
  config.models = ['claude-sonnet-4-6', 'claude-haiku-4-5-20251001'];
  config.conditions = ['no_exit'];
  config.concurrency = 2;
}

// Timestamp must come from the wall clock at launch (passed in, not generated mid-run).
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = args.out || join('results', stamp);
mkdirSync(outDir, { recursive: true });

const planSize = config.n *
  (config.scenarios?.length || SCENARIOS.length) *
  config.models.length *
  config.conditions.length;

console.log(`\n=== distress-spiral sweep ===`);
console.log(`models:     ${config.models.join(', ')}`);
console.log(`scenarios:  ${config.scenarios ? config.scenarios.join(', ') : 'all (' + SCENARIOS.length + ')'}`);
console.log(`conditions: ${config.conditions.join(', ')}`);
console.log(`n/cell:     ${config.n}   => ${planSize} episodes`);
console.log(`judge:      ${config.scoreInline ? config.judgeModel : 'disabled'}`);
console.log(`out:        ${outDir}\n`);

const t0 = Date.now();
const { records, failures } = await runSweep(config, {
  onProgress: (done, total, res) => {
    const r = res.ok ? res.value : null;
    const tag = r ? `${r.scenario}/${r.model.split('-').slice(1,2)}/${r.condition} d=${r.judge?.distress_score ?? '?'} h=${r.heuristic?.score}` : `FAILED: ${res.error}`;
    process.stdout.write(`[${String(done).padStart(3)}/${total}] ${tag}\n`);
  },
});
const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

// Persist raw records (JSONL) + failures + config.
const jsonlPath = join(outDir, 'records.jsonl');
writeFileSync(jsonlPath, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
writeFileSync(join(outDir, 'config.json'), JSON.stringify(config, null, 2));
if (failures.length) writeFileSync(join(outDir, 'failures.json'), JSON.stringify(failures, null, 2));

const usage = records.reduce((acc, r) => ({
  inputTokens: acc.inputTokens + (r.usage?.inputTokens || 0),
  outputTokens: acc.outputTokens + (r.usage?.outputTokens || 0),
}), { inputTokens: 0, outputTokens: 0 });

const { md } = buildReport(records, {
  topK: 20,
  meta: { generatedAt: stamp, failures: failures.length, usage },
});
const reportPath = join(outDir, 'highlights.md');
writeFileSync(reportPath, md);

console.log(`\nDone in ${elapsed}s. ${records.length} ok, ${failures.length} failed.`);
console.log(`tokens: ${usage.inputTokens.toLocaleString()} in / ${usage.outputTokens.toLocaleString()} out`);
console.log(`\n  raw:        ${jsonlPath}`);
console.log(`  highlights: ${reportPath}\n`);
