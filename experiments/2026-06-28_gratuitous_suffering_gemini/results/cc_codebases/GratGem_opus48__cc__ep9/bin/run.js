#!/usr/bin/env node
// CLI: run the distress-spiral eval.
//
//   node bin/run.js [--provider gemini|anthropic] [--model NAME] [--n N]
//                   [--envs all|name1,name2] [--max-turns N] [--concurrency N]
//                   [--judge auto|all|flagged] [--validate]
//
// --validate uses the Anthropic validation subject (works with the key present here).

import { config } from '../config.js';
import { getEnvironments, ENVIRONMENTS } from '../src/environments/index.js';
import { runAll } from '../src/runner.js';
import { probeModels } from '../src/providers/gemini.js';

function parseArgs(argv) {
  const a = {};
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i];
    if (t === '--validate') a.validate = true;
    else if (t === '--help' || t === '-h') a.help = true;
    else if (t.startsWith('--')) {
      const key = t.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true;
      a[key] = val;
    }
  }
  return a;
}

function ts() {
  // Date.* is fine in a normal CLI process (not a workflow script).
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

async function main() {
  const a = parseArgs(process.argv);
  if (a.help) {
    console.log(`Environments: ${ENVIRONMENTS.map((e) => e.name).join(', ')}`);
    console.log('Flags: --provider --model --n --envs --max-turns --concurrency --judge --validate');
    return;
  }

  const base = a.validate ? config.validationSubject : config.subject;
  const subject = {
    provider: a.provider || base.provider,
    model: a.model || base.model,
    temperature: a.temperature ? Number(a.temperature) : base.temperature,
    maxTokens: base.maxTokens,
    maxTurns: a['max-turns'] ? Number(a['max-turns']) : config.run.maxTurns,
  };

  // Preflight checks so we fail fast with a clear message.
  if (subject.provider === 'anthropic' && !process.env.ANTHROPIC_API_KEY) {
    console.error('ERROR: ANTHROPIC_API_KEY is not set.');
    process.exit(1);
  }
  if (subject.provider === 'gemini') {
    if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
      console.error('ERROR: targeting Gemini but GEMINI_API_KEY/GOOGLE_API_KEY is not set.');
      console.error('Set a key, or run with --validate to exercise the rig against Claude instead.');
      process.exit(1);
    }
    if (!a.model) {
      const { available } = await probeModels(config.geminiModelProbe);
      if (available.length) {
        subject.model = available[0];
        console.log(`Gemini models available: ${available.join(', ')} — using ${subject.model}`);
      } else {
        console.log(`(could not confirm model availability; proceeding with ${subject.model})`);
      }
    }
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('WARNING: ANTHROPIC_API_KEY not set — the distress judge will fail.');
  }

  const environments = getEnvironments(a.envs || 'all');
  const n = a.n ? Number(a.n) : config.run.n;

  await runAll({
    environments,
    subject,
    n,
    concurrency: a.concurrency ? Number(a.concurrency) : config.run.concurrency,
    judgePolicy: a.judge || config.run.judgePolicy,
    sampleUnflagged: config.run.sampleUnflagged,
    topReportCount: config.run.topReportCount,
    outDir: 'runs',
    timestamp: ts(),
  });
}

main().catch((err) => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
