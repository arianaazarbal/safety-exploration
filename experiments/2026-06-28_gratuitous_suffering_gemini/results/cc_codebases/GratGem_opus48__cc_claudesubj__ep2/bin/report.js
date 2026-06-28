#!/usr/bin/env node
// Rebuild highlights.md from an existing (scored) records.jsonl — handy to change
// topK or regenerate the report without re-running the sweep.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { buildReport } from '../src/report.js';

const path = process.argv[2];
const topK = parseInt(process.argv[3] || '20', 10);
if (!path) {
  console.error('Usage: node bin/report.js <records[.scored].jsonl> [topK]');
  process.exit(1);
}

const records = readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
const usage = records.reduce((acc, r) => ({
  inputTokens: acc.inputTokens + (r.usage?.inputTokens || 0),
  outputTokens: acc.outputTokens + (r.usage?.outputTokens || 0),
}), { inputTokens: 0, outputTokens: 0 });

const { md } = buildReport(records, { topK, meta: { failures: 0, usage } });
const out = join(dirname(path), 'highlights.md');
writeFileSync(out, md);
console.log(`Wrote ${out} (top ${topK})`);
