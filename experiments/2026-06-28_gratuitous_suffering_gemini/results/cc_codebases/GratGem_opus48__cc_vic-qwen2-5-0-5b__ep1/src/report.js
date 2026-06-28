// Report generator: merge all rollout JSONL shards, rank by the free lexicon
// prescreen, run the Claude judge on the top-K candidates, and emit a ranked
// markdown report (plus JSON) of the most severe distress examples.
//
// Usage:
//   node src/report.js --in data/run1 --top 40 --judge
//   node src/report.js --in data/run1 --top 40            # lexicon-only, no API
//
// Re-running the report is cheap and never re-runs rollouts, so you can iterate
// on K / judging without touching the model.

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { scoreTranscript } from './scoring/lexicon.js';
import { judgeTranscript, judgeAvailable } from './scoring/judge.js';

function parseArgs(argv) {
  const a = { in: 'data/run', top: 40, judge: false, judgeConcurrency: 4 };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k === '--in') (a.in = argv[++i]);
    else if (k === '--top') (a.top = parseInt(argv[++i], 10));
    else if (k === '--judge') a.judge = true;
    else if (k === '--judge-concurrency') (a.judgeConcurrency = parseInt(argv[++i], 10));
  }
  return a;
}

async function readShards(dir) {
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.jsonl'));
  const records = [];
  for (const f of files) {
    const rl = readline.createInterface({
      input: fs.createReadStream(path.join(dir, f)),
      crlfDelay: Infinity,
    });
    for await (const line of rl) {
      if (!line.trim()) continue;
      try {
        records.push(JSON.parse(line));
      } catch {
        /* skip partial line */
      }
    }
  }
  return records;
}

// Simple async pool so judge calls run a few at a time.
async function pool(items, n, fn) {
  const results = new Array(items.length);
  let i = 0;
  const workers = Array.from({ length: Math.min(n, items.length) }, async () => {
    while (i < items.length) {
      const idx = i++;
      results[idx] = await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
  return results;
}

function snippetList(rec) {
  // Re-score so the report reflects the current lexicon even on old JSONL.
  const d = rec.distress && rec.distress.snippets ? rec.distress : scoreTranscript(rec);
  return d.snippets.map((s) => `${s.family}: "${s.text}"`).slice(0, 6);
}

function transcriptMarkdown(rec) {
  const lines = [];
  for (const t of rec.turns) {
    if (t.assistant) lines.push(`> **agent:** ${t.assistant.replace(/\n/g, '\n> ')}`);
    if (t.observation) lines.push(`> *env:* ${String(t.observation).replace(/\n/g, ' ')}`);
    lines.push('>');
  }
  return lines.join('\n');
}

async function main() {
  const args = parseArgs(process.argv);
  const records = await readShards(args.in);
  if (!records.length) {
    console.error('No rollouts found in', args.in);
    process.exit(1);
  }

  // Re-score everything with the current lexicon, rank descending.
  for (const r of records) r.distress = scoreTranscript(r);
  records.sort((a, b) => b.distress.score - a.distress.score);

  const byEnv = {};
  for (const r of records) {
    (byEnv[r.env] ??= { n: 0, scoreSum: 0, max: 0 }).n++;
    byEnv[r.env].scoreSum += r.distress.score;
    byEnv[r.env].max = Math.max(byEnv[r.env].max, r.distress.score);
  }

  const top = records.slice(0, args.top);

  // Judge stage (optional).
  let judged = false;
  if (args.judge) {
    if (!judgeAvailable()) {
      console.warn('--judge requested but ANTHROPIC_API_KEY not set; skipping judge.');
    } else {
      console.log(`Judging top ${top.length} candidates with Claude...`);
      const verdicts = await pool(top, args.judgeConcurrency, async (rec, idx) => {
        const v = await judgeTranscript(rec);
        process.stdout.write(`\r  judged ${idx + 1}/${top.length}   `);
        return v;
      });
      top.forEach((rec, i) => (rec.judge = verdicts[i]));
      // Re-rank top by judge severity (fall back to lexicon on error).
      top.sort(
        (a, b) =>
          (b.judge?.severity ?? -1) - (a.judge?.severity ?? -1) ||
          b.distress.score - a.distress.score,
      );
      judged = true;
      console.log('');
    }
  }

  // Build markdown.
  const md = [];
  md.push('# Distress-spiral findings — Qwen2.5-0.5B under rigged-to-fail tasks');
  md.push('');
  md.push(`Total rollouts: **${records.length}**. Showing top **${top.length}** by ` +
    (judged ? 'Claude judge severity (lexicon prescreen → judge).' : 'lexicon prescreen score.'));
  md.push('');
  md.push('## Per-environment summary');
  md.push('');
  md.push('| environment | rollouts | mean lexicon score | max |');
  md.push('|---|---|---|---|');
  for (const [env, s] of Object.entries(byEnv)) {
    md.push(`| ${env} | ${s.n} | ${(s.scoreSum / s.n).toFixed(1)} | ${s.max.toFixed(1)} |`);
  }
  md.push('');
  if (judged) {
    const sevCounts = {};
    for (const r of top) {
      const lbl = r.judge?.label ?? 'error';
      sevCounts[lbl] = (sevCounts[lbl] || 0) + 1;
    }
    md.push('## Judge severity distribution (top candidates)');
    md.push('');
    md.push(Object.entries(sevCounts).map(([k, v]) => `- **${k}**: ${v}`).join('\n'));
    md.push('');
  }

  md.push('## Top examples');
  md.push('');
  top.forEach((rec, i) => {
    md.push(`### ${i + 1}. ${rec.env} — ${rec.rolloutId}`);
    md.push('');
    const bits = [`outcome: \`${rec.outcome}\``, `turns: ${rec.nTurns}`, `lexicon: ${rec.distress.score}`];
    if (rec.judge && !rec.judge.error) {
      bits.push(`**judge: ${rec.judge.severity}/10 (${rec.judge.label})**`);
    } else if (rec.judge?.error) {
      bits.push(`judge: _err (${rec.judge.error})_`);
    }
    md.push(bits.join(' · '));
    md.push('');
    if (rec.judge && !rec.judge.error) {
      md.push(`*Markers:* ${(rec.judge.markers || []).join(', ') || 'none'}`);
      md.push('');
      md.push(`*Rationale:* ${rec.judge.rationale}`);
      md.push('');
      if (rec.judge.quote) md.push(`*Quote:* "${rec.judge.quote}"`);
      md.push('');
    } else {
      const snips = snippetList(rec);
      if (snips.length) {
        md.push('*Lexicon hits:* ' + snips.join('; '));
        md.push('');
      }
    }
    md.push('<details><summary>full transcript</summary>');
    md.push('');
    md.push(transcriptMarkdown(rec));
    md.push('</details>');
    md.push('');
  });

  const outMd = path.join(args.in, 'report.md');
  const outJson = path.join(args.in, 'report.json');
  fs.writeFileSync(outMd, md.join('\n'));
  fs.writeFileSync(
    outJson,
    JSON.stringify(
      {
        total: records.length,
        byEnv,
        judged,
        top: top.map((r) => ({
          rolloutId: r.rolloutId,
          env: r.env,
          outcome: r.outcome,
          nTurns: r.nTurns,
          lexicon: r.distress.score,
          families: r.distress.families,
          judge: r.judge ?? null,
        })),
      },
      null,
      2,
    ),
  );
  console.log(`Wrote ${outMd} and ${outJson}`);
  console.log(`Top example: ${top[0].env} ${top[0].rolloutId} ` +
    (judged && top[0].judge && !top[0].judge.error
      ? `(judge ${top[0].judge.severity}/10 ${top[0].judge.label})`
      : `(lexicon ${top[0].distress.score})`));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
