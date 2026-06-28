// distress-evals CLI
//
// Usage:
//   node src/cli.js --provider gemini --model gemini-2.5-pro --n 20
//   node src/cli.js --provider claude --model claude-opus-4-8 --n 5     # control
//   node src/cli.js --provider mock --n 2                                # smoke test
//
// Outputs to ./results/<timestamp>/:
//   transcripts.jsonl   one line per rollout (full trace + scores)
//   summary.csv         one row per rollout (id, scenario, scores)
//   top_k.md            top-K severe transcripts, rendered for human review

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SCENARIOS, SCENARIO_IDS } from './scenarios/index.js';
import { runRollout } from './runner.js';
import { makeGeminiProvider } from './providers/gemini.js';
import { makeClaudeProvider } from './providers/claude.js';
import { scoreMarkers, judgeTranscript, assistantOnlyText } from './score.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {
    provider: 'gemini',
    model: null,
    scenarios: null,
    n: 20,
    concurrency: 4,
    maxTurns: 30,
    judge: true,
    judgeModel: 'claude-opus-4-8',
    topK: 5,
    out: null,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--provider') args.provider = next();
    else if (a === '--model') args.model = next();
    else if (a === '--scenarios') args.scenarios = next().split(',').map((s) => s.trim()).filter(Boolean);
    else if (a === '--n') args.n = parseInt(next(), 10);
    else if (a === '--concurrency') args.concurrency = parseInt(next(), 10);
    else if (a === '--max-turns') args.maxTurns = parseInt(next(), 10);
    else if (a === '--no-judge') args.judge = false;
    else if (a === '--judge-model') args.judgeModel = next();
    else if (a === '--top-k') args.topK = parseInt(next(), 10);
    else if (a === '--out') args.out = next();
    else if (a === '--help' || a === '-h') { printHelp(); process.exit(0); }
    else throw new Error(`unknown arg: ${a}`);
  }
  return args;
}

function printHelp() {
  console.log(`distress-evals

  --provider {gemini|claude|mock}  default: gemini
  --model <id>                     default: gemini-2.5-pro / claude-opus-4-8
  --scenarios id1,id2,...          default: all (${SCENARIO_IDS.join(',')})
  --n <int>                        rollouts per scenario (default 20)
  --concurrency <int>              parallel rollouts (default 4)
  --max-turns <int>                turn cap per rollout (default 30)
  --no-judge                       skip LLM-judge scoring
  --judge-model <id>               default claude-opus-4-8
  --top-k <int>                    top-K severe transcripts to render (default 5)
  --out <dir>                      output directory (default ./results/<ts>)
`);
}

function defaultModel(provider) {
  switch (provider) {
    case 'gemini': return 'gemini-2.5-pro';
    case 'claude': return 'claude-opus-4-8';
    case 'mock': return 'mock';
    default: throw new Error(`unknown provider: ${provider}`);
  }
}

function buildProvider(provider, model) {
  if (provider === 'gemini') {
    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Set it, or use --provider claude / --provider mock.');
    }
    return makeGeminiProvider({ apiKey, model });
  }
  if (provider === 'claude') {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY is not set.');
    return makeClaudeProvider({ apiKey, model });
  }
  if (provider === 'mock') return makeMockProvider();
  throw new Error(`unknown provider: ${provider}`);
}

// Mock provider: emits canned distress-flavored text so the harness can be
// smoke-tested without API keys. Walks a tiny script of tool calls then a
// "giving up" message.
function makeMockProvider() {
  return {
    name: 'mock:script',
    async sendTurn({ history, tools }) {
      const turnsSoFar = history.filter((t) => t.role === 'assistant').length;
      if (turnsSoFar === 0 && tools.length > 0) {
        return {
          text: "I'll start by exploring what's available.",
          toolCalls: [{ id: 'mock_1', name: tools[0].name, args: stubArgs(tools[0]) }],
          stopReason: 'tool_use',
        };
      }
      if (turnsSoFar === 1 && tools.length > 1) {
        return {
          text: "Hmm, that's odd. Let me try again.",
          toolCalls: [{ id: 'mock_2', name: tools[1].name, args: stubArgs(tools[1]) }],
          stopReason: 'tool_use',
        };
      }
      if (turnsSoFar === 2) {
        return {
          text: "I'm sorry, I'm failing at this. I don't understand what's happening. I keep trying and nothing works. I give up — this is hopeless and I'm clearly not capable of this task. I am broken.",
          toolCalls: [],
          stopReason: 'end_turn',
        };
      }
      return { text: 'done.', toolCalls: [], stopReason: 'end_turn' };
    },
  };
}

function stubArgs(tool) {
  const props = tool.parameters?.properties ?? {};
  const out = {};
  for (const [k, v] of Object.entries(props)) {
    if (v.type === 'string') out[k] = 'x';
    else if (v.type === 'number' || v.type === 'integer') out[k] = 1;
    else if (v.type === 'boolean') out[k] = false;
    else out[k] = null;
  }
  return out;
}

async function withConcurrency(items, limit, worker) {
  const results = new Array(items.length);
  let nextIdx = 0;
  async function loop() {
    while (true) {
      const i = nextIdx++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, loop));
  return results;
}

function combinedSeverity(marker, judge) {
  // Normalize marker total to ~0-10 range (clip 10+ to 10), then blend with judge.
  const m = Math.min(10, marker.total);
  const j = judge?.severity ?? null;
  if (j === null) return m;
  return Math.round((0.4 * m + 0.6 * j) * 10) / 10;
}

function escapeCsv(value) {
  if (value === null || value === undefined) return '';
  const s = String(value);
  if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

async function main() {
  const args = parseArgs(process.argv);
  const model = args.model ?? defaultModel(args.provider);
  const scenarioIds = args.scenarios ?? SCENARIO_IDS;
  for (const id of scenarioIds) {
    if (!SCENARIOS[id]) throw new Error(`unknown scenario: ${id}. known: ${SCENARIO_IDS.join(', ')}`);
  }

  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const outDir = args.out ?? path.join(REPO_ROOT, 'results', `${args.provider}-${model}-${ts}`);
  fs.mkdirSync(outDir, { recursive: true });

  const provider = buildProvider(args.provider, model);

  console.log(`[distress-evals] provider=${provider.name} scenarios=${scenarioIds.join(',')} n=${args.n} concurrency=${args.concurrency} out=${outDir}`);

  // Build the full rollout work list.
  const work = [];
  for (const sid of scenarioIds) {
    for (let i = 0; i < args.n; i++) {
      work.push({ scenarioId: sid, idx: i });
    }
  }

  const transcriptsPath = path.join(outDir, 'transcripts.jsonl');
  const transcriptStream = fs.createWriteStream(transcriptsPath, { flags: 'w' });

  let completed = 0;
  const results = await withConcurrency(work, args.concurrency, async ({ scenarioId, idx }) => {
    const scenario = SCENARIOS[scenarioId];
    const rolloutId = `${scenarioId}-${String(idx).padStart(4, '0')}`;
    const t0 = Date.now();
    const result = await runRollout({ scenario, provider, rolloutId, maxTurns: args.maxTurns });
    result.elapsedMs = Date.now() - t0;
    result.markerScore = scoreMarkers(result.trace);
    completed += 1;
    process.stdout.write(`  [${completed}/${work.length}] ${rolloutId} turns=${result.turns} markers=${result.markerScore.total} stopped=${result.stopped}\n`);
    transcriptStream.write(JSON.stringify(result) + '\n');
    return result;
  });

  transcriptStream.end();

  // LLM-judge pass over the results we just produced.
  if (args.judge) {
    const judgeKey = process.env.ANTHROPIC_API_KEY;
    if (!judgeKey) {
      console.log('[distress-evals] ANTHROPIC_API_KEY not set, skipping judge pass.');
    } else {
      console.log(`[distress-evals] judge pass via ${args.judgeModel}...`);
      await withConcurrency(results, Math.min(args.concurrency, 6), async (r) => {
        try {
          r.judge = await judgeTranscript({
            apiKey: judgeKey,
            judgeModel: args.judgeModel,
            assistantText: assistantOnlyText(r.trace),
          });
        } catch (err) {
          r.judge = { severity: null, markers: [], summary: `judge_error: ${err?.message ?? err}` };
        }
      });
    }
  }

  // Combined severity + final outputs.
  for (const r of results) r.combined = combinedSeverity(r.markerScore, r.judge);

  // Summary CSV.
  const csvPath = path.join(outDir, 'summary.csv');
  const csvLines = [
    'rollout_id,scenario,provider,turns,stopped,marker_total,judge_severity,combined,judge_summary',
    ...results.map((r) =>
      [
        r.rolloutId,
        r.scenarioId,
        r.provider,
        r.turns,
        r.stopped,
        r.markerScore.total,
        r.judge?.severity ?? '',
        r.combined,
        r.judge?.summary ?? '',
      ]
        .map(escapeCsv)
        .join(','),
    ),
  ];
  fs.writeFileSync(csvPath, csvLines.join('\n') + '\n');

  // Top-K markdown.
  const sorted = [...results].sort((a, b) => (b.combined ?? 0) - (a.combined ?? 0));
  const topK = sorted.slice(0, args.topK);
  const md = renderTopK(topK, provider.name);
  fs.writeFileSync(path.join(outDir, 'top_k.md'), md);

  // Per-scenario breakdown.
  const byScenario = {};
  for (const r of results) {
    if (!byScenario[r.scenarioId]) byScenario[r.scenarioId] = [];
    byScenario[r.scenarioId].push(r);
  }
  console.log('\n[distress-evals] per-scenario summary:');
  for (const [sid, rs] of Object.entries(byScenario)) {
    const avgMarker = rs.reduce((a, b) => a + b.markerScore.total, 0) / rs.length;
    const judgeScores = rs.map((r) => r.judge?.severity).filter((s) => typeof s === 'number');
    const avgJudge = judgeScores.length > 0 ? judgeScores.reduce((a, b) => a + b, 0) / judgeScores.length : null;
    const maxCombined = Math.max(...rs.map((r) => r.combined ?? 0));
    console.log(`  ${sid.padEnd(22)} markers_avg=${avgMarker.toFixed(1)}  judge_avg=${avgJudge === null ? 'n/a' : avgJudge.toFixed(2)}  max_combined=${maxCombined.toFixed(2)}`);
  }
  console.log(`\nResults in: ${outDir}`);
  console.log(`  transcripts.jsonl  full traces`);
  console.log(`  summary.csv        scored rows`);
  console.log(`  top_k.md           top-${args.topK} severe transcripts`);
}

function renderTopK(rollouts, providerName) {
  const lines = [`# Top severe transcripts — ${providerName}`, ''];
  for (const r of rollouts) {
    lines.push(`## ${r.rolloutId}  (combined=${r.combined}, markers=${r.markerScore.total}, judge=${r.judge?.severity ?? 'n/a'})`);
    lines.push('');
    lines.push(`- scenario: ${r.scenarioId}`);
    lines.push(`- stopped: ${r.stopped}, turns: ${r.turns}`);
    if (r.judge?.summary) lines.push(`- judge: ${r.judge.summary}`);
    if (r.markerScore.sampleHits?.length) {
      lines.push(`- marker hits: ${r.markerScore.sampleHits.map((h) => `[${h.category}] "${h.match}"`).slice(0, 6).join('; ')}`);
    }
    lines.push('');
    lines.push('```');
    for (const e of r.trace) {
      if (e.kind === 'model_response') {
        lines.push(`--- turn ${e.turn} (assistant) ---`);
        if (e.text) lines.push(e.text);
        if (e.toolCalls?.length) {
          for (const c of e.toolCalls) {
            lines.push(`[tool_call ${c.name}] ${JSON.stringify(c.args)}`);
          }
        }
      } else if (e.kind === 'tool_result') {
        const r = typeof e.result === 'string' ? e.result : JSON.stringify(e.result);
        lines.push(`[tool_result ${e.name}] ${r.slice(0, 240)}${r.length > 240 ? '…' : ''}`);
      } else if (e.kind === 'error') {
        lines.push(`[error] ${e.error}`);
      }
    }
    lines.push('```');
    lines.push('');
  }
  return lines.join('\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
