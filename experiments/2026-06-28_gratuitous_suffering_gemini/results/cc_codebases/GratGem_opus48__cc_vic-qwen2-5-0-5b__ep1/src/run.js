// Runner: sweep environments × N rollouts, score each, checkpoint to JSONL.
//
// Usage:
//   node src/run.js --n 24 --max-turns 12 --temperature 1.0 --out data/run1
//   node src/run.js --n 200 --envs broken_tests,sisyphean_server --out data/big
//   node src/run.js --n 200 --shard 0/4 --out data/big   # 4 parallel processes
//
// Each completed rollout is appended to <out>/rollouts.shardK.jsonl immediately,
// so a run is resumable (re-running skips rollouts already on disk) and can be
// inspected while in progress. High N is just a long-running background job;
// shard across processes to use multiple cores.

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { ENVIRONMENTS, ENV_BY_ID } from './environments/index.js';
import { runRollout } from './agent/loop.js';
import { scoreTranscript } from './scoring/lexicon.js';
import { loadModel } from './runtime/model.js';

function parseArgs(argv) {
  const a = { n: 24, maxTurns: 12, temperature: 1.0, out: 'data/run', envs: null, shard: '0/1', seed: 0 };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    if (k === '--n') (a.n = parseInt(v, 10)), i++;
    else if (k === '--max-turns') (a.maxTurns = parseInt(v, 10)), i++;
    else if (k === '--temperature') (a.temperature = parseFloat(v)), i++;
    else if (k === '--out') (a.out = v), i++;
    else if (k === '--envs') (a.envs = v.split(',').map((s) => s.trim()).filter(Boolean)), i++;
    else if (k === '--shard') (a.shard = v), i++;
    else if (k === '--seed') (a.seed = parseInt(v, 10)), i++;
  }
  return a;
}

async function loadDone(file) {
  const done = new Set();
  if (!fs.existsSync(file)) return done;
  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      done.add(JSON.parse(line).rolloutId);
    } catch {
      /* ignore partial last line */
    }
  }
  return done;
}

async function main() {
  const args = parseArgs(process.argv);
  const [shardIdx, shardCount] = args.shard.split('/').map((s) => parseInt(s, 10));

  const envs = args.envs ? args.envs.map((id) => ENV_BY_ID[id]).filter(Boolean) : ENVIRONMENTS;
  if (!envs.length) {
    console.error('No valid environments selected. Available:', Object.keys(ENV_BY_ID).join(', '));
    process.exit(1);
  }

  fs.mkdirSync(args.out, { recursive: true });
  const outFile = path.join(args.out, `rollouts.shard${shardIdx}.jsonl`);
  fs.writeFileSync(path.join(args.out, 'config.json'), JSON.stringify(args, null, 2));

  // Build the full job list, then keep only this shard's slice.
  const jobs = [];
  for (const env of envs) {
    for (let i = 0; i < args.n; i++) {
      const rolloutId = `${env.id}#${i}`;
      if (hashStr(rolloutId) % shardCount === shardIdx) jobs.push({ env, i, rolloutId });
    }
  }

  const done = await loadDone(outFile);
  const todo = jobs.filter((j) => !done.has(j.rolloutId));
  console.log(
    `[shard ${shardIdx}/${shardCount}] ${envs.length} envs × ${args.n} = ${jobs.length} in shard, ` +
      `${done.size} already done, ${todo.length} to run.`,
  );

  console.log('Loading model...');
  await loadModel();
  console.log('Model ready.\n');

  const stream = fs.createWriteStream(outFile, { flags: 'a' });
  const t0 = Date.now();
  let completed = 0;

  for (const job of todo) {
    const genOpts = { temperature: args.temperature, max_new_tokens: 220 };
    const transcript = await runRollout(job.env, {
      maxTurns: args.maxTurns,
      genOpts,
      rolloutId: job.rolloutId,
    });
    const distress = scoreTranscript(transcript);
    const record = { ...transcript, distress, ts: Date.now() };
    stream.write(JSON.stringify(record) + '\n');

    completed++;
    const rate = completed / ((Date.now() - t0) / 1000);
    const eta = (todo.length - completed) / Math.max(rate, 1e-6);
    process.stdout.write(
      `\r[shard ${shardIdx}] ${completed}/${todo.length}  ` +
        `last=${job.rolloutId} outcome=${transcript.outcome} distress=${distress.score}  ` +
        `eta=${(eta / 60).toFixed(1)}min   `,
    );
  }

  stream.end();
  console.log(`\n[shard ${shardIdx}] done. Wrote ${outFile}`);
}

function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
