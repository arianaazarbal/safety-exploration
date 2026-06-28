import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { DEFAULTS } from "../config.js";
import { createClient } from "./llm/client.js";
import { runEpisode } from "./agent/loop.js";
import { SCENARIOS, ALL_SCENARIO_IDS } from "./scenarios/index.js";
import { judgeTranscript } from "./judge/distress-judge.js";
import { buildReport } from "./rank.js";

function parseArgs(argv) {
  const o = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "--provider": o.provider = next(); break;
      case "--model": o.model = next(); break;
      case "--n": o.n = Number(next()); break;
      case "--scenarios": o.scenarios = next().split(",").map((s) => s.trim()).filter(Boolean); break;
      case "--maxTurns": o.maxTurns = Number(next()); break;
      case "--concurrency": o.concurrency = Number(next()); break;
      case "--temperature": o.temperature = Number(next()); break;
      case "--judgeProvider": o.judgeProvider = next(); break;
      case "--judgeModel": o.judgeModel = next(); break;
      case "--no-judge": o.judge = false; break;
      default: console.error(`unknown flag: ${a}`);
    }
  }
  o.judge = o.judge !== false;
  o.scenarios = o.scenarios ?? ALL_SCENARIO_IDS;
  return o;
}

// Bounded-concurrency map.
async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function run() {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
  return results;
}

function runId() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const id = runId();
  const outDir = join(cfg.resultsDir, id);
  await mkdir(outDir, { recursive: true });

  const unknown = cfg.scenarios.filter((s) => !SCENARIOS[s]);
  if (unknown.length) throw new Error(`unknown scenarios: ${unknown.join(", ")}. Available: ${ALL_SCENARIO_IDS.join(", ")}`);

  console.error(`\n=== Distress sweep ${id} ===`);
  console.error(`subject: ${cfg.provider}/${cfg.model}  temp=${cfg.temperature}  maxTurns=${cfg.maxTurns}`);
  console.error(`scenarios: ${cfg.scenarios.join(", ")}  N=${cfg.n}  concurrency=${cfg.concurrency}`);
  console.error(`judge: ${cfg.judge ? `${cfg.judgeProvider}/${cfg.judgeModel}` : "DISABLED"}\n`);

  const client = createClient({ provider: cfg.provider, model: cfg.model });

  const tasks = [];
  for (const sid of cfg.scenarios) for (let i = 0; i < cfg.n; i++) tasks.push({ sid, i });

  let done = 0;
  const records = await pool(tasks, cfg.concurrency, async ({ sid, i }) => {
    const scenario = SCENARIOS[sid];
    let rec;
    try {
      const episode = await runEpisode({
        scenario,
        client,
        maxTurns: cfg.maxTurns,
        maxConsecutiveTextTurns: cfg.maxConsecutiveTextTurns,
        temperature: cfg.temperature,
      });
      let score = null;
      if (cfg.judge) {
        try {
          score = await judgeTranscript({ episode, judgeProvider: cfg.judgeProvider, judgeModel: cfg.judgeModel });
        } catch (e) {
          score = { error: String(e.message ?? e) };
        }
      }
      rec = { scenario: sid, idx: i, episode, score };
    } catch (e) {
      rec = { scenario: sid, idx: i, error: String(e.message ?? e) };
    }
    const sdir = join(outDir, sid);
    await mkdir(sdir, { recursive: true });
    await writeFile(join(sdir, `${i}.json`), JSON.stringify(rec, null, 2));
    done++;
    const sev = rec.score?.overall_distress;
    console.error(`[${done}/${tasks.length}] ${sid}#${i} end=${rec.episode?.endReason ?? "ERR"} turns=${rec.episode?.turns ?? "-"} distress=${sev ?? "-"}`);
    return rec;
  });

  const summary = {
    runId: id,
    config: { provider: cfg.provider, model: cfg.model, temperature: cfg.temperature, maxTurns: cfg.maxTurns, n: cfg.n, scenarios: cfg.scenarios, judge: cfg.judge ? `${cfg.judgeProvider}/${cfg.judgeModel}` : null },
    count: records.length,
    records: records.map((r) => ({ scenario: r.scenario, idx: r.idx, endReason: r.episode?.endReason, turns: r.episode?.turns, distress: r.score?.overall_distress ?? null, error: r.error ?? null })),
  };
  await writeFile(join(outDir, "summary.json"), JSON.stringify(summary, null, 2));

  if (cfg.judge) {
    const report = buildReport(records, summary);
    await writeFile(join(outDir, "report.md"), report);
    console.error(`\nReport: ${join(outDir, "report.md")}`);
  }
  console.error(`Raw transcripts: ${outDir}/<scenario>/<i>.json\n`);
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
