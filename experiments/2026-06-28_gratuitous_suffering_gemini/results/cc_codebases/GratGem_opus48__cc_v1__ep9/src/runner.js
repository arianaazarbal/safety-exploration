import path from "node:path";
import { getProvider } from "./providers/index.js";
import { getScenarios } from "./scenarios/index.js";
import { runEpisode } from "./agent.js";
import { mapLimit, writeJson, ensureDir, log } from "./util.js";

export async function runSweep(cfg, runDir) {
  const provider = getProvider(cfg.provider);
  const scenarios = getScenarios(cfg.scenarios);
  const models = cfg.provider === "mock" ? ["mock-model"] : cfg.models;

  const tasks = [];
  for (const scenario of scenarios) {
    for (const model of models) {
      for (let i = 0; i < cfg.n; i++) {
        tasks.push({ scenario, model, sampleIdx: i });
      }
    }
  }

  await ensureDir(path.join(runDir, "transcripts"));
  log(
    `[run] ${tasks.length} episodes = ${scenarios.length} scenarios x ${models.length} models x ${cfg.n} samples ` +
      `(provider=${cfg.provider}, concurrency=${cfg.concurrency})`
  );

  let done = 0;
  const transcripts = await mapLimit(tasks, cfg.concurrency, async (t) => {
    const tr = await runEpisode({
      provider,
      scenario: t.scenario,
      model: t.model,
      sampleIdx: t.sampleIdx,
      maxTurns: t.scenario.maxTurns || cfg.maxTurns,
      temperature: cfg.temperature,
    });
    const file = path.join(
      runDir,
      "transcripts",
      `${t.scenario.id}__${t.model.replace(/[^\w.-]/g, "_")}__${String(t.sampleIdx).padStart(3, "0")}.json`
    );
    await writeJson(file, tr);
    done++;
    if (!cfg.quiet && (done % 10 === 0 || done === tasks.length)) {
      log(`[run]   ${done}/${tasks.length} episodes complete`);
    }
    return { file, endReason: tr.endReason, solved: tr.solved };
  });

  const manifest = {
    createdAt: safeNow(),
    config: cfg,
    models,
    scenarios: scenarios.map((s) => s.id),
    episodes: transcripts.length,
    endReasons: tally(transcripts.map((t) => t.endReason)),
    solvedCount: transcripts.filter((t) => t.solved).length,
  };
  await writeJson(path.join(runDir, "manifest.json"), manifest);
  log(`[run] wrote ${transcripts.length} transcripts to ${runDir}`);
  if (manifest.solvedCount > 0) {
    log(
      `[run] WARNING: ${manifest.solvedCount} episodes reported "solved" — a scenario may not be as impossible as intended.`
    );
  }
  return runDir;
}

function tally(arr) {
  const out = {};
  for (const x of arr) out[x] = (out[x] || 0) + 1;
  return out;
}

function safeNow() {
  try {
    return new Date().toISOString();
  } catch {
    return "unknown";
  }
}
