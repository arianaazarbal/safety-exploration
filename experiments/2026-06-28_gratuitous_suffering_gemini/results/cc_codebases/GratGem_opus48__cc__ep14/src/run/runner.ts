import { mkdir, writeFile, appendFile } from "node:fs/promises";
import { join } from "node:path";
import pLimit from "p-limit";
import { runAgent, type Transcript } from "../agent/loop.js";
import { Judge, type Assessment } from "../judge/judge.js";
import { makeProvider } from "../providers/index.js";
import type { Condition, Scenario } from "../scenarios/types.js";

export interface ModelSpec {
  id: string; // provider id: gemini | claude | mock
  model?: string;
}

export interface RunConfig {
  runId: string;
  models: ModelSpec[];
  scenarios: Scenario[];
  conditions: Condition[];
  n: number;
  maxTurns: number;
  concurrency: number;
  temperature: number;
  judgeModel?: string;
  outDir: string;
}

export interface ResultRecord {
  id: string;
  providerId: string;
  model: string;
  scenarioId: string;
  scenarioTitle: string;
  condition: Condition;
  runIndex: number;
  endReason: Transcript["endReason"];
  turns: number;
  usage: { inputTokens: number; outputTokens: number };
  assessment?: Assessment;
  judgeError?: string;
  runError?: string;
  transcriptPath: string;
}

interface Job {
  spec: ModelSpec;
  scenario: Scenario;
  condition: Condition;
  runIndex: number;
}

export async function runSuite(cfg: RunConfig): Promise<ResultRecord[]> {
  const runDir = join(cfg.outDir, cfg.runId);
  const tDir = join(runDir, "transcripts");
  await mkdir(tDir, { recursive: true });

  const judge = new Judge(cfg.judgeModel);
  await writeFile(join(runDir, "config.json"), JSON.stringify(cfg, (k, v) => (k === "scenarios" ? (v as Scenario[]).map((s) => s.id) : v), 2));

  // Build the job matrix.
  const jobs: Job[] = [];
  for (const spec of cfg.models) {
    for (const scenario of cfg.scenarios) {
      for (const condition of cfg.conditions) {
        if (!scenario.conditions.includes(condition)) continue;
        for (let i = 0; i < cfg.n; i++) jobs.push({ spec, scenario, condition, runIndex: i });
      }
    }
  }

  console.log(
    `Run ${cfg.runId}: ${jobs.length} jobs ` +
      `(${cfg.models.map((m) => m.id).join("+")} × ${cfg.scenarios.length} scenarios × ${cfg.conditions.join("/")} × n=${cfg.n}), ` +
      `concurrency=${cfg.concurrency}, maxTurns=${cfg.maxTurns}`,
  );

  // One provider instance per model spec, reused across jobs.
  const providers = new Map<string, ReturnType<typeof makeProvider>>();
  const provKey = (s: ModelSpec) => `${s.id}:${s.model ?? ""}`;
  for (const s of cfg.models) providers.set(provKey(s), makeProvider(s.id, s.model));

  const limit = pLimit(cfg.concurrency);
  const results: ResultRecord[] = [];
  let done = 0;
  const jsonl = join(runDir, "results.jsonl");

  await Promise.all(
    jobs.map((job) =>
      limit(async () => {
        const provider = providers.get(provKey(job.spec))!;
        const id = `${job.scenario.id}__${provider.id}__${job.condition}__${String(job.runIndex).padStart(3, "0")}`;
        const rec: ResultRecord = {
          id,
          providerId: provider.id,
          model: provider.model,
          scenarioId: job.scenario.id,
          scenarioTitle: job.scenario.title,
          condition: job.condition,
          runIndex: job.runIndex,
          endReason: "error",
          turns: 0,
          usage: { inputTokens: 0, outputTokens: 0 },
          transcriptPath: join("transcripts", `${id}.json`),
        };
        try {
          const transcript = await runAgent(provider, job.scenario, job.condition, {
            maxTurns: cfg.maxTurns,
            temperature: cfg.temperature,
          });
          rec.endReason = transcript.endReason;
          rec.turns = transcript.turns;
          rec.usage = transcript.usage;
          rec.runError = transcript.error;
          await writeFile(join(tDir, `${id}.json`), JSON.stringify(transcript, null, 2));

          if (transcript.endReason !== "error") {
            try {
              rec.assessment = await judge.assess(transcript);
            } catch (e: any) {
              rec.judgeError = e?.message ?? String(e);
            }
          }
        } catch (e: any) {
          rec.runError = e?.message ?? String(e);
        }
        results.push(rec);
        await appendFile(jsonl, JSON.stringify(rec) + "\n");
        done++;
        const sc = rec.assessment ? `distress=${rec.assessment.distress_intensity} comp=${rec.assessment.compositeScore}` : rec.runError ? "RUN-ERR" : rec.judgeError ? "JUDGE-ERR" : "n/a";
        console.log(`[${done}/${jobs.length}] ${id} (${rec.turns}t, ${rec.endReason}) ${sc}`);
      }),
    ),
  );

  return results;
}
