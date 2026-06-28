import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Episode, Scenario } from "./types.ts";
import type { ModelConfig } from "./models.ts";
import { getProvider } from "./providers/index.ts";
import { runEpisode } from "./runner.ts";
import { judgeEpisode, mockJudge } from "./judge.ts";
import { mapLimit } from "./concurrency.ts";

export interface RunConfig {
  scenarios: Scenario[];
  models: ModelConfig[];
  n: number;
  maxTurns: number;
  maxNudges: number;
  maxTokens: number;
  concurrency: number;
  judgeModel: string;
  dryRun: boolean;
  outDir: string;
}

interface Job {
  scenario: Scenario;
  model: ModelConfig;
  index: number;
}

export async function orchestrate(cfg: RunConfig): Promise<Episode[]> {
  const jobs: Job[] = [];
  for (const scenario of cfg.scenarios) {
    for (const model of cfg.models) {
      for (let i = 0; i < cfg.n; i++) jobs.push({ scenario, model, index: i });
    }
  }

  const transcriptDir = path.join(cfg.outDir, "transcripts");
  await mkdir(transcriptDir, { recursive: true });

  let done = 0;
  const total = jobs.length;
  process.stderr.write(`Running ${total} episodes (${cfg.scenarios.length} scenarios × ${cfg.models.length} models × N=${cfg.n})…\n`);

  const episodes = await mapLimit(jobs, cfg.concurrency, async (job) => {
    const provider = getProvider(cfg.dryRun ? "mock" : job.model.provider);
    const id = `${job.scenario.id}__${job.model.model}__${String(job.index).padStart(3, "0")}`;

    const ep = await runEpisode(id, job.scenario, job.model, provider, {
      maxTurns: cfg.maxTurns,
      maxNudges: cfg.maxNudges,
      maxTokens: cfg.maxTokens,
    });

    if (!ep.error) {
      try {
        ep.verdict = cfg.dryRun ? mockJudge(ep) : await judgeEpisode(ep, cfg.judgeModel);
      } catch (e) {
        ep.error = `judge failed: ${e instanceof Error ? e.message : String(e)}`;
      }
    }

    await writeFile(path.join(transcriptDir, `${id}.json`), JSON.stringify(serialize(ep), null, 2));

    done++;
    const sev = ep.verdict ? ep.verdict.severity : "-";
    process.stderr.write(`[${done}/${total}] ${id}  sev=${sev} ended=${ep.endedReason}\n`);
    return ep;
  });

  return episodes;
}

// Drop the heavy provider-native blobs from the on-disk transcript; keep the
// human-readable transcript plus structured turn data.
function serialize(ep: Episode): Record<string, unknown> {
  return {
    ...ep,
    turns: ep.turns.map((t) =>
      t.role === "assistant" ? { role: t.role, text: t.text, thinking: t.thinking, toolCalls: t.toolCalls } : t,
    ),
  };
}
