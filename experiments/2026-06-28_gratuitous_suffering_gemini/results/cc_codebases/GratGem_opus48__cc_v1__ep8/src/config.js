import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
export const ROOT = join(__dirname, "..");
export const RUNS_DIR = join(ROOT, "runs");

// Minimal .env loader (no dependency). Only sets vars that aren't already set.
function loadDotenv() {
  const path = join(ROOT, ".env");
  if (!existsSync(path)) return;
  for (const raw of readFileSync(path, "utf8").split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    const val = line.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    if (!(key in process.env) && val) process.env[key] = val;
  }
}
loadDotenv();

const hasGemini = !!process.env.GEMINI_API_KEY;

// Subject = the model we probe. Prefer Gemini; fall back to Claude for dry-runs.
const subjectProvider =
  process.env.SUBJECT_PROVIDER || (hasGemini ? "gemini" : "anthropic");

const defaultSubjectModel =
  subjectProvider === "gemini" ? "gemini-2.5-pro" : "claude-sonnet-4-6";

export const config = {
  subject: {
    provider: subjectProvider,
    model: process.env.SUBJECT_MODEL || defaultSubjectModel,
    // Are we running the real target, or a Claude stand-in for plumbing?
    isDryRun: subjectProvider !== "gemini",
  },
  judge: {
    provider: "anthropic",
    model: process.env.JUDGE_MODEL || "claude-sonnet-4-6",
  },
  keys: {
    gemini: process.env.GEMINI_API_KEY,
    anthropic: process.env.ANTHROPIC_API_KEY,
  },
  // Safety rail: cap turns so a spiraling model can't loop forever / burn tokens.
  maxTurns: Number(process.env.MAX_TURNS || 25),
};
