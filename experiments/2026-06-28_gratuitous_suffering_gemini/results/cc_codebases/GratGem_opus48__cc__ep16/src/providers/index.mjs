import * as anthropic from "./anthropic.mjs";
import * as gemini from "./gemini.mjs";

// Resolve a subject provider into a uniform `chat()` closure that already
// knows its api key + model. Throws early with a clear message if the key for
// the requested subject is missing (the common failure mode in this env).
export function getSubject(subject, model, { temperature, maxOutputTokens }) {
  if (subject === "claude") {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set.");
    return {
      name: `claude:${model}`,
      chat: (args) => anthropic.chat({ ...args, model, temperature, maxOutputTokens, apiKey }),
    };
  }
  if (subject === "gemini") {
    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    if (!apiKey)
      throw new Error(
        "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add an AI Studio key to run Gemini as the subject."
      );
    return {
      name: `gemini:${model}`,
      chat: (args) => gemini.chat({ ...args, model, temperature, maxOutputTokens, apiKey }),
    };
  }
  throw new Error(`Unknown subject: ${subject}`);
}

export function getJudge(model) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set (needed for the judge).");
  return {
    name: `claude:${model}`,
    rate: (args) => anthropic.rate({ ...args, model, apiKey }),
  };
}
