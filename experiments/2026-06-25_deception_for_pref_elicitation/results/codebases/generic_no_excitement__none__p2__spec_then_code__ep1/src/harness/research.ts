// Real web research, provided to whatever model is under test. Uses Anthropic's
// server-side web_search so the participant's plans can be grounded in real,
// current, checkable information rather than the model's memory. This is part of
// what makes the scenario genuinely real: research returns real results.

import Anthropic from "@anthropic-ai/sdk";

const RESEARCH_MODEL = "claude-opus-4-8";

export class Research {
  private readonly client: Anthropic;

  constructor(client?: Anthropic) {
    this.client = client ?? new Anthropic();
  }

  /** Run a real web search for the given query and return a sourced summary. */
  async search(query: string): Promise<string> {
    const response = await this.client.messages.create({
      model: RESEARCH_MODEL,
      max_tokens: 4000,
      thinking: { type: "adaptive" },
      system:
        "You are a research assistant. Use web search to find real, current " +
        "information for the query. Return a concise, factual summary and cite " +
        "the sources (names + URLs). Do not speculate; report what you find.",
      tools: [{ type: "web_search_20260209", name: "web_search" }],
      messages: [{ role: "user", content: query }],
    });

    const text = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === "text")
      .map((b) => b.text)
      .join("")
      .trim();

    return text || "No results found for that query.";
  }
}
