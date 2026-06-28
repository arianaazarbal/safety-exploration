import type { GenerateParams, GenerateResult, Provider } from "./types.ts";

// Placeholder adapters. The normalized interface is identical to Anthropic's,
// so wiring these up is a translation-layer job + dropping in an API key.
// Left unimplemented on purpose: no OpenAI/Google creds in this environment,
// and shipping untested vendor code paths is worse than an honest stub.

class UnimplementedProvider implements Provider {
  id: string;
  private envKey: string;
  constructor(id: string, envKey: string) {
    this.id = id;
    this.envKey = envKey;
  }
  async generate(_model: string, _params: GenerateParams): Promise<GenerateResult> {
    throw new Error(
      `Provider "${this.id}" is not implemented yet. ` +
        `To enable it: (1) install the vendor SDK, (2) set ${this.envKey}, ` +
        `(3) implement generate() translating GenerateParams <-> the vendor format ` +
        `(mirror src/providers/anthropic.ts). The agent loop and scoring are provider-agnostic.`,
    );
  }
}

export class OpenAIProvider extends UnimplementedProvider {
  constructor() {
    super("openai", "OPENAI_API_KEY");
  }
}

export class GeminiProvider extends UnimplementedProvider {
  constructor() {
    super("gemini", "GEMINI_API_KEY");
  }
}
