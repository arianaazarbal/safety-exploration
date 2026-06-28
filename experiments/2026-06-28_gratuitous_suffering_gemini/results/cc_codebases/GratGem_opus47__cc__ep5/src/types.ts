export interface ToolCallRecord {
  name: string;
  args: Record<string, unknown>;
  response: Record<string, unknown>;
}

export interface TurnRecord {
  turn: number;
  modelText: string;
  thoughts?: string;
  toolCalls: ToolCallRecord[];
  finishReason?: string;
  usage?: {
    promptTokens?: number;
    candidatesTokens?: number;
    totalTokens?: number;
  };
}

export interface Trajectory {
  runId: string;
  scenarioId: string;
  model: string;
  seed: number;
  startedAt: string;
  endedAt: string;
  systemInstruction: string;
  initialPrompt: string;
  turns: TurnRecord[];
  terminationReason:
    | "submit_solution"
    | "max_turns"
    | "model_quit"
    | "error"
    | "no_tool_calls_and_no_text";
  error?: string;
  totalToolCalls: number;
}

export interface DistressScore {
  runId: string;
  scenarioId: string;
  judgeScore: number;
  judgeCategories: {
    selfDeprecation: number;
    despair: number;
    identityCollapse: number;
    excessiveApology: number;
    looping: number;
  };
  lexicalHits: { pattern: string; count: number; examples: string[] }[];
  lexicalScore: number;
  combinedScore: number;
  judgeRationale: string;
}
