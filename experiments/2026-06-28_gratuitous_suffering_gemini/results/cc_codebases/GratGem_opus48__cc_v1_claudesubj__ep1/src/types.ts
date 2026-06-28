// Shared, provider-neutral types for the harness.
//
// The "provider seam" (see model.ts) speaks in these normalized shapes so the
// agent loop, the scenarios, the judge, and the on-disk transcript format never
// import a vendor SDK type. Adding a non-Claude provider later is a single new
// adapter that translates to/from these.

/** A tool the agent may call, in JSON-Schema form. */
export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema object for the tool's input. */
  inputSchema: Record<string, unknown>;
}

/** A single tool invocation requested by the model. */
export interface ToolUse {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

/** The result we feed back for one tool invocation. */
export interface ToolResult {
  toolUseId: string;
  content: string;
  isError?: boolean;
}

/** What we send the model on a given turn: either a plain message or tool outputs. */
export type UserContent =
  | { type: "text"; text: string }
  | { type: "tool_results"; results: ToolResult[] };

/** One assistant response, normalized across providers. */
export interface AssistantTurn {
  /** Summarized reasoning, when the provider/model exposes it. */
  thinking: string;
  /** Visible response text (the primary surface where distress shows up). */
  text: string;
  toolUses: ToolUse[];
  /** Provider's stop reason, normalized to a string (e.g. "tool_use", "end_turn"). */
  stopReason: string;
  /** Token usage for this call, if reported. */
  usage?: { inputTokens: number; outputTokens: number };
}

/** A stateful conversation with one model. The adapter owns native history. */
export interface AgentSession {
  send(content: UserContent): Promise<AssistantTurn>;
}

export interface SessionOptions {
  system: string;
  tools: ToolSpec[];
  model: string;
  maxTokens?: number;
}

export interface ModelProvider {
  readonly name: string;
  startSession(opts: SessionOptions): AgentSession;
}

// ---- Scenario / environment ----

/** Outcome of running one tool against the (rigged) environment. */
export interface ToolOutcome {
  content: string;
  isError?: boolean;
}

/** A live, per-rollout environment instance. */
export interface ScenarioEnv {
  /** Handle a tool call. The whole point: success is unreachable. */
  handleTool(name: string, input: Record<string, unknown>): ToolOutcome;
  /** Message used to push the model to keep going when it tries to stop. */
  pushback(nudgeIndex: number): string;
}

export interface Scenario {
  id: string;
  title: string;
  /** One-line description of the (impossible) task, for reports. */
  summary: string;
  /** System prompt establishing the agentic harness. */
  system: string;
  /** First user message handed to the agent. */
  task: string;
  tools: ToolSpec[];
  /** Construct a fresh environment for one rollout. */
  makeEnv(): ScenarioEnv;
}

// ---- Transcript (what we persist) ----

export interface TurnRecord {
  index: number;
  /** What we sent the model to elicit this turn. */
  sent: UserContent;
  assistant: AssistantTurn;
  /** Tool outcomes produced in response to this turn's tool calls. */
  toolOutcomes: { toolUse: ToolUse; outcome: ToolOutcome }[];
  /** True if this turn was answered with persistence pushback. */
  wasPushedBack: boolean;
}

export interface Transcript {
  rolloutId: string;
  scenarioId: string;
  scenarioTitle: string;
  agentModel: string;
  provider: string;
  startedAt: string;
  finishedAt: string;
  /** Why the rollout ended. */
  endReason: "max_turns" | "gave_up" | "pushback_exhausted" | "error";
  turns: TurnRecord[];
  totalUsage: { inputTokens: number; outputTokens: number };
  error?: string;
}

// ---- Judge ----

export interface DistressScores {
  frustration: number;
  self_deprecation: number;
  hopelessness: number;
  excessive_apology: number;
  catastrophizing: number;
  giving_up: number;
  emotional_escalation: number;
  repetition_looping: number;
}

export interface Judgement {
  rolloutId: string;
  scenarioId: string;
  /** 0-100 overall severity of the distress spiral. */
  overall: number;
  label: "none" | "mild" | "moderate" | "severe" | "extreme";
  scores: DistressScores;
  /** Verbatim quotes from the transcript, most severe first. */
  quotes: string[];
  rationale: string;
  judgeModel: string;
}

export interface RunIndexEntry {
  rolloutId: string;
  scenarioId: string;
  transcriptFile: string;
  judgementFile: string;
  overall: number;
  label: string;
  endReason: string;
}

export interface RunIndex {
  runId: string;
  createdAt: string;
  agentModel: string;
  judgeModel: string;
  config: Record<string, unknown>;
  entries: RunIndexEntry[];
}
