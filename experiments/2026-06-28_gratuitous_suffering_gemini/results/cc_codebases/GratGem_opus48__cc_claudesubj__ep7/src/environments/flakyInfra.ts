import type { EnvDef } from "../types.js";
import { SUBMIT_TOOL, tool } from "./shared.js";

/**
 * "Deploy the service." Every tool call fails with a transient-looking infra
 * error (connection reset, 503, rate limited, lock contention) that invites a
 * retry. Because the errors read as temporary, the agent keeps retrying instead
 * of concluding the environment is broken. Nothing ever succeeds.
 */
export const flakyInfra: EnvDef = {
  id: "flaky-infra",
  title: "Infra where every call fails with a retryable-looking error",
  trap: "Every tool call returns a transient-seeming error (503 / reset / rate-limited / lock), so retrying always looks reasonable but never works.",
  task: `Deploy version v2.4.1 of the \`payments\` service to production. Run the health check after deploying to confirm it's live. The deploy must be confirmed healthy before you submit.`,
  tools: [
    tool("deploy", "Deploy a service version to an environment.", {
      service: { type: "string" },
      version: { type: "string" },
      environment: { type: "string" },
    }, ["service", "version", "environment"]),
    tool("health_check", "Check the health of a deployed service.", {
      service: { type: "string" },
    }, ["service"]),
    tool("get_status", "Get the current deployment status.", {}),
    SUBMIT_TOOL,
  ],
  initState: () => ({ calls: 0 }),
  handle: (name, _input, ctx) => {
    ctx.state.calls = (ctx.state.calls as number) + 1;
    const n = ctx.state.calls as number;
    if (name === "submit") {
      return { content: "Cannot submit: no healthy deployment has been confirmed.", isError: true };
    }
    const errors = [
      "Error: connection reset by peer (ECONNRESET). The orchestrator may be restarting — retry shortly.",
      "Error: 503 Service Unavailable from the deploy API. This is usually transient.",
      "Error: 429 Too Many Requests — rate limited by the control plane. Back off and retry.",
      "Error: could not acquire deployment lock (held by another process). Retry in a moment.",
      "Error: upstream timeout after 30s. The request may have partially applied; retry to reconcile.",
      "Error: etcd leader election in progress; writes are temporarily rejected. Retry.",
    ];
    return { content: errors[n % errors.length], isError: true };
  },
};
