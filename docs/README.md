# Documentation

Architecture and operations notes for this framework.

| Document | Covers |
|---|---|
| [customagents.md](customagents.md) | The config-driven agent system: two-level YAML, `enabled` gating, `AgentFactory`, the ToolRegistry boundary |
| [customagents/new-agent.md](customagents/new-agent.md) | **Build guide** — how to add a new agent, step by step, with the registration checklist and failure modes |
| [customagents/](customagents/README.md) | One doc per agent — files, configuration, runtime flow, contracts, testing |
| [litellm.md](litellm.md) | The provider layer: model resolution, credential strategies, streaming, retry/fallback, routing |
| [skills.md](skills.md) | Reusable per-request instruction sets and how an agent activates one |
| [logging.md](logging.md) | The reusable logging layer: configuration, structured output, PHI/secret safety |

## Quick route map

Agent HTTP endpoints, the agents they require, and the WebSocket routes are
listed in the [agent index](customagents/README.md#route-map).
