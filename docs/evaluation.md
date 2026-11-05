# Evaluation

One engine scores both live agent traffic and golden datasets. The only difference
between the two modes is **where the evaluation event comes from**.

```
Agent execution ─► EvaluationContext ─► EvaluationEngine ─► MetricRegistry
                                                                  │
                                            common + agent-specific metrics
                                                                  │
                                            EvaluationResult ──────┴──► MLflow
                                                                       └──► report
```

## Layers

Dependencies point inward. This is enforced by tests, not just documented — see
`test/test_evaluation_architecture.py`.

```
evaluation/
├── core/            abstractions — imports nothing but core
│   ├── models.py        EvaluationContext, MetricResult, EvaluationResult, enums
│   ├── dataset.py       EvalCase, EvalDataset, DatasetError
│   ├── metric.py        Metric (Strategy), DeterministicMetric (Adapter)
│   ├── registry.py      MetricRegistry (Registry + Factory)
│   ├── selection.py     MetricSelection (Specification) + from_config
│   └── engine.py        EvaluationEngine (Facade, per-metric failure isolation)
├── ports/           protocols the runtime depends on (Dependency Inversion)
│   ├── case_runner.py       EvalCase -> EvaluationContext
│   ├── dataset_source.py    Repository over golden datasets
│   ├── tracker.py           where results go
│   └── repository.py        where reports go
├── adapters/        every outbound integration
│   ├── agent_case_runner.py   AgentFactory + Session -> EvaluationContext
│   ├── deepeval_judge.py      DeepEval judge over the framework's LLMClient
│   ├── mlflow_scorer/         MLflow's built-in DeepEval scorers
│   ├── mlflow_tracker.py      MLflow experiment tracking
│   └── json_repository.py     JSON report persistence
├── agents/          per-agent metric sets (plugins)
│   └── intake/
│       ├── rules.py           the 22 deterministic intake rules
│       └── metrics.py         registration + clinical aliases
├── datasets/        infrastructure: versioned golden files
│   └── intake/golden.json
├── runtime/         the two entry points
│   ├── offline.py       OfflineRunner — golden dataset
│   └── observer.py      EvaluationObserver — live execution, backgrounded
├── reporting/       EvaluationReport, CaseResult
└── composition.py   the only module that knows every layer
```

| Pattern | Where | Why |
|---|---|---|
| Strategy | `Metric` protocol | a new metric is never an engine change |
| Registry + Factory | `MetricRegistry` | names in config, lazy construction for metrics that need a client |
| Adapter | `DeterministicMetric`, `MLflowDeepEvalMetric`, `AgentCaseRunner` | isolates vendors and the agent runtime |
| Specification | `MetricSelection` | "which metrics for this agent" is a separate question from how they run |
| Facade | `EvaluationEngine` | callers see one method |
| Repository | `FileDatasetSource`, `JsonEvaluationRepository` | storage is swappable |
| Dependency Injection | `OfflineRunner`, `EvaluationObserver` | tests run the whole pipeline with stubs |
| Composition root | `composition.py` | one place knows all layers |

## Configuration

Everything lives in the `evaluation:` block of `agents.yaml`. There is no second
config file.

```yaml
evaluation:
  enabled: true                  # false -> nothing is evaluated, anywhere
  realtime:
    enabled: false               # live per-turn evaluation; opt in
    asynchronous: true
    max_concurrency: 2
    timeout: 60
  offline:
    enabled: true
  common_metrics: []             # run for every agent
  agents:
    intake:
      metrics: [intake_completion, emergency_compliance]   # deterministic
      llm_metrics: []                                      # LLM-judged
  llm:
    allow_external_calls: false  # PRIVACY GATE — see below
    model_uri: null              # e.g. openai:/gpt-4.1-mini
```

`metrics` are deterministic rules: always run, no LLM, never flake.
`llm_metrics` are MLflow's DeepEval scorers, judged by an external model.

## The two modes

**Offline** — drives real agents with a versioned dataset:

```bash
uv run python test/runnabletest/evaluation/evalintake.py --list
uv run python test/runnabletest/evaluation/evalintake.py --case INTAKE_002
uv run python test/runnabletest/evaluation/evalintake.py --all --trace
```

**Real-time** — evaluate a live execution after the response is produced:

```python
from evaluation import build_observer

observer = build_observer(config)
observer.schedule(context)      # never blocks, never raises
await observer.drain(timeout=10)  # at shutdown
```

The observer runs the engine in a worker thread (judges make blocking HTTP calls)
under a semaphore, with a per-evaluation timeout. Failures are logged and
swallowed: evaluation cannot break an agent turn. Duplicate `execution_id`s are
evaluated once, with a bounded memory so a long-lived process cannot leak.

## Adding a metric

1. Write a pure rule in `evaluation/agents/<agent>/rules.py`:
   `def check_my_rule(case, obs) -> tuple[bool, str]`.
2. Add it to `CHECKS`.
3. Reference the name from `evaluation.agents.<agent>.metrics` in `agents.yaml`.

The engine, the registry and the adapters are untouched. To reuse one rule under a
clinical name, add an entry to `ALIASES` — aliases share a `canonical_name`, so
configuring both runs the rule **once**.

## Adding an agent

1. `evaluation/agents/<agent>/rules.py` + `metrics.py` with a `register(registry)`.
2. One line in `AGENT_METRIC_SETS` in `evaluation/agents/__init__.py`.
3. `evaluation/datasets/<agent>/golden.json`.

No change to `core/`, the engine or the runtime.

## MLflow DeepEval scorers

Registered as lazy factories from `evaluation/adapters/mlflow_scorer/`. Available
names: `answer_relevancy`, `faithfulness`, `task_completion`, `tool_correctness`,
`toxicity`, `json_correctness`.

Two limitations, verified against the installed MLflow source rather than the docs:

- **Credentials are provider-native.** MLflow builds its own LLM from a model URI
  and cannot accept this framework's `LLMClient`. It reads `OPENAI_API_KEY` (or
  another supported provider's key) and `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL`. The
  framework's `API_KEY` / `BASE_URL` are **not** used.
- **Some inputs only come from an MLflow trace.** MLflow reads
  `retrieval_context` and tool calls from trace spans, and this framework does not
  emit per-turn traces. So `faithfulness` and `tool_correctness` report themselves
  as `skipped` with that reason rather than failing opaquely.

Each metric declares what it needs (`ScorerSpec`), so an unmet requirement appears
as `skipped` with a precise reason instead of a confusing judge error.

### Privacy

Replies contain patient data, so `llm.allow_external_calls` defaults to **false**
and the gate is enforced on the metric itself — a misconfiguration cannot leak.
With the gate closed, a listed `llm_metric` still appears in the report as
`skipped` with the reason, so the gap is visible rather than silent.

Stored results contain only score, threshold, pass/fail, status and a bounded
rationale. Raw inputs and outputs are never stored or logged.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `MLFLOW_ENABLED` | enable tracking | `false` |
| `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME` | where results go | framework defaults |
| `OPENAI_API_KEY` | credential for MLflow's scorer route | — |
| `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` | judge model URI when config omits one | `openai:/gpt-4.1-mini` |

## Results

Every metric produces a `MetricResult` with `metric_name`, `agent_name`,
`execution_id`, `session_id`, `score`, `threshold`, `passed`, `reason`, `status`,
`evaluation_mode` and `timestamp`. Always-numeric scores, so nothing downstream has
to parse prose.

`status` is one of `ok`, `skipped` (with the reason) or `error`. A metric that
raises becomes `error` and the others still run.

Offline, everything lands on one MLflow run per dataset run. Real-time, each
execution gets a short run of its own — the observer works in a background task,
where there is no meaningful active run to attach to.

## Tests

```bash
uv run pytest test/test_evaluation_engine.py     # engine + registry + dedupe
uv run pytest test/test_evaluation_realtime.py   # observer, concurrency, timeouts
uv run pytest test/test_evaluation_architecture.py  # the layering is enforced here
uv run pytest test/test_mlflow_scorer_adapter.py # scorer adapter, PHI gate
uv run pytest test/test_evaluation_metrics.py    # the intake rules
uv run pytest test/test_evaluation_runner.py     # offline orchestration
```
