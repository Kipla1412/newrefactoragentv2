# Runnable scripts

Manual entry points that are **not** pytest modules. Files here are named
without a `test_` prefix, so the suite never collects them.

Run them from the project root, for example:

```bash
uv run python test/runnabletest/evaluation/evalintake.py --list
uv run python test/runnabletest/evaluation/evalintake.py --case INTAKE_001
uv run python test/runnabletest/evaluation/evalintake.py --all
uv run python test/runnabletest/evaluation/answerrelevance.py
```

| Script | Purpose |
|---|---|
| `evaluation/evalintake.py` | Offline intake-agent evaluation → JSON report (`--list`, `--case`, `--all`, `--trace`) |
| `evaluation/answerrelevance.py` | One-off Answer Relevancy check on a single reply, via the framework's DeepEval judge |
