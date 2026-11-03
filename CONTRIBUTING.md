# Contributing

## Development setup

Python 3.11 or later. Pixeltable does not support 3.10.

```bash
pip install -e ".[dev]"
```

`openai-agents` is pinned to a git commit. The session-adapter helpers this package calls are not in the PyPI 0.24.0 release.

## Tests and lint

```bash
pytest tests/ -v
ruff check . && ruff format --check .
mypy openai_agents_pixeltable
```

Tests use a Pixeltable catalog under a temp directory, not `~/.pixeltable`. They run an agent with a scripted model and do not call an API.

## Releasing

Do not publish while `openai-agents` is pinned to a git commit. PyPI rejects a distribution whose metadata declares a direct URL dependency. Publish only after the session helpers from [openai/openai-agents-python#5428](https://github.com/openai/openai-agents-python/pull/5428) are in a PyPI release of `openai-agents` and this package depends on that release.
