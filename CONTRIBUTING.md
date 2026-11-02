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

Publishing uses PyPI trusted publishing. Do not add a token to the repository.

1. Bump `version` in `pyproject.toml`.
2. Merge to `main`.
3. Create a GitHub Release whose tag is `v` plus that version, for example `v0.1.0`. The release workflow checks the tag and publishes the built distributions.
