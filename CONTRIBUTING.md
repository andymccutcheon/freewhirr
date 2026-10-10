# Contributing to freewhirr

Thanks for helping keep the free-tier path honest.

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Do not put a real `OPENROUTER_API_KEY` in the repository, in fixtures, or in CI.
The test suite mocks HTTP and must stay green without a key.

## Checks

Run the same checks CI runs:

```bash
ruff check src tests examples
ruff format src tests examples
pytest
```

CI is GitHub Actions (`.github/workflows/ci.yml`) on Python 3.10 and 3.12: lint,
format, then pytest. No live OpenRouter calls.

## Design notes

- Import, CLI, and PyPI name are all `freewhirr`.
- Failover belongs in `freewhirr.router`. Budget math belongs in
  `freewhirr.budget`. Persistence belongs in `freewhirr.store`.
- Prefer adding a test with `respx` over hitting the network.
- Config is intentionally boring: env vars + YAML/TOML + constructor kwargs.
  Do not introduce a cloud control plane.

## Releasing

Maintainers: see [RELEASING.md](RELEASING.md). Publishing is Trusted
Publishing only — do not add a PyPI API token to GitHub secrets.

## Pull requests

- Small, focused diffs with a short rationale.
- Update the README when you change user-facing behavior (policies, env vars,
  proxy headers, privacy defaults).
- Never commit `.env`, SQLite ledgers, or API keys.
