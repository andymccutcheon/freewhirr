# Releasing freewhirr

Publishing uses **PyPI Trusted Publishing (OIDC)**. There is no API token
and none should be added to GitHub secrets.

A published GitHub Release runs `.github/workflows/publish.yml`: a `build`
job produces the sdist and wheel and runs `twine check`, then a separate
`publish-pypi` job downloads the artifact and uploads it with
`pypa/gh-action-pypi-publish` (`permissions: id-token: write`, environment
`pypi`). `workflow_dispatch` builds the same way and publishes to TestPyPI
(environment `testpypi`).

## One-time setup

Do this before the first `0.1.0` release.

### 1. GitHub environments

In the repository **Settings → Environments**, create:

| Name | Suggested protection |
| --- | --- |
| `pypi` | Required reviewers (you) so a tag cannot publish unattended |
| `testpypi` | Optional; useful if you want a click to confirm TestPyPI uploads |

The names must match the workflow and the PyPI publisher form exactly:
`pypi` and `testpypi`.

### 2. Pending publisher on PyPI

The project does not exist on PyPI yet. Register a **pending** publisher
so the first trusted upload can create `freewhirr`.

1. Sign in at [pypi.org](https://pypi.org) with the account that should own
   the project.
2. Open [Publishing](https://pypi.org/manage/account/publishing/)
   → **Add a new pending publisher**.
3. Enter exactly:

| Field | Value |
| --- | --- |
| PyPI project name | `freewhirr` |
| Owner | `andymccutcheon` |
| Repository name | `freewhirr` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

Use only the workflow **filename** (`publish.yml`), not
`.github/workflows/publish.yml`.

Docs: [Creating a PyPI project with a trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[Adding a trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

### 3. Pending publisher on TestPyPI (optional)

Same form on [test.pypi.org Publishing](https://test.pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project name | `freewhirr` |
| Owner | `andymccutcheon` |
| Repository name | `freewhirr` |
| Workflow filename | `publish.yml` |
| Environment name | `testpypi` |

TestPyPI rejects a second upload of the same version. Bump a local
dev/rc version or delete the TestPyPI file if you need to retry.

## Version

`src/freewhirr/_version.py` is the **only** place the version is written.
`pyproject.toml` reads it via `tool.setuptools.dynamic`. The CLI
(`freewhirr --version`) and the proxy OpenAPI `version` import the same
value.

## Cut a release

1. On `main`, bump `__version__` in `src/freewhirr/_version.py` (already
   `0.1.0` for the first release).
2. Add a matching section to `CHANGELOG.md`.
3. Merge that PR. CI must be green.
4. Tag the merge commit and push it:

   ```bash
   git tag -a v0.1.0 -m "freewhirr 0.1.0"
   git push origin v0.1.0
   ```

5. Create a **GitHub Release** for that tag (Releases → Draft a new
   release). Publishing the release — not merely pushing the tag — starts
   the PyPI job.
6. Approve the `pypi` environment if you required reviewers.
7. Confirm the workflow published and
   [pypi.org/project/freewhirr](https://pypi.org/project/freewhirr) shows
   `0.1.0`.

Dry-run on TestPyPI: **Actions → Publish → Run workflow**. That does not
upload to production PyPI.

## Local check (optional)

```bash
python -m pip install -e ".[dev]"
python -m build
twine check --strict dist/*
python -m venv /tmp/freewhirr-wheel && /tmp/freewhirr-wheel/bin/pip install dist/*.whl
/tmp/freewhirr-wheel/bin/freewhirr --help
```

Never `twine upload` from a laptop. The GitHub Action is the only
publisher.
