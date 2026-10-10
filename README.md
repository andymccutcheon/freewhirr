# freeloader

**Use every free LLM before you pay a cent.**

`freeloader` is a free-first router for [OpenRouter](https://openrouter.ai).
It walks an ordered list of **$0 models**, fails over when one of them rate-limits,
errors, times out, or blows the context window, and only then — if you opt in —
touches a paid model under budgets you set yourself.

Use it as:

- a **Python library** (sync and async)
- an **OpenAI-compatible proxy** so apps in any language can point `base_url` at it
- a small **CLI**

```text
pip install freeloader-llm
```

> **PyPI name.** The import and CLI stay `freeloader`. The distribution name is
> `freeloader-llm` because [`freeloader`](https://pypi.org/project/freeloader/)
> is already taken by an unrelated Terraform CLI.

## Why

OpenRouter publishes a rotating catalog of free models. They are generous, and
they are also flaky: 429s, provider outages, tiny context windows. Paying humans
usually hard-code one cheap paid model and leak money on work a free model
would have done.

`freeloader` flips that default.

| Policy | What happens when free models are exhausted |
| --- | --- |
| `stop` | Raise a clear error. **Never spend money.** |
| `paid` | Escalate to an ordered paid list, only inside the guardrails you configured. Once a budget is hit, behave like `stop`. |

## 30-second quickstart

```bash
python -m pip install freeloader-llm
export OPENROUTER_API_KEY=sk-or-v1-...
```

```python
from freeloader import Freeloader

with Freeloader() as client:
    result = client.complete(
        messages=[{"role": "user", "content": "Explain vector clocks in one paragraph."}]
    )
    print(result.model)
    print(result.content)
```

Same thing from the CLI:

```bash
freeloader chat "Explain vector clocks in one paragraph."
```

The API key is read from `OPENROUTER_API_KEY`. There is no other vendor lock-in:
settings also come from a YAML or TOML file, environment variables, or
constructor kwargs.

## Exhaustion policies

### `stop` — never spend money

```yaml
# examples/stop.yaml
exhaustion_policy: stop
include_discovered_free: true
```

```python
from freeloader import Freeloader, FreeModelsExhausted, load_config

client = Freeloader(load_config("examples/stop.yaml"))
try:
    client.complete(messages=[{"role": "user", "content": "hi"}])
except FreeModelsExhausted as exc:
    print("free tier exhausted; nothing was billed")
    print(exc.attempts)
```

### `paid` — escalate with guardrails

```yaml
# examples/paid.yaml
exhaustion_policy: paid
include_discovered_free: true
paid_models:
  - openai/gpt-4o-mini
max_paid_per_request: 0.05
max_paid_per_day: 1.00
max_paid_per_month: 10.00
max_paid_share: 0.1          # at most 10% of successful requests
paid_only_high_complexity: true
```

Paid fallback is skipped — and the call fails like `stop` — when any of these
are true:

- the daily or monthly spend ledger would go over the cap
- the model's estimated worst-case cost exceeds `max_paid_per_request`
- another paid success would push the paid share above `max_paid_share`
- `paid_only_high_complexity` is on and the request was not tagged `high`

```python
result = client.complete(
    messages=[{"role": "user", "content": "Draft the migration RFC."}],
    complexity="high",
)
```

Ledgers live on disk (SQLite by default, or a JSON file). They are local to
the machine that ran the request. Nothing is phoned home.

## OpenAI-compatible proxy

Any language that can speak the OpenAI HTTP API can sit behind freeloader.

```bash
freeloader --config examples/stop.yaml serve --port 8787
```

```python
# examples/proxy_openai.py
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-used")
response = client.chat.completions.create(
    model="freeloader",
    messages=[{"role": "user", "content": "Give me a one-line haiku about buses."}],
)
print(response.choices[0].message.content)
```

The proxy:

- accepts `POST /v1/chat/completions` (including SSE streaming)
- lists the current free catalog at `GET /v1/models`
- honors `X-Freeloader-Complexity: high` for the paid-complexity gate
- ignores the client's `model` field unless you set `honor_requested_model: true`
  (so a hardcoded `gpt-4o-mini` in an app cannot accidentally skip the free list)

## Cost-savings stats

Every request is logged: models tried, outcome, latency, tokens, and actual
cost from OpenRouter's `usage.cost` when present (otherwise an estimate from
the catalog). Compare that to an all-paid baseline (`baseline_paid_model`,
default `openai/gpt-4o-mini`):

```bash
$ freeloader stats
freeloader stats
  Requests:          142
  Free successes:    131 (94.2%)
  Paid successes:    8 (5.8%)
  Failures:          3
  Spent:             $0.0410
  All-paid baseline: $1.8800
  Estimated saved:   $1.8390
```

`Estimated saved` is `baseline − spent`. It is an accounting aid, not an
invoice.

## Privacy

Most free OpenRouter endpoints may train on prompts. If that is unacceptable,
set:

```yaml
deny_data_collection: true
```

or `FREELOADER_DENY_DATA_COLLECTION=true`.

freeloader then sends OpenRouter's provider preference
`{"data_collection": "deny"}` on every completion. OpenRouter will only route
to providers that do not collect prompts for training.

**This will shrink — sometimes to zero — the set of free endpoints.** OpenRouter
will return an error that no endpoint matches your data policy. That is the
tradeoff working as designed. `stop` will then fail closed; `paid` may still
escalate to a paid model that honors the same preference.

See OpenRouter's [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
docs for `data_collection` and the related `zdr` (zero data retention) flag.

## How routing works

1. Load free models from your list, OpenRouter's `/models` catalog (pricing
   `0`), or both. The catalog is cached in memory for `model_cache_ttl`
   seconds (default one hour).
2. Try each free model in order. Retry the same model with jittered exponential
   backoff on 429s, 5xx, and timeouts. Fail over immediately on context-length
   errors, most 4xx, and optional quality-check failures.
3. If a quality check is set (JSON Schema, minimum length, or any callable)
   and it rejects the response, treat that like a failover and try the next
   model. Quality checks are skipped for streaming.
4. When the free list is exhausted, apply the exhaustion policy.
5. Streaming failovers only happen **before the first token**. After tokens
   have been sent to the caller, a mid-stream error is surfaced rather than
   silently switching models.

## Configuration

Lookup order: **defaults < YAML/TOML file < environment variables < constructor kwargs**.

Files, first match wins:

- `FREELOADER_CONFIG`
- `./freeloader.yaml`, `./freeloader.yml`, `./freeloader.toml`
- `$XDG_CONFIG_HOME/freeloader/` (same filenames)

| Setting | Env var | Default |
| --- | --- | --- |
| API key | `OPENROUTER_API_KEY` | (required) |
| Base URL | `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| Policy | `FREELOADER_EXHAUSTION_POLICY` | `stop` |
| Free model pin list | `FREELOADER_FREE_MODELS` | (discover) |
| Paid model list | `FREELOADER_PAID_MODELS` | `[]` |
| Merge discovered free models | `FREELOADER_INCLUDE_DISCOVERED` | `true` |
| Max $ / request | `FREELOADER_MAX_PAID_PER_REQUEST` | unset |
| Max $ / day | `FREELOADER_MAX_PAID_PER_DAY` | unset |
| Max $ / month | `FREELOADER_MAX_PAID_PER_MONTH` | unset |
| Max paid share (0–1) | `FREELOADER_MAX_PAID_SHARE` | unset |
| Paid only if `complexity=high` | `FREELOADER_PAID_ONLY_HIGH_COMPLEXITY` | `false` |
| `data_collection: deny` | `FREELOADER_DENY_DATA_COLLECTION` | `false` |
| Catalog TTL seconds | `FREELOADER_MODEL_CACHE_TTL` | `3600` |
| HTTP timeout | `FREELOADER_TIMEOUT` | `60` |
| Extra retries / model | `FREELOADER_MAX_RETRIES` | `2` |
| Ledger path | `FREELOADER_STATE_PATH` | `~/.freeloader/state.sqlite` |
| Ledger backend | `FREELOADER_STATE_BACKEND` | `sqlite` (`json` also works) |
| Savings baseline model | `FREELOADER_BASELINE_MODEL` | `openai/gpt-4o-mini` |

```bash
freeloader config          # resolved settings, key redacted
freeloader models          # free list the next call would walk
```

## Library surface

```python
from freeloader import (
    Freeloader,
    FreeloaderConfig,
    ExhaustionPolicy,
    MinLengthCheck,
    JsonSchemaCheck,
)

client = Freeloader(
    FreeloaderConfig(
        api_key="...",
        exhaustion_policy=ExhaustionPolicy.STOP,
    )
)

result = client.complete(messages=[...], quality=MinLengthCheck(40))
async_result = await client.acomplete(messages=[...])

for chunk in client.stream(messages=[...]):
    print(chunk["choices"][0]["delta"].get("content") or "", end="")

# OpenAI-shaped namespace
client.chat.completions.create(messages=[...])
```

`ChatResult` exposes `.content`, `.model`, `.cost_usd`, `.usage`,
`.models_tried`, `.paid`, and `.to_openai()`.

## FAQ

**Does this make live OpenRouter calls in CI?**
No. Tests mock HTTP with `respx`. You can run `pytest` without a key.

**Is the PyPI name `freeloader`?**
No. Install `freeloader-llm`. Import `freeloader`. The upstream name was
already occupied.

**Will `deny_data_collection` break free models?**
Often, yes. Most free endpoints collect prompts. The flag is an explicit
privacy/availability tradeoff, not a silent default.

**Can I pin a model from the OpenAI SDK?**
By default the proxy ignores `model` so existing clients cannot skip the free
list. Set `honor_requested_model: true` if you want a requested free model
tried first.

**How is "money saved" computed?**
Each success stores the actual (or estimated) cost and a baseline cost for the
same token counts on `baseline_paid_model`. `freeloader stats` subtracts
those. If you never configured a paid baseline in the catalog, a GPT-4o-mini
price is used so the number is still in the right order of magnitude.

**What happens if the catalog request fails?**
If you supplied `free_models`, those are still used. If you rely on discovery
alone, the call fails with a clear error — it does not fall through to paid
unless you listed paid models and the policy is `paid`.

**Does streaming support failover?**
Yes, until the first token is forwarded. After that, switching models would
corrupt the stream, so the error is returned.

**Where should I store the API key?**
Environment variable or a gitignored local config file. Never commit it.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version:

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
pytest
```

## License

MIT. See [LICENSE](LICENSE).
