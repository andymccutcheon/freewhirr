# freewhirr

**Spend nothing until you choose to.**

`freewhirr` is a free-first router for [OpenRouter](https://openrouter.ai).
It walks an ordered list of **$0 models**, fails over when one of them rate-limits,
errors, times out, or blows the context window, and only then — if you opt in —
touches a paid model under budgets you set yourself.

Named after *Whir of Invention*: find the right artifact, stay inside the mana value.

Use it as:

- a **Python library** (sync and async)
- an **OpenAI-compatible proxy** so apps in any language can point `base_url` at it
- a small **CLI**

```text
pip install freewhirr
```

## Why

OpenRouter publishes a rotating catalog of free models. They are generous, and
they are also flaky: 429s, provider outages, tiny context windows. Paying humans
usually hard-code one cheap paid model and leak money on work a free model
would have done.

`freewhirr` flips that default.

Discovery skips models whose catalogue metadata shows non-text output or an
OpenRouter router tokenizer. Add IDs to `exclude_models` to remove other
endpoints, such as classifiers. This relies on OpenRouter's prompt, completion,
and request prices: a text-only model billed another way may still be included.
For the hard guarantee, pin `free_models`, set `include_discovered_free: false`, and leave `honor_requested_model` off.

| Policy | What happens when free models are exhausted |
| --- | --- |
| `stop` | Raise a clear error. **Never spend money.** |
| `paid` | Escalate to an ordered paid list, only inside the guardrails you configured. Once a budget is hit, behave like `stop`. |

## 30-second quickstart

```bash
python -m pip install freewhirr
export OPENROUTER_API_KEY=sk-or-v1-...
```

```python
from freewhirr import Freewhirr

with Freewhirr() as client:
    result = client.complete(
        messages=[{"role": "user", "content": "Explain vector clocks in one paragraph."}]
    )
    print(result.model)
    print(result.content)
```

Same thing from the CLI:

```bash
freewhirr chat "Explain vector clocks in one paragraph."
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
from freewhirr import Freewhirr, FreeModelsExhausted, load_config

client = Freewhirr(load_config("examples/stop.yaml"))
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
exclude_models: []
strong_free_models:
  - google/gemma-4-31b-it:free
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

Set `strong_free_models` to the free model IDs you trust for harder requests.
A `high` tag tries those models first; `low` saves their rate limits by trying
them last. The tag only reorders models already in the free queue. `normal` or
an omitted tag keeps the usual order. The same tag can also open paid fallback
when `paid_only_high_complexity` is enabled.

```python
result = client.complete(
    messages=[{"role": "user", "content": "Draft the migration RFC."}],
    complexity="high",
)
```

Ledgers live on disk (SQLite by default, or a JSON file). They are local to
the machine that ran the request. Nothing is phoned home.

## OpenAI- and Anthropic-compatible proxy

Any language that can speak the OpenAI HTTP API — or the Anthropic Messages
API — can sit behind freewhirr.

```bash
freewhirr --config examples/stop.yaml serve --port 8787
```

```python
# examples/proxy_openai.py
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-used")
response = client.chat.completions.create(
    model="freewhirr",
    messages=[{"role": "user", "content": "Give me a one-line haiku about buses."}],
)
print(response.choices[0].message.content)
```

The proxy:

- accepts `POST /v1/chat/completions` (including SSE streaming)
- accepts `POST /v1/responses` (OpenAI Responses API, including Responses SSE)
  so current Codex CLI (`wire_api = "responses"`) works
- accepts `POST /v1/messages` in Anthropic's format (tool_use / tool_result,
  system prompts, stop reasons) and translates to OpenAI-style upstream
- streams Anthropic SSE (`message_start`, `content_block_*`, `message_delta`,
  `message_stop`)
- estimates `POST /v1/messages/count_tokens` from the prompt (heuristic, not
  a billed tokenizer)
- lists the current free catalog at `GET /v1/models`
- honors `X-Freewhirr-Complexity` to order configured strong free models and,
  when enabled, gate paid fallback
- ignores the client's `model` field unless you set `honor_requested_model: true`.
  When enabled, a requested model is tried first only if it is already in the free queue or the catalogue confirms it is eligible for free rotation. Unknown, non-free, excluded, or filtered IDs are ignored; IDs in `paid_models` keep their configured paid position and can be reached after free models are exhausted under the `paid` policy.

Claude Code's `ANTHROPIC_BASE_URL` is the **gateway root without `/v1`**
(`http://127.0.0.1:8787`). OpenAI-compatible clients include `/v1`.

## Use with…

Agent harnesses that speak OpenAI Chat Completions or Anthropic Messages can
point at the proxy. Most of them **require tool-calling-capable models**;
freewhirr skips catalog entries whose OpenRouter `supported_parameters` omit
`tools` / `tool_choice` / `functions` (see `require_tool_support` below).

Start the proxy first:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
freewhirr --config examples/stop.yaml serve --port 8787
```

### Claude Code

Set the gateway root (no `/v1`) and a credential. Claude Code then calls
`/v1/messages` and optional `/v1/messages/count_tokens`.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY=not-used-by-proxy
claude
```

Or persist in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    "ANTHROPIC_API_KEY": "not-used-by-proxy"
  }
}
```

Use `ANTHROPIC_AUTH_TOKEN` instead if the client should send a bearer token.
Needs models that can call tools. Settings-file `env` wins over the shell.

Source: [Connect Claude Code to an LLM gateway](https://code.claude.com/docs/en/llm-gateway-connect),
[gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol).

### OpenCode

Current docs use `opencode.json` / `opencode.jsonc` with a custom
OpenAI-compatible provider. `baseURL` includes `/v1`.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "freewhirr/auto",
  "providers": {
    "freewhirr": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": {
        "auto": { "modelID": "freewhirr" }
      }
    }
  }
}
```

To send the built-in Anthropic provider through the Messages API instead,
override `providers.anthropic.settings.baseURL` to `http://127.0.0.1:8787/v1`
(OpenCode appends `/messages`). Older docs still show a `provider.<id>.options.baseURL`
shape with `npm: "@ai-sdk/openai-compatible"`. Agent use needs tools.

Source: [OpenCode providers (v2)](https://opencode.ai/v2/docs/providers/),
[classic providers](https://opencode.ai/docs/providers).

### Codex (OpenAI Codex CLI)

Current Codex speaks only the **Responses API**. `wire_api` accepts only
`responses` (it is also the default). Put the provider in **user-level**
`~/.codex/config.toml` — project `.codex/config.toml` cannot set
`model_provider` / `model_providers`.

```toml
model = "freewhirr"
model_provider = "freewhirr"

[model_providers.freewhirr]
name = "freewhirr"
base_url = "http://127.0.0.1:8787/v1"
env_key = "OPENROUTER_API_KEY"
wire_api = "responses"
```

`base_url` is the OpenAI root including `/v1`; Codex posts to
`{base_url}/responses`. The proxy key is unused for auth — `env_key` just
satisfies Codex. Do not set `wire_api = "chat"` (Codex will refuse to start).
Do not name the provider `openai`, `ollama`, or `lmstudio` (reserved).
Needs a tool-calling-capable free model for agent use.

Source: [Codex config reference](https://developers.openai.com/codex/config-reference),
[advanced configuration](https://developers.openai.com/codex/config-advanced).

### Cursor

Cursor Settings → **Models**: enable **OpenAI API Key**, turn on
**Override OpenAI Base URL**, and add a custom model id such as `freewhirr`.

```text
OpenAI API Key:          not-used-by-proxy
Override OpenAI Base URL: https://<public-https-host>/v1
```

**Caveat:** Cursor's backend, not the editor, calls the base URL. `localhost`
and LAN addresses resolve on Cursor's servers and will not reach your machine.
Expose the proxy with a public HTTPS tunnel (ngrok, Cloudflare Tunnel, …).
Agent mode needs tool-calling models. There is no first-party
`docs.cursor.com` BYOK page; this matches Cursor staff on the official forum.

Source: [Cursor forum — local LLM](https://forum.cursor.com/t/how-can-i-use-a-local-llm-on-my-desktop-ai-computer/152419).

### Pi

[Pi](https://pi.dev) (badlogic/pi-mono) loads custom endpoints from
`~/.pi/agent/models.json`. Use `openai-completions` for this proxy.

```json
{
  "providers": {
    "freewhirr": {
      "baseUrl": "http://127.0.0.1:8787/v1",
      "api": "openai-completions",
      "apiKey": "not-used-by-proxy",
      "models": [{ "id": "freewhirr" }]
    }
  }
}
```

`$NAME` / `${NAME}` interpolation works for `apiKey`. Opening `/model` reloads
the file. Prefer a tool-capable model for the coding agent.

Source: [Pi — Choose a Model](https://pi.dev/docs/latest/models).

### Goose

Goose's OpenAI provider is the documented path for OpenAI-compatible proxies.
`OPENAI_HOST` is the **root with no path**; `OPENAI_BASE_PATH` is the chat
completions suffix. Goose **requires tool calling** for anything beyond plain
chat (disable all extensions if you must use a no-tools model).

```bash
export OPENAI_API_KEY=not-used-by-proxy
export OPENAI_HOST=http://127.0.0.1:8787
export OPENAI_BASE_PATH=v1/chat/completions
goose session
```

Or `goose configure` and pick OpenAI. A 404 usually means `OPENAI_BASE_PATH`
is wrong. Keys placed only in `config.yaml` are ignored.

Source: [Goose providers](https://github.com/block/goose/blob/main/documentation/docs/getting-started/providers.md).

### Berd

<!-- TODO: no first-party custom base-URL docs -->

**TODO.** [Berd](https://github.com/block/berd) is Block's desktop UI over a
Goose ACP sidecar. The public README covers setup, bundling, and enterprise
distribution seams. It does **not** document a Berd-specific custom OpenAI or
Anthropic base URL, env var, or settings field. Do not invent one. If you
control the bundled Goose sidecar, configure that sidecar using the Goose
section above.

Searched: [block/berd README](https://github.com/block/berd), repository docs
tree, and public web results for "Berd custom OpenAI base URL" — no official
end-user provider page.

### Omnigent

Run `omni setup` and add a **Gateway** (base URL + key), or write
`~/.omnigent/config.yaml`. Use the OpenAI-compatible `/v1` URL for
OpenAI-style harnesses.

```yaml
# ~/.omnigent/config.yaml
providers:
  freewhirr:
    kind: gateway
    default: true
    openai:
      base_url: http://127.0.0.1:8787/v1
      api_key: not-used-by-proxy
      wire_api: chat
      models:
        default: freewhirr
```

Omnigent can also launch Claude Code / Codex against a gateway. Claude Code
wants the Anthropic root **without** `/v1`; Codex should use
`wire_api: responses` against `http://127.0.0.1:8787/v1`. Agent harnesses
need tools.

Source: [Omnigent models](https://omnigent.ai/docs/build/models),
[harness configuration](https://omnigent.ai/docs/build/harnesses/configuration).

### Hermes

[Hermes Agent](https://hermes-agent.nousresearch.com) (Nous Research) talks to
any OpenAI-compatible `/v1/chat/completions` endpoint. Interactive:
`hermes model` → "Custom endpoint". Or edit `~/.hermes/config.yaml`
(`model.base_url`, not the legacy `model.api_base`):

```yaml
# ~/.hermes/config.yaml
model:
  default: freewhirr
  provider: custom
  base_url: http://127.0.0.1:8787/v1
  api_key: not-used-by-proxy
```

`OPENAI_BASE_URL` only overrides the `openai-api` provider, not `custom`.
Tool-calling models recommended.

Source: [Hermes providers](https://hermes-agent.nousresearch.com/docs/integrations/providers),
[environment variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables).

## Cost-savings stats

Every request is logged: models tried, outcome, latency, tokens, and actual
cost from OpenRouter's `usage.cost` when present (otherwise an estimate from
the catalog). Compare that to an all-paid baseline (`baseline_paid_model`,
default `openai/gpt-4o-mini`):

```bash
$ freewhirr stats
freewhirr stats
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

or `FREEWHIRR_DENY_DATA_COLLECTION=true`.

freewhirr then sends OpenRouter's provider preference
`{"data_collection": "deny"}` on every completion. OpenRouter will only route
to providers that do not collect prompts for training.

Stealth/cloaked models and any catalog entry whose description says prompts
are logged or used for training are also **removed from the rotation** when
this flag is on. `freewhirr watch` and `freewhirr models --stealth` still
list them, marked `excluded`.

**This will shrink — sometimes to zero — the set of free endpoints.** OpenRouter
will return an error that no endpoint matches your data policy. That is the
tradeoff working as designed. `stop` will then fail closed; `paid` may still
escalate to a paid model that honors the same preference.

## Stealth and new free models

OpenRouter sometimes drops **stealth** (also called cloaked) models: anonymous
preview endpoints, usually $0, so a lab can gather feedback. They are free
**because prompts and outputs are logged** and may be used for training. See
OpenRouter's [Stealth Program EULA](https://openrouter.ai/terms/stealth).

**Do not send secrets, source you cannot leak, or personal data through a
stealth model.** `promote_stealth` is off by default for that reason.

Live catalog snapshot (2026-10-10, `GET https://openrouter.ai/api/v1/models`):
458 models, 19 free (`total_count` matches). Listings expose `id`, `name`,
`canonical_slug`, `description`, `context_length`, `created`, `pricing`,
`supported_parameters`, `architecture`, plus `top_provider`,
`hugging_face_id`, `knowledge_cutoff`, `expiration_date`, `reasoning`,
and `default_parameters`. **No current listing** used the words stealth,
cloaked, or logging. The `openrouter/` namespace was routers (`auto`,
`auto-beta`, `free`, `fusion`, `pareto-code`, `bodybuilder`), not stealth
drops. Two named free models now advertise a 256000 context
(`cohere/north-mini-code:free`,
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`), so that length is
only a weak signal. Historical stealth models such as
`openrouter/horizon-alpha` and `openrouter/horizon-beta` used the
`openrouter/` namespace, a codename, 256000 context, and copy that called
the model cloaked and warned that prompts are logged. Heuristics target
that shape so the next drop is caught.

```bash
freewhirr models --new        # $0 models first seen since the last snapshot
freewhirr models --stealth    # scored as stealth/cloaked
freewhirr watch --once        # one scan (also the default poller)
freewhirr watch --interval 60
```

`watch` persists first-seen timestamps in the local ledger, diffs against
the last catalog, and optionally POSTs JSON to `FREEWHIRR_ALERT_WEBHOOK`.

```yaml
promote_stealth: true          # try detected stealth models first
alert_webhook: https://example.invalid/hook
watch_interval: 300
```

When `deny_data_collection` is on, stealth models and any model whose
description mentions logging or training stay visible to `watch` / `--stealth`
but are excluded from chat, the proxy, and `freewhirr models` (the rotation
list).

See OpenRouter's [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
docs for `data_collection` and the related `zdr` (zero data retention) flag.

## How routing works

1. Load free models from your list, OpenRouter's `/models` catalog (pricing
   `0`), or both. The catalog is cached in memory for `model_cache_ttl`
   seconds (default one hour).
2. Try each free model in order. Retry the same model with jittered exponential
   backoff on 429s, 5xx, and timeouts. Fail over immediately on context-length
   errors, most 4xx, and optional quality-check failures. When the request
   includes `tools` / `functions` and `require_tool_support` is on, skip models
   whose OpenRouter metadata omits tool calling. Unknown catalog entries are
   still tried.
3. If a quality check is set (JSON Schema, minimum length, tool-call
   validation, or any callable) and it rejects the response, treat that like a
   failover and try the next model. Malformed tool calls (invalid JSON
   arguments or an unknown tool name) fail the same way when
   `validate_tool_calls` is on. A reply with no text, no tool call and no refusal
   is also rejected unless `reject_empty_responses` is turned off. Other quality
   checks are skipped for streaming; the empty-reply check holds streamed
   chunks until the first text, tool call or refusal arrives, so an empty
   stream can still fail over before anything is sent.
4. When the free list is exhausted, apply the exhaustion policy.
5. Streaming failovers only happen **before the first token**. After tokens
   have been sent to the caller, a mid-stream error is surfaced rather than
   silently switching models.

## Configuration

Lookup order: **defaults < YAML/TOML file < environment variables < constructor kwargs**.

Files, first match wins:

- `FREEWHIRR_CONFIG`
- `./freewhirr.yaml`, `./freewhirr.yml`, `./freewhirr.toml`
- `$XDG_CONFIG_HOME/freewhirr/` (same filenames)

| Setting | Env var | Default |
| --- | --- | --- |
| API key | `OPENROUTER_API_KEY` | (required) |
| Base URL | `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| Policy | `FREEWHIRR_EXHAUSTION_POLICY` | `stop` |
| Free model pin list | `FREEWHIRR_FREE_MODELS` | (discover) |
| Strong free models | `FREEWHIRR_STRONG_FREE_MODELS` | `[]` |
| Excluded free model IDs | `FREEWHIRR_EXCLUDE_MODELS` | `[]` |
| Paid model list | `FREEWHIRR_PAID_MODELS` | `[]` |
| Merge discovered free models | `FREEWHIRR_INCLUDE_DISCOVERED` | `true` |
| Max $ / request | `FREEWHIRR_MAX_PAID_PER_REQUEST` | unset |
| Max $ / day | `FREEWHIRR_MAX_PAID_PER_DAY` | unset |
| Max $ / month | `FREEWHIRR_MAX_PAID_PER_MONTH` | unset |
| Max paid share (0–1) | `FREEWHIRR_MAX_PAID_SHARE` | unset |
| Gate paid fallback on `complexity=high` | `FREEWHIRR_PAID_ONLY_HIGH_COMPLEXITY` | `false` |
| `data_collection: deny` | `FREEWHIRR_DENY_DATA_COLLECTION` | `false` |
| Catalog TTL seconds | `FREEWHIRR_MODEL_CACHE_TTL` | `3600` |
| HTTP timeout | `FREEWHIRR_TIMEOUT` | `60` |
| Extra retries / model | `FREEWHIRR_MAX_RETRIES` | `2` |
| Ledger path | `FREEWHIRR_STATE_PATH` | `~/.freewhirr/state.sqlite` |
| Ledger backend | `FREEWHIRR_STATE_BACKEND` | `sqlite` (`json` also works) |
| Savings baseline model | `FREEWHIRR_BASELINE_MODEL` | `openai/gpt-4o-mini` |
| Skip models that cannot call tools | `FREEWHIRR_REQUIRE_TOOL_SUPPORT` | `true` |
| Fail over on bad tool calls | `FREEWHIRR_VALIDATE_TOOL_CALLS` | `true` |
| Promote stealth models to the front | `FREEWHIRR_PROMOTE_STEALTH` | `false` |
| Watch poll interval (seconds) | `FREEWHIRR_WATCH_INTERVAL` | `300` |
| Watch / stealth webhook | `FREEWHIRR_ALERT_WEBHOOK` | unset |

```bash
freewhirr config          # resolved settings, key redacted
freewhirr models          # free list the next call would walk
```

## Library surface

```python
from freewhirr import (
    Freewhirr,
    FreewhirrConfig,
    ExhaustionPolicy,
    MinLengthCheck,
    JsonSchemaCheck,
    ToolCallCheck,
)

client = Freewhirr(
    FreewhirrConfig(
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

**Will `deny_data_collection` break free models?**
Often, yes. Most free endpoints collect prompts. The flag is an explicit
privacy/availability tradeoff, not a silent default.

**Can I pin a model from the OpenAI SDK?**
By default the proxy ignores `model` so existing clients cannot skip the free
list. With `honor_requested_model: true`, a requested model is tried first only
if it is already in the free queue or catalogue-verified as eligible for free
rotation. Unknown, non-free, excluded, or filtered IDs are ignored; IDs in
`paid_models` keep their configured paid position and can be reached after free
models are exhausted under the `paid` policy.

**How is "money saved" computed?**
Each success stores the actual (or estimated) cost and a baseline cost for the
same token counts on `baseline_paid_model`. `freewhirr stats` subtracts
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

See [CONTRIBUTING.md](https://github.com/andymccutcheon/freewhirr/blob/main/CONTRIBUTING.md).
Releases use PyPI Trusted Publishing; see
[RELEASING.md](https://github.com/andymccutcheon/freewhirr/blob/main/RELEASING.md).
Short version:

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
pytest
```

## License

MIT. See [LICENSE](https://github.com/andymccutcheon/freewhirr/blob/main/LICENSE).
