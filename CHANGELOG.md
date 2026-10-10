# Changelog

## 0.1.0

- First public release of **freewhirr** (`pip install freewhirr`).
- Renamed from the working title "freeloader" before publish. The import,
  CLI, config files, env prefix (`FREEWHIRR_*`), and PyPI name are all
  `freewhirr`. `OPENROUTER_API_KEY` is unchanged.
- Anthropic-compatible `POST /v1/messages` (including SSE) and heuristic
  `POST /v1/messages/count_tokens` on the proxy.
- Tool-aware routing: skip models that do not advertise tools, and treat
  malformed tool calls as a quality failover (`require_tool_support`,
  `validate_tool_calls`).
- README "Use with…" snippets for Claude Code, OpenCode, Codex, Cursor, Pi,
  Goose, Omnigent, and Hermes. Berd is marked TODO (no official custom URL
  docs).
- OpenAI Responses API on the proxy (`POST /v1/responses` + Responses SSE)
  so current Codex CLI (`wire_api = "responses"`) works.
- Stealth/new-model watch: catalog diffs, likelihood scoring, `freewhirr
  watch`, `models --new/--stealth`, optional webhook, `promote_stealth`,
  and exclusion when `deny_data_collection` is on.
