"""Point an Anthropic-API client at a local freewhirr proxy.

Start the proxy in another terminal first:

    export OPENROUTER_API_KEY=sk-or-...
    freewhirr --config examples/stop.yaml serve --port 8787

Claude Code and other Anthropic-API clients set ANTHROPIC_BASE_URL to the
proxy root (no /v1). They then POST /v1/messages themselves.
"""

from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8787",
    api_key="not-used-by-proxy",
)
response = client.messages.create(
    model="freewhirr",
    max_tokens=64,
    messages=[{"role": "user", "content": "Give me a one-line haiku about buses."}],
)
print(response.model)
print("".join(block.text for block in response.content if block.type == "text"))
