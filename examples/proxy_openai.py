"""Point the official OpenAI SDK at a local freeloader proxy.

Start the proxy in another terminal first:

    export OPENROUTER_API_KEY=sk-or-...
    freeloader --config examples/stop.yaml serve --port 8787

Then run this file. The SDK talks HTTP to localhost; freeloader talks to
OpenRouter and walks the free-model list.
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-used-by-proxy")
response = client.chat.completions.create(
    model="freeloader",
    messages=[{"role": "user", "content": "Give me a one-line haiku about buses."}],
)
print(response.model)
print(response.choices[0].message.content)

# Tag a request as high-complexity so a `paid` policy with
# paid_only_high_complexity can escalate after free models fail.
client.chat.completions.create(
    model="freeloader",
    messages=[{"role": "user", "content": "Design a 12-factor migration plan."}],
    extra_headers={"X-Freeloader-Complexity": "high"},
)
