"""Smallest library example. Requires OPENROUTER_API_KEY."""

from freeloader import Freeloader

with Freeloader() as client:
    result = client.complete(messages=[{"role": "user", "content": "Say hello in one sentence."}])
    print(result.model, result.content)
    print("cost", result.cost_usd, "tried", result.models_tried)
