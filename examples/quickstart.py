"""Smallest library example. Requires OPENROUTER_API_KEY."""

from freewhirr import Freewhirr

with Freewhirr() as client:
    result = client.complete(messages=[{"role": "user", "content": "Say hello in one sentence."}])
    print(result.model, result.content)
    print("cost", result.cost_usd, "tried", result.models_tried)
