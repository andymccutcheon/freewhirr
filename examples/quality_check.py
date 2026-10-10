"""Fail over when a response does not match a JSON schema."""

from freeloader import Freeloader, JsonSchemaCheck, MinLengthCheck

schema = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}

with Freeloader() as client:
    result = client.complete(
        messages=[
            {
                "role": "user",
                "content": 'Reply with JSON only: {"answer": "..."}',
            }
        ],
        quality=JsonSchemaCheck(schema),
    )
    print(result.content)

    # Quality checks can be stacked by wrapping them:
    def both(payload: dict) -> bool:
        return MinLengthCheck(8)(payload) and JsonSchemaCheck(schema)(payload)

    result = client.complete(
        messages=[{"role": "user", "content": 'JSON please: {"answer": "ok"}'}],
        quality=both,
    )
    print(result.model)
