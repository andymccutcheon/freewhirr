from freewhirr import NonEmptyResponseCheck


def _payload(content=None, **message):
    msg = {"role": "assistant", "content": content}
    msg.update(message)
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


def test_empty_content_rejected():
    check = NonEmptyResponseCheck()
    assert not check(_payload(""))
    assert not check(_payload(None))
    assert not check(_payload("   \n"))


def test_text_accepted():
    assert NonEmptyResponseCheck()(_payload("42"))


def test_tool_call_without_text_is_valid():
    call = [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assert NonEmptyResponseCheck()(_payload("", tool_calls=call))


def test_refusal_is_valid():
    assert NonEmptyResponseCheck()(_payload("", refusal="I can't help with that"))


import httpx
import respx

from tests.conftest import FREE_A, FREE_B, completion_payload, make_client, mock_completions, mock_models


@respx.mock
def test_router_fails_over_on_empty_reply(tmp_path):
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "")),
            FREE_B: httpx.Response(200, json=completion_payload(FREE_B, "ok")),
        }
    )
    client = make_client(tmp_path, free_models=[FREE_A, FREE_B], include_discovered_free=False)
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    assert result.model == FREE_B


@respx.mock
def test_empty_check_can_be_disabled(tmp_path):
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, ""))})
    client = make_client(
        tmp_path, free_models=[FREE_A], include_discovered_free=False, reject_empty_responses=False
    )
    assert client.complete(messages=[{"role": "user", "content": "hi"}]).model == FREE_A
