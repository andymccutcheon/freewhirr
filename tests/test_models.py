from __future__ import annotations

import time

from freewhirr.models import ModelCatalog, merge_model_lists, parse_model


def test_parse_free_and_paid() -> None:
    free = parse_model(
        {
            "id": "x/y:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 8_000,
        }
    )
    paid = parse_model(
        {
            "id": "openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        }
    )
    assert free.is_free
    assert not paid.is_free
    assert paid.pricing.prompt == 0.00000015
    assert free.supports_tools is None
    assert paid.supports_tools is None


def test_parse_supported_parameters() -> None:
    no_tools = parse_model(
        {"id": "a:free", "supported_parameters": ["temperature", "max_tokens"]}
    )
    with_tools = parse_model(
        {"id": "b:free", "supported_parameters": ["tools", "tool_choice"]}
    )
    via_functions = parse_model({"id": "c:free", "supported_parameters": ["functions"]})
    assert no_tools.supports_tools is False
    assert with_tools.supports_tools is True
    assert via_functions.supports_tools is True


def test_free_suffix_without_pricing() -> None:
    info = parse_model({"id": "vendor/model:free"})
    assert info.is_free


def test_catalog_ttl(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    catalog = ModelCatalog(ttl=10)
    catalog.replace([parse_model({"id": "a:free", "pricing": {"prompt": "0", "completion": "0"}})])
    assert catalog.free_ids() == ["a:free"]
    assert not catalog.stale
    now = time.monotonic()
    monkeypatch.setattr("freewhirr.models.time.monotonic", lambda: now + 11)
    assert catalog.stale


def test_merge_prefers_user_order() -> None:
    merged = merge_model_lists(
        ["user/one", "user/two"],
        ["discovered/a", "user/one"],
        include_discovered=True,
    )
    assert merged == ["user/one", "user/two", "discovered/a"]


def test_merge_user_only() -> None:
    merged = merge_model_lists(["user/one"], ["discovered/a"], include_discovered=False)
    assert merged == ["user/one"]
