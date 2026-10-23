from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.pricing import litellm_pricing


def test_litellm_helpers_when_dependency_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(litellm_pricing, "litellm", None)

    assert litellm_pricing.get_litellm_model_cost() == {}
    assert litellm_pricing.get_model_pricing("gpt-4o") is None
    assert litellm_pricing.estimate_cost("gpt-4o", input_tokens=1, output_tokens=1) is None
    assert litellm_pricing.list_available_models() == []


def test_litellm_model_pricing_exact_match_and_defaults(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(
        model_cost={
            "gpt-4o": {
                "input_cost_per_token": 0.0000025,
                "output_cost_per_token": 0.00001,
                "max_tokens": 128000,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    assert litellm_pricing.get_litellm_model_cost() == fake_litellm.model_cost
    pricing = litellm_pricing.get_model_pricing("gpt-4o")
    assert pricing is not None
    assert pricing.model == "gpt-4o"
    assert pricing.input_cost_per_1m == 2.5
    assert pricing.output_cost_per_1m == 10.0
    assert pricing.max_tokens == 128000
    assert pricing.max_input_tokens is None
    assert pricing.max_output_tokens is None
    assert pricing.supports_vision is False
    assert pricing.supports_function_calling is False
    assert (
        litellm_pricing.estimate_cost("gpt-4o", input_tokens=200_000, output_tokens=300_000) == 3.5
    )
    assert litellm_pricing.list_available_models() == ["gpt-4o"]


def test_litellm_model_pricing_uses_provider_prefixes(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(
        model_cost={
            "openai/gpt-4o-mini": {
                "input_cost_per_token": 0.00000015,
                "output_cost_per_token": 0.0000006,
                "supports_vision": True,
                "supports_function_calling": True,
                "max_input_tokens": 64000,
                "max_output_tokens": 16000,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    pricing = litellm_pricing.get_model_pricing("gpt-4o-mini")
    assert pricing is not None
    assert pricing.input_cost_per_1m == 0.15
    assert pricing.output_cost_per_1m == 0.6
    assert pricing.max_input_tokens == 64000
    assert pricing.max_output_tokens == 16000
    assert pricing.supports_vision is True
    assert pricing.supports_function_calling is True


def test_litellm_model_pricing_uses_aliases_and_zero_cost_defaults(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(
        model_cost={
            "claude-sonnet-4-5": {
                "input_cost_per_token": None,
                "output_cost_per_token": None,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    pricing = litellm_pricing.get_model_pricing("claude-3-5-sonnet-20241022")
    assert pricing is not None
    assert pricing.model == "claude-3-5-sonnet-20241022"
    assert pricing.input_cost_per_1m == 0
    assert pricing.output_cost_per_1m == 0
    assert litellm_pricing.estimate_cost("claude-3-5-sonnet-20241022", input_tokens=1) == 0


def test_litellm_pricing_aliases_removed_claude_sonnet_key(monkeypatch) -> None:
    def cost_per_token(*, model, **kwargs):
        if model in {"claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"}:
            return (0.000003, 0.000015)
        raise ValueError("unknown model")

    fake_litellm = SimpleNamespace(
        model_cost={
            "claude-sonnet-4-5": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "anthropic/claude-sonnet-4-5": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
        },
        cost_per_token=cost_per_token,
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)
    monkeypatch.setattr(litellm_pricing, "_resolved_model_cache", {})

    assert litellm_pricing.resolve_litellm_model("claude-sonnet-4-20250514") == "claude-sonnet-4-5"
    assert (
        litellm_pricing.resolve_litellm_model("anthropic/claude-sonnet-4-20250514")
        == "anthropic/claude-sonnet-4-5"
    )
    for removed_name in ("claude-sonnet-4-20250514", "anthropic/claude-sonnet-4-20250514"):
        pricing = litellm_pricing.get_model_pricing(removed_name)
        assert pricing is not None
        assert pricing.input_cost_per_1m == 3.0
        assert pricing.output_cost_per_1m == 15.0


def test_legacy_claude_3_sonnet_alias_preserves_sonnet_rates(monkeypatch) -> None:
    model_cost = {
        "claude-sonnet-4-5": {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
        }
    }

    def cost_per_token(*, model, **kwargs):
        if model == "claude-sonnet-4-5":
            return (0.000003, 0.000015)
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(
        litellm_pricing,
        "litellm",
        SimpleNamespace(model_cost=model_cost, cost_per_token=cost_per_token),
    )
    monkeypatch.setattr(litellm_pricing, "_resolved_model_cache", {})

    assert litellm_pricing.resolve_litellm_model("claude-3-sonnet-20240229") == "claude-sonnet-4-5"
    pricing = litellm_pricing.get_model_pricing("claude-3-sonnet-20240229")
    assert pricing is not None
    assert pricing.input_cost_per_1m == 3.0
    assert pricing.output_cost_per_1m == 15.0


@pytest.mark.parametrize(
    ("catalog", "expected_input_price"),
    [
        ({"claude-sonnet-4-5": {"input_cost_per_token": 0.000003}}, 3.0),
        ({"anthropic/claude-sonnet-4-5": {"input_cost_per_token": 0.000003}}, 3.0),
    ],
)
@pytest.mark.parametrize(
    "model_name", ["claude-sonnet-4-20250514", "anthropic/claude-sonnet-4-20250514"]
)
def test_litellm_pricing_aliases_support_bare_or_prefixed_catalog_keys(
    monkeypatch, catalog, expected_input_price, model_name
) -> None:
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", SimpleNamespace(model_cost=catalog))

    pricing = litellm_pricing.get_model_pricing(model_name)

    assert pricing is not None
    assert pricing.input_cost_per_1m == expected_input_price


@pytest.mark.parametrize(
    ("catalog_key", "model_name", "expected_model"),
    [
        (
            "claude-sonnet-4-5",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5",
        ),
        (
            "anthropic/claude-sonnet-4-5",
            "claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-5",
        ),
        (
            "claude-sonnet-4-5",
            "anthropic/claude-sonnet-4-20250514",
            "claude-sonnet-4-5",
        ),
        (
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-5",
        ),
    ],
)
def test_litellm_model_resolution_aliases_bare_or_prefixed_catalog_keys(
    monkeypatch, catalog_key, model_name, expected_model
) -> None:
    catalog = {catalog_key: {"input_cost_per_token": 0.000003}}

    def cost_per_token(*, model, **kwargs):
        if model in catalog:
            return (0.000003, 0.000015)
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(
        litellm_pricing,
        "litellm",
        SimpleNamespace(model_cost=catalog, cost_per_token=cost_per_token),
    )
    monkeypatch.setattr(litellm_pricing, "_resolved_model_cache", {})

    assert litellm_pricing.resolve_litellm_model(model_name) == expected_model


@pytest.mark.parametrize(
    "model_name", ["claude-sonnet-4-20250514", "anthropic/claude-sonnet-4-20250514"]
)
def test_original_claude_catalog_key_precedes_alias(monkeypatch, model_name) -> None:
    original = {
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.000016,
    }
    alias = {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }
    catalog = {
        "anthropic/claude-sonnet-4-20250514": original,
        "claude-sonnet-4-5": alias,
    }

    def cost_per_token(*, model, **kwargs):
        if model == "anthropic/claude-sonnet-4-20250514":
            return (0.000004, 0.000016)
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(
        litellm_pricing,
        "litellm",
        SimpleNamespace(model_cost=catalog, cost_per_token=cost_per_token),
    )
    monkeypatch.setattr(litellm_pricing, "_resolved_model_cache", {})

    pricing = litellm_pricing.get_model_pricing(model_name)

    assert pricing is not None
    assert pricing.input_cost_per_1m == 4.0
    assert pricing.output_cost_per_1m == 16.0
    assert litellm_pricing.resolve_litellm_model(model_name) == (
        "anthropic/claude-sonnet-4-20250514"
    )


def test_litellm_model_resolution_does_not_cache_unknown_names(monkeypatch) -> None:
    model_name = "claude-future-model"
    catalog = {model_name: {"input_cost_per_token": 0.000003}}

    def cost_per_token(*, model, **kwargs):
        if model in catalog and model != model_name:
            return (0.000003, 0.000015)
        raise ValueError("unknown model")

    fake_litellm = SimpleNamespace(
        model_cost=catalog,
        cost_per_token=cost_per_token,
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)
    monkeypatch.setattr(litellm_pricing, "_resolved_model_cache", {})

    assert litellm_pricing.resolve_litellm_model(model_name) == model_name
    assert model_name not in litellm_pricing._resolved_model_cache

    catalog["anthropic/claude-future-model"] = {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }
    assert litellm_pricing.resolve_litellm_model(model_name) == "anthropic/claude-future-model"


def test_litellm_model_pricing_returns_none_for_unknown_models(monkeypatch) -> None:
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", SimpleNamespace(model_cost={}))
    assert litellm_pricing.get_model_pricing("missing") is None
