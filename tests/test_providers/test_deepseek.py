"""Tests for DeepSeek model pricing and cost estimation."""

from types import SimpleNamespace

import pytest

from headroom.pricing.deepseek_prices import (
    DEEPSEEK_PRICES,
    get_deepseek_registry,
)
from headroom.pricing.registry import PricingRegistry


class TestDeepSeekPricingModule:
    """Tests for the DeepSeek pricing data module."""

    def test_deepseek_pricing_contains_v4_models(self):
        assert "deepseek-v4-flash" in DEEPSEEK_PRICES
        assert "deepseek-v4-pro" in DEEPSEEK_PRICES
        assert len(DEEPSEEK_PRICES) == 2

    def test_deepseek_v4_flash_pricing(self):
        pricing = DEEPSEEK_PRICES["deepseek-v4-flash"]
        assert pricing.input_per_1m == 0.14
        assert pricing.output_per_1m == 0.28
        assert pricing.cached_input_per_1m == 0.0028
        assert pricing.context_window == 1_000_000
        assert pricing.provider == "deepseek"
        assert pricing.notes is not None

    def test_deepseek_v4_pro_pricing(self):
        pricing = DEEPSEEK_PRICES["deepseek-v4-pro"]
        assert pricing.input_per_1m == 0.435
        assert pricing.output_per_1m == 0.87
        assert pricing.cached_input_per_1m == 0.003625
        assert pricing.context_window == 1_000_000
        assert pricing.provider == "deepseek"
        assert pricing.notes is not None

    def test_get_deepseek_registry(self):
        registry = get_deepseek_registry()
        assert isinstance(registry, PricingRegistry)
        assert registry.get_price("deepseek-v4-flash") is not None
        assert registry.get_price("deepseek-v4-pro") is not None
        assert registry.get_price("nonexistent") is None

    def test_registry_staleness_and_source_url(self):
        registry = get_deepseek_registry()
        assert registry.source_url == "https://api-docs.deepseek.com/quick_start/pricing"
        # NOTE: no is_stale() assertion here — it compares LAST_UPDATED against
        # date.today(), so it would spuriously fail 30 days after every pricing
        # refresh regardless of code changes. Staleness logic is covered
        # clock-independently by test_is_stale_uses_last_updated_age below.

    def test_is_stale_uses_last_updated_age(self):
        from datetime import date, timedelta

        fresh = PricingRegistry(last_updated=date.today(), source_url="x")
        assert not fresh.is_stale()
        stale = PricingRegistry(
            last_updated=date.today()
            - timedelta(days=PricingRegistry.STALENESS_THRESHOLD_DAYS + 1),
            source_url="x",
        )
        assert stale.is_stale()

    def test_deepseek_registry_estimate_cost(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost("deepseek-v4-flash", input_tokens=1_000_000)
        assert cost.cost_usd == 0.14
        assert "input" in cost.breakdown
        assert cost.pricing_date is not None

    def test_deepseek_registry_estimate_cost_with_cached(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost(
            "deepseek-v4-flash",
            input_tokens=1_000_000,
            cached_input_tokens=1_000_000,
        )
        assert cost.cost_usd == 0.14 + 0.0028


class TestDeepSeekLiteLLMInjection:
    """Tests for DeepSeek V4 pricing injection into litellm."""

    def test_deepseek_v4_models_in_litellm_model_cost(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        assert "deepseek-v4-flash" in litellm.model_cost
        assert "deepseek-v4-pro" in litellm.model_cost

    def test_deepseek_v4_prefixed_models_in_litellm_model_cost(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        assert "deepseek/deepseek-v4-flash" in litellm.model_cost
        assert "deepseek/deepseek-v4-pro" in litellm.model_cost

    def test_injects_fallback_when_upstream_catalog_lacks_models(self, monkeypatch):
        import headroom.pricing.litellm_pricing as lp

        fake_litellm = SimpleNamespace(model_cost={})
        monkeypatch.setattr(lp, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(lp, "litellm", fake_litellm)
        lp._inject_deepseek_pricing()

        assert (
            fake_litellm.model_cost["deepseek-v4-flash"]
            is lp._DEEPSEEK_V4_PRICING["deepseek-v4-flash"]
        )
        assert (
            fake_litellm.model_cost["deepseek-v4-pro"] is lp._DEEPSEEK_V4_PRICING["deepseek-v4-pro"]
        )
        assert (
            fake_litellm.model_cost["deepseek/deepseek-v4-flash"]
            is lp._DEEPSEEK_V4_PRICING["deepseek-v4-flash"]
        )

    def test_upstream_catalog_prices_are_not_overwritten(self, monkeypatch):
        import headroom.pricing.litellm_pricing as lp

        upstream_bare = {"input_cost_per_token": 0.3 / 1_000_000, "source": "upstream"}
        upstream_prefixed = {"input_cost_per_token": 0.3001 / 1_000_000, "source": "upstream"}
        fake_litellm = SimpleNamespace(
            model_cost={
                "deepseek-v4-flash": upstream_bare,
                "deepseek/deepseek-v4-flash": upstream_prefixed,
            }
        )
        monkeypatch.setattr(lp, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(lp, "litellm", fake_litellm)
        lp._inject_deepseek_pricing()

        assert fake_litellm.model_cost["deepseek-v4-flash"] is upstream_bare
        assert fake_litellm.model_cost["deepseek/deepseek-v4-flash"] is upstream_prefixed

    def test_cost_per_token_uses_prefixed_upstream_model(self, monkeypatch):
        import headroom.pricing.litellm_pricing as lp

        catalog = {
            "deepseek/deepseek-v4-flash": {
                "input_cost_per_token": 0.3 / 1_000_000,
                "output_cost_per_token": 1.2 / 1_000_000,
            }
        }

        def cost_per_token(*, model, prompt_tokens, completion_tokens, **kwargs):
            info = catalog[model]
            return (
                prompt_tokens * info["input_cost_per_token"],
                completion_tokens * info["output_cost_per_token"],
            )

        fake_litellm = SimpleNamespace(model_cost=catalog, cost_per_token=cost_per_token)
        monkeypatch.setattr(lp, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(lp, "litellm", fake_litellm)
        monkeypatch.setattr(lp, "_resolved_model_cache", {})

        resolved = lp.resolve_litellm_model("deepseek-v4-flash")
        assert resolved == "deepseek/deepseek-v4-flash"
        assert fake_litellm.cost_per_token(
            model=resolved,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        ) == pytest.approx((0.3, 1.2))

    def test_resolver_ignores_bare_catalog_entry_when_sdk_requires_prefix(self, monkeypatch):
        import headroom.pricing.litellm_pricing as lp

        catalog = {
            "deepseek-v4-flash": {"input_cost_per_token": 0.3 / 1_000_000},
            "deepseek/deepseek-v4-flash": {"input_cost_per_token": 0.3 / 1_000_000},
        }

        def cost_per_token(*, model, **kwargs):
            if model == "deepseek/deepseek-v4-flash":
                return (0.0000003, 0.0000012)
            raise ValueError("DeepSeek models need the provider prefix")

        monkeypatch.setattr(lp, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(
            lp,
            "litellm",
            SimpleNamespace(model_cost=catalog, cost_per_token=cost_per_token),
        )
        monkeypatch.setattr(lp, "_resolved_model_cache", {})

        assert lp.resolve_litellm_model("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"

    def test_resolve_litellm_model_prefixes_deepseek(self):
        from headroom.pricing.litellm_pricing import resolve_litellm_model

        resolved = resolve_litellm_model("deepseek-v4-flash")
        assert resolved == "deepseek/deepseek-v4-flash"

    def test_injection_does_not_overwrite_existing_upstream_entries(self):
        """If litellm upstream already has these, our injection is a no-op."""
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        # Force-inject with wrong value, then verify the injection guard
        litellm.model_cost["deepseek-v4-flash"] = {"input_cost_per_token": 999}
        # Reimport to trigger _inject_deepseek_pricing — but it should NOT overwrite
        import importlib

        import headroom.pricing.litellm_pricing as lp

        importlib.reload(lp)
        assert litellm.model_cost["deepseek-v4-flash"]["input_cost_per_token"] == 999
        # Reset to correct value
        litellm.model_cost["deepseek-v4-flash"] = {
            "input_cost_per_token": 0.14 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
            "cache_read_input_token_cost": 0.0028 / 1_000_000,
            "litellm_provider": "deepseek",
            "max_tokens": 384_000,
            "max_input_tokens": 1_000_000,
        }


class TestDeepSeekAnthropicProviderFallback:
    """Tests that Anthropic provider's _get_pricing handles DeepSeek models."""

    def test_deepseek_v4_flash_fallback(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-v4-flash")
        assert pricing is not None
        assert pricing["input"] == 0.14
        assert pricing["output"] == 0.28
        assert pricing["cached_input"] == 0.0028

    def test_deepseek_v4_pro_fallback(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-v4-pro")
        assert pricing is not None
        assert pricing["input"] == 0.435
        assert pricing["output"] == 0.87
        assert pricing["cached_input"] == 0.003625

    def test_deepseek_unknown_model_returns_none(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-unknown-model")
        assert pricing is None

    def test_deepseek_partial_match_v4_flash_alias(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        # Should match via partial match (flash in v4-flash)
        pricing = provider._get_pricing("deepseek-v4-flash-v1")
        assert pricing is not None

    def test_estimate_cost_deepseek_v4_flash(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-v4-flash",
        )
        assert cost is not None
        assert cost == 0.14

    def test_estimate_cost_deepseek_v4_flash_with_cache(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-v4-flash",
            cached_tokens=1_000_000,
        )
        assert cost is not None
        # All 1M input tokens are cached, so cost = cached_input only
        assert cost == 0.0028
