from __future__ import annotations

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.proxy import savings_tracker  # noqa: E402


def test_removed_sonnet_4_model_is_priced_in_savings_history() -> None:
    model = "claude-sonnet-4-20250514"

    savings = savings_tracker._estimate_compression_savings_usd(model, 1000)
    input_cost = savings_tracker._estimate_input_cost_usd(model, 1000)
    assert savings == pytest.approx(0.003)
    assert input_cost == pytest.approx(0.003)
