"""Tests for Kompress compressor.

Covers:
- Lazy imports: module importable without torch installed
- is_kompress_available(): correct detection of [ml] extra
- KompressConfig / KompressResult: dataclass defaults
- KompressCompressor: passthrough for short content, fallback on error
- Transform interface: apply() method
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ── Import safety (the whole point of the fix) ─────────────────────────


class TestLazyImports:
    """The module must be importable without torch/transformers."""

    def test_is_kompress_available_importable(self) -> None:
        """is_kompress_available can be imported even without torch."""
        from headroom.transforms.kompress_compressor import is_kompress_available

        # Should return bool (True or False depending on environment)
        result = is_kompress_available()
        assert isinstance(result, bool)

    def test_module_import_without_torch(self) -> None:
        """Importing the module with torch blocked should not raise."""
        import sys

        # Block torch AND onnxruntime imports
        with patch.dict(
            sys.modules,
            {"torch": None, "torch.nn": None, "onnxruntime": None},
        ):
            from headroom.transforms.kompress_compressor import (
                _is_pytorch_available,
            )

            # Without both torch and onnxruntime, should return False
            assert _is_pytorch_available() is False
            # Note: is_kompress_available() may still return True if onnxruntime
            # was already imported before patching. Test the individual checkers.

    def test_dataclasses_importable_without_torch(self) -> None:
        """KompressConfig, KompressResult, KompressCompressor are importable without torch."""
        from headroom.transforms.kompress_compressor import (
            KompressCompressor,  # noqa: F401
            KompressConfig,
            KompressResult,
        )

        # These don't need torch to instantiate
        config = KompressConfig()
        assert config.device == "auto"
        assert config.enable_ccr is True

        result = KompressResult(
            compressed="hello",
            original="hello world",
            original_tokens=2,
            compressed_tokens=1,
            compression_ratio=0.5,
        )
        assert result.tokens_saved == 1
        assert result.savings_percentage == 50.0


class TestKompressBackendSelection:
    def test_selected_backend_aliases(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "mps")
        assert kmod._selected_backend() == "pytorch_mps"

        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "coreml")
        assert kmod._selected_backend() == "onnx_coreml"

        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "cpu")
        assert kmod._selected_backend() == "onnx_cpu"

        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "onnx_cuda")
        assert kmod._selected_backend() == "onnx_cuda"

        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "unknown")
        assert kmod._selected_backend() == "auto"

    def test_unrecognized_backend_warns_and_falls_back_to_auto(self, monkeypatch, caplog) -> None:
        import headroom.transforms.kompress_compressor as kmod

        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "tpu")
        with caplog.at_level(logging.WARNING, logger=kmod.logger.name):
            assert kmod._selected_backend() == "auto"

        assert any(
            "unrecognized" in record.getMessage() and "tpu" in record.getMessage()
            for record in caplog.records
        )

    def test_valid_backend_values_do_not_warn(self, monkeypatch, caplog) -> None:
        import headroom.transforms.kompress_compressor as kmod

        with caplog.at_level(logging.WARNING, logger=kmod.logger.name):
            for value in ("auto", "onnx", "cpu", "coreml", "mps", "torch", "ONNX-CPU"):
                monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", value)
                kmod._selected_backend()
            monkeypatch.delenv("HEADROOM_KOMPRESS_BACKEND", raising=False)
            kmod._selected_backend()

        assert not caplog.records

    def test_forced_pytorch_mps_backend_uses_mps_device(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        calls: list[tuple[str, str]] = []
        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "pytorch_mps")
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(
            kmod,
            "_load_kompress_pytorch",
            lambda model_id, device, *, allow_download=True: (
                calls.append((model_id, device)) or ("model", "tokenizer", "pytorch")
            ),
        )

        assert kmod._load_kompress("model-a", device="auto") == ("model", "tokenizer", "pytorch")
        assert calls == [("model-a", "mps")]

    def test_forced_coreml_backend_uses_onnx_coreml(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        calls: list[tuple[str, str]] = []
        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "onnx_coreml")
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(
            kmod,
            "_load_kompress_onnx",
            lambda model_id, *, provider="cpu", allow_download=True: (
                calls.append((model_id, provider)) or ("model", "tokenizer", "onnx_coreml")
            ),
        )

        assert kmod._load_kompress("model-b") == ("model", "tokenizer", "onnx_coreml")
        assert calls == [("model-b", "coreml")]

    def test_forced_onnx_cuda_backend_uses_cuda_provider(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        calls: list[tuple[str, str]] = []
        monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", "onnx_cuda")
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(
            kmod,
            "_load_kompress_onnx",
            lambda model_id, *, provider="cpu", allow_download=True: (
                calls.append((model_id, provider)) or ("model", "tokenizer", "onnx_cuda")
            ),
        )

        assert kmod._load_kompress("model-cuda") == ("model", "tokenizer", "onnx_cuda")
        assert calls == [("model-cuda", "cuda")]

    def test_auto_backend_preserves_onnx_first(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        calls: list[tuple[str, str]] = []
        monkeypatch.delenv("HEADROOM_KOMPRESS_BACKEND", raising=False)
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_is_onnx_available", lambda: True)
        monkeypatch.setattr(kmod, "_is_pytorch_available", lambda: True)
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: False)
        monkeypatch.setattr(
            kmod,
            "_load_kompress_onnx",
            lambda model_id, *, provider="cpu", allow_download=True: (
                calls.append(("onnx", provider)) or ("model", "tokenizer", "onnx")
            ),
        )
        monkeypatch.setattr(
            kmod,
            "_load_kompress_pytorch",
            lambda model_id, device, *, allow_download=True: (
                calls.append(("pytorch", device)) or ("model", "tokenizer", "pytorch")
            ),
        )

        assert kmod._load_kompress("model-c") == ("model", "tokenizer", "onnx")
        assert calls == [("onnx", "cpu")]

    def test_auto_backend_uses_cuda_when_gpu_present(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        calls: list[tuple[str, str]] = []
        monkeypatch.delenv("HEADROOM_KOMPRESS_BACKEND", raising=False)
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_is_onnx_available", lambda: True)
        monkeypatch.setattr(kmod, "_is_pytorch_available", lambda: True)
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: True)
        monkeypatch.setattr(
            kmod,
            "_load_kompress_onnx",
            lambda model_id, *, provider="cpu", allow_download=True: (
                calls.append(("onnx", provider)) or ("model", "tokenizer", "onnx_cuda")
            ),
        )

        assert kmod._load_kompress("model-d") == ("model", "tokenizer", "onnx_cuda")
        assert calls == [("onnx", "cuda")]

    def test_onnx_session_options_read_thread_caps(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        created: list[SimpleNamespace] = []

        class FakeSessionOptions:
            def __init__(self) -> None:
                self.intra_op_num_threads = None
                self.inter_op_num_threads = None
                self.enable_cpu_mem_arena = True
                self.enable_mem_pattern = True

        fake_ort = SimpleNamespace(
            SessionOptions=lambda: created.append(FakeSessionOptions()) or created[-1]
        )
        monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_INTRA_THREADS", "2")
        monkeypatch.setenv("HEADROOM_KOMPRESS_ONNX_INTER_THREADS", "1")

        options = kmod._onnx_session_options(fake_ort)

        assert options.intra_op_num_threads == 2
        assert options.inter_op_num_threads == 1
        assert options.enable_cpu_mem_arena is False
        assert options.enable_mem_pattern is False

    def test_onnx_filename_candidates_prefer_fp32(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        monkeypatch.delenv("HEADROOM_KOMPRESS_ONNX_FILENAME", raising=False)
        candidates = kmod._onnx_filename_candidates(prefer_fp32=True)
        assert candidates[0] == kmod._ONNX_FP32_FILENAME
        # All defaults still present (just reordered).
        assert set(candidates) == set(kmod._DEFAULT_ONNX_FILENAMES)

    def test_onnx_filename_candidates_default_order_unchanged(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        monkeypatch.delenv("HEADROOM_KOMPRESS_ONNX_FILENAME", raising=False)
        assert kmod._onnx_filename_candidates() == kmod._DEFAULT_ONNX_FILENAMES

    def test_cuda_provider_falls_back_to_cpu_when_unavailable(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        seen: dict[str, object] = {}
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: False)

        def fake_session(model_id, providers, *, allow_download=True, prefer_fp32=False):
            seen["providers"] = providers
            seen["prefer_fp32"] = prefer_fp32
            return SimpleNamespace(
                get_providers=lambda: ["CPUExecutionProvider"]
            ), "onnx/model.onnx"

        monkeypatch.setattr(kmod, "_create_onnx_session", fake_session)
        monkeypatch.setattr(kmod, "_OnnxModel", lambda session: "model")
        monkeypatch.setattr(kmod, "_load_modernbert_tokenizer", lambda *a, **k: "tokenizer")

        _, _, backend = kmod._load_kompress_onnx("org/model", provider="cuda")

        # Degraded to CPU: providers are CPU-only, fp32 preference dropped, label onnx.
        assert seen["providers"] == ["CPUExecutionProvider"]
        assert seen["prefer_fp32"] is False
        assert backend == "onnx"

    def test_cuda_provider_builds_cuda_list_when_available(self, monkeypatch) -> None:
        import headroom.transforms.kompress_compressor as kmod

        seen: dict[str, object] = {}
        monkeypatch.delenv("HEADROOM_KOMPRESS_CUDA_DEVICE", raising=False)
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: True)

        def fake_session(model_id, providers, *, allow_download=True, prefer_fp32=False):
            seen["providers"] = providers
            seen["prefer_fp32"] = prefer_fp32
            return SimpleNamespace(
                get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ), kmod._ONNX_FP32_FILENAME

        monkeypatch.setattr(kmod, "_create_onnx_session", fake_session)
        monkeypatch.setattr(kmod, "_OnnxModel", lambda session: "model")
        monkeypatch.setattr(kmod, "_load_modernbert_tokenizer", lambda *a, **k: "tokenizer")

        _, _, backend = kmod._load_kompress_onnx("org/model", provider="cuda")

        assert seen["providers"] == [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        assert seen["prefer_fp32"] is True
        assert backend == "onnx_cuda"

    def test_cuda_device_ordinal_from_env(self, monkeypatch) -> None:
        """HEADROOM_KOMPRESS_CUDA_DEVICE picks a non-zero GPU ordinal."""
        import headroom.transforms.kompress_compressor as kmod

        seen: dict[str, object] = {}
        monkeypatch.setenv("HEADROOM_KOMPRESS_CUDA_DEVICE", "1")
        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: True)

        def fake_session(model_id, providers, *, allow_download=True, prefer_fp32=False):
            seen["providers"] = providers
            return SimpleNamespace(
                get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ), kmod._ONNX_FP32_FILENAME

        monkeypatch.setattr(kmod, "_create_onnx_session", fake_session)
        monkeypatch.setattr(kmod, "_OnnxModel", lambda session: "model")
        monkeypatch.setattr(kmod, "_load_modernbert_tokenizer", lambda *a, **k: "tokenizer")

        kmod._load_kompress_onnx("org/model", provider="cuda")
        assert seen["providers"][0] == ("CUDAExecutionProvider", {"device_id": 1})

    def test_cuda_warns_when_session_falls_back_to_cpu(self, monkeypatch, caplog) -> None:
        """CUDA EP available but the session comes back CPU-only → loud warning."""
        import logging

        import headroom.transforms.kompress_compressor as kmod

        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: True)

        def fake_session(model_id, providers, *, allow_download=True, prefer_fp32=False):
            # Provider requested but silently dropped (the ldconfig/lib-path prod bug).
            return SimpleNamespace(
                get_providers=lambda: ["CPUExecutionProvider"]
            ), kmod._ONNX_FP32_FILENAME

        monkeypatch.setattr(kmod, "_create_onnx_session", fake_session)
        monkeypatch.setattr(kmod, "_OnnxModel", lambda session: "model")
        monkeypatch.setattr(kmod, "_load_modernbert_tokenizer", lambda *a, **k: "tokenizer")

        with caplog.at_level(logging.WARNING):
            _, _, backend = kmod._load_kompress_onnx("org/model", provider="cuda")

        assert backend == "onnx_cuda"  # label unchanged even though inference is CPU
        assert any("fell back to CPU providers" in r.message for r in caplog.records)

    def test_cuda_warns_when_loaded_artifact_not_fp32(self, monkeypatch, caplog) -> None:
        """CUDA EP present but fp32 missed → int8 loaded → MatMulNBits runs on CPU."""
        import logging

        import headroom.transforms.kompress_compressor as kmod

        monkeypatch.setattr(kmod, "_kompress_cache", {})
        monkeypatch.setattr(kmod, "_onnx_cuda_available", lambda: True)

        def fake_session(model_id, providers, *, allow_download=True, prefer_fp32=False):
            # CUDA registered, but fp32 was a cache miss so int8-wo loaded instead.
            return SimpleNamespace(
                get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ), "onnx/model_quantized.onnx"

        monkeypatch.setattr(kmod, "_create_onnx_session", fake_session)
        monkeypatch.setattr(kmod, "_OnnxModel", lambda session: "model")
        monkeypatch.setattr(kmod, "_load_modernbert_tokenizer", lambda *a, **k: "tokenizer")

        with caplog.at_level(logging.WARNING):
            kmod._load_kompress_onnx("org/model", provider="cuda")

        assert any("MatMulNBits" in r.message for r in caplog.records)

    def test_onnx_cuda_uses_batched_path(self, monkeypatch) -> None:
        """onnx_cuda must take the GPU batch path, unlike onnx CPU."""
        import headroom.transforms.kompress_compressor as kmod
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()

        # Single-content batching: enabled for onnx_cuda, not for onnx CPU.
        assert compressor._should_batch_single_content(object(), "onnx_cuda") is True
        assert compressor._should_batch_single_content(object(), "onnx") is False
        assert compressor._should_batch_single_content(object(), "onnx_coreml") is False

        # Sequential-fallback: onnx_cuda batches (False), onnx CPU/CoreML don't (True).
        monkeypatch.setattr(kmod, "_kompress_cache", {"m": ("model", "tok", "onnx_cuda")})
        compressor.config.model_id = "m"
        assert compressor._should_use_sequential_fallback() is False

        monkeypatch.setattr(kmod, "_kompress_cache", {"m": ("model", "tok", "onnx")})
        assert compressor._should_use_sequential_fallback() is True

        monkeypatch.setattr(kmod, "_kompress_cache", {"m": ("model", "tok", "onnx_coreml")})
        assert compressor._should_use_sequential_fallback() is True


# ── KompressResult ──────────────────────────────────────────────────────


class TestKompressResult:
    def test_tokens_saved(self) -> None:
        from headroom.transforms.kompress_compressor import KompressResult

        r = KompressResult(
            compressed="a b",
            original="a b c d",
            original_tokens=4,
            compressed_tokens=2,
            compression_ratio=0.5,
        )
        assert r.tokens_saved == 2

    def test_tokens_saved_no_negative(self) -> None:
        from headroom.transforms.kompress_compressor import KompressResult

        r = KompressResult(
            compressed="a b c d e",
            original="a b c",
            original_tokens=3,
            compressed_tokens=5,
            compression_ratio=1.67,
        )
        assert r.tokens_saved == 0

    def test_savings_percentage_zero_tokens(self) -> None:
        from headroom.transforms.kompress_compressor import KompressResult

        r = KompressResult(
            compressed="",
            original="",
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=1.0,
        )
        assert r.savings_percentage == 0.0

    def test_default_model(self) -> None:
        from headroom.transforms.kompress_compressor import HF_MODEL_ID, KompressResult

        r = KompressResult(
            compressed="x",
            original="x y",
            original_tokens=2,
            compressed_tokens=1,
            compression_ratio=0.5,
        )
        assert r.model_used == HF_MODEL_ID


# ── KompressCompressor (without model) ──────────────────────────────────


class TestKompressCompressorPassthrough:
    """Test compressor behavior that doesn't require the actual model."""

    def test_short_content_passthrough(self) -> None:
        """Content under 10 words should pass through unchanged."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        result = compressor.compress("hello world")
        assert result.compressed == "hello world"
        assert result.compression_ratio == 1.0
        assert result.original_tokens == 2
        assert result.compressed_tokens == 2

    def test_empty_content_passthrough(self) -> None:
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        result = compressor.compress("")
        assert result.compressed == ""
        assert result.compression_ratio == 1.0

    def test_fallback_on_model_error(self) -> None:
        """If _load_kompress fails, compress should return passthrough."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        long_text = " ".join(f"word{i}" for i in range(20))

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=RuntimeError("no model"),
        ):
            result = compressor.compress(long_text)
            assert result.compressed == long_text
            assert result.compression_ratio == 1.0


# ── Transform interface ─────────────────────────────────────────────────


class TestKompressTransformInterface:
    def test_apply_short_messages_unchanged(self) -> None:
        """Messages with <10 words should pass through apply() unchanged."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "short"},
        ]
        tokenizer = MagicMock()
        tokenizer.count_text = MagicMock(return_value=5)

        result = compressor.apply(messages, tokenizer)
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "hello"
        assert result.messages[1]["content"] == "short"

    def test_apply_preserves_user_messages(self) -> None:
        """User messages should never be compressed."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        long_text = " ".join(f"word{i}" for i in range(50))
        messages = [{"role": "user", "content": long_text}]
        tokenizer = MagicMock()
        tokenizer.count_text = MagicMock(return_value=50)

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=RuntimeError("should not be called"),
        ):
            result = compressor.apply(messages, tokenizer)
            assert result.messages[0]["content"] == long_text


# ── compress_batch ──────────────────────────────────────────────────────


class TestKompressCompressorBatch:
    """Tests for the batched compression API (compress_batch).

    These exercise the non-model paths — passthrough handling, argument
    validation, order preservation, and fallback behavior on model-load
    failure. The actual batched inference path is covered by integration
    tests that require the model to be downloaded.
    """

    def test_empty_batch_returns_empty_list(self) -> None:
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        result = compressor.compress_batch([])
        assert result == []

    def test_all_short_texts_passthrough_without_model(self) -> None:
        """Texts under 10 words must passthrough; model never loaded."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        contents = ["hello", "world", "short text here"]

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=AssertionError("model should not be loaded for short texts"),
        ):
            results = compressor.compress_batch(contents)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.compressed == contents[i]
            assert r.compression_ratio == 1.0

    def test_order_preserved(self) -> None:
        """Output order must match input order even when model load fails."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        long_texts = [
            " ".join(f"alpha{i}" for i in range(20)),
            " ".join(f"beta{i}" for i in range(20)),
            " ".join(f"gamma{i}" for i in range(20)),
        ]

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=RuntimeError("no model"),
        ):
            results = compressor.compress_batch(long_texts)

        assert len(results) == 3
        assert results[0].original.startswith("alpha0")
        assert results[1].original.startswith("beta0")
        assert results[2].original.startswith("gamma0")

    def test_mixed_short_and_long_passthrough_on_model_failure(self) -> None:
        """Short texts passthrough; long texts fall back to passthrough on model failure."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        contents = [
            "short",
            " ".join(f"word{i}" for i in range(20)),  # triggers model path
            "also short",
        ]

        with patch(
            "headroom.transforms.kompress_compressor._load_kompress",
            side_effect=RuntimeError("no model"),
        ):
            results = compressor.compress_batch(contents)

        assert len(results) == 3
        assert results[0].compressed == "short"
        assert results[0].compression_ratio == 1.0
        assert results[1].compression_ratio == 1.0  # passthrough fallback
        assert results[2].compressed == "also short"

    def test_ratio_list_length_mismatch_raises(self) -> None:
        """If target_ratio is a list it must match contents length."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        contents = ["a b c", "d e f"]

        # Too short
        try:
            compressor.compress_batch(contents, target_ratio=[0.5])
            raise AssertionError("expected ValueError for length mismatch")
        except ValueError as e:
            assert "length" in str(e).lower()

        # Too long
        try:
            compressor.compress_batch(contents, target_ratio=[0.5, 0.5, 0.5])
            raise AssertionError("expected ValueError for length mismatch")
        except ValueError as e:
            assert "length" in str(e).lower()

    def test_batch_of_one_equivalent_to_single_compress_on_short_text(self) -> None:
        """Batch-of-one with short text should produce identical passthrough."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        text = "hello world"

        single = compressor.compress(text)
        batch = compressor.compress_batch([text])

        assert len(batch) == 1
        assert batch[0].compressed == single.compressed
        assert batch[0].compression_ratio == single.compression_ratio
        assert batch[0].original_tokens == single.original_tokens

    def test_uniform_ratio_scalar(self) -> None:
        """A scalar target_ratio must apply to every text in the batch."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        # Short texts — passthrough regardless of ratio
        contents = ["short a", "short b", "short c"]

        results = compressor.compress_batch(contents, target_ratio=0.3)

        assert len(results) == 3
        for r, original in zip(results, contents, strict=True):
            assert r.compressed == original  # short passthrough

    def test_per_item_ratio_list_with_nones(self) -> None:
        """A list of ratios with some None entries must be accepted."""
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        contents = ["short a", "short b", "short c"]
        ratios: list[float | None] = [0.5, None, 0.25]

        # Short texts always passthrough; validating the list shape alone.
        results = compressor.compress_batch(contents, target_ratio=ratios)
        assert len(results) == 3


# ── unload_kompress_model ───────────────────────────────────────────────


class TestUnloadKompressModel:
    def test_unload_when_no_model(self) -> None:
        import headroom.transforms.kompress_compressor as kmod
        from headroom.transforms.kompress_compressor import unload_kompress_model

        # Ensure no model is loaded (previous tests may have set the cache)
        kmod._kompress_cache.clear()

        # Should return False when no model is loaded
        assert unload_kompress_model() is False
