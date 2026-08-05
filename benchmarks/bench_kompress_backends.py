"""Benchmark Kompress response speed across execution backends.

Headline = single-request `compress()` latency (p50/p95) — the wall time a caller
feels. Batched throughput is secondary. ONE config per process (the backend is
fixed at model load), so run this once per `--config` and compare the JSONs.

Reuses `benchmarks.bench_latency.generate_plain_text` (realistic input, not a
degenerate repeated snippet) and `headroom.perf.analyzer._percentile` (one
correct interpolated percentile, not a third hand-rolled one).

Every config HARD-FAILS on a silent fallback so the numbers can't lie:
  - onnx_cuda: the session must carry CUDAExecutionProvider, AND the fp32
    artifact must already be cached (int8-wo's MatMulNBits has no CUDA kernel and
    would silently run on the CPU fallback provider while still reporting CUDA).
  - pytorch: the model params must actually be on the CUDA device (the backend
    label is "pytorch" even when device=auto degrades to CPU).
  - any: at least one timed call must really compress (ratio < 1.0) — compress()
    swallows inference errors into a near-free passthrough, which would otherwise
    manufacture a fake "winner".

Clean-latency conditions are forced here, not left to the operator: the 20s
compression deadline is disabled (no partial-work truncation) and CCR store I/O
is turned off (disk writes are not model speed).

Usage (see docs/superpowers/plans/2026-07-01-kompress-backend-speed-comparison.md):

    python benchmarks/bench_kompress_backends.py --config onnx_cpu   --out onnx_cpu.json
    python benchmarks/bench_kompress_backends.py --config onnx_cuda  --out onnx_cuda.json
    python benchmarks/bench_kompress_backends.py --summarize *.json
"""

import argparse
import json
import os
import sys
import time

# Repo root on path so `benchmarks.*` imports work when run as a plain file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean-latency env MUST be set before headroom is imported (config read at load).
os.environ.setdefault("HEADROOM_COMPRESSION_DEADLINE_MS", "0")  # no truncation

_FP32 = "onnx/kompress-fp32.onnx"

# name -> backend env, forced onnx artifact, expected cache label, expected device.
CONFIGS = {
    "onnx_cpu": {"backend": "onnx_cpu", "onnx_file": None, "label": "onnx", "device": None},
    # fp32-on-CPU control: isolates hardware from precision vs the int8-wo prod default.
    "onnx_cpu_fp32": {"backend": "onnx_cpu", "onnx_file": _FP32, "label": "onnx", "device": None},
    "onnx_cuda": {"backend": "onnx_cuda", "onnx_file": _FP32, "label": "onnx_cuda", "device": None},
    "pytorch": {"backend": "pytorch", "onnx_file": None, "label": "pytorch", "device": "cuda"},
}

# Target tokens (~4 chars/token via generate_plain_text). Small included on purpose:
# a ~150M model at batch=1/256 tokens is where GPU launch overhead can lose to CPU,
# and real tool outputs are often small.
SIZES = {"small": 256, "medium": 1024, "large": 4096}


def _configure_env(cfg: dict) -> None:
    os.environ["HEADROOM_KOMPRESS_BACKEND"] = cfg["backend"]
    if cfg["onnx_file"]:
        os.environ["HEADROOM_KOMPRESS_ONNX_FILENAME"] = cfg["onnx_file"]
    else:
        os.environ.pop("HEADROOM_KOMPRESS_ONNX_FILENAME", None)


def _assert_artifact_cached(model_id: str, filename: str) -> None:
    """Fail loudly if a forced artifact is not already in the local cache.

    Without this, the override-first candidate list keeps the defaults as a safety
    net, so a missing fp32 silently falls back to int8-wo on the CUDA EP.
    """
    from headroom.transforms import kompress_compressor as kc

    try:
        kc.hf_hub_download_local_first(model_id, filename, allow_network=False)
    except Exception as exc:
        sys.exit(f"FAIL: {filename} not in local cache ({exc}); pre-download it before this run")


def _assert_loaded(cfg: dict, model_id: str):
    from headroom.transforms import kompress_compressor as kc

    entry = kc._kompress_cache.get(model_id)
    if not entry:
        sys.exit("FAIL: model did not load into the cache")
    model, _tok, backend = entry
    if backend != cfg["label"]:
        sys.exit(f"FAIL: loaded backend={backend!r} but config expects {cfg['label']!r}")

    session = getattr(model, "_session", None)
    providers = session.get_providers() if session and hasattr(session, "get_providers") else None
    if cfg["backend"] == "onnx_cuda" and (
        not providers or "CUDAExecutionProvider" not in providers
    ):
        sys.exit(f"FAIL: onnx_cuda did not get the CUDA provider (providers={providers})")

    if cfg["device"] is not None:
        dev = None
        if hasattr(model, "parameters"):
            try:
                dev = next(model.parameters()).device.type
            except StopIteration:
                pass
        if dev != cfg["device"]:
            sys.exit(f"FAIL: {cfg['backend']} loaded on device={dev!r}, expected {cfg['device']!r}")
    return backend, providers


def _stats(samples_ms: list[float]) -> dict:
    from headroom.perf.analyzer import _percentile

    return {
        "mean_ms": round(sum(samples_ms) / len(samples_ms), 2),
        "p50_ms": round(_percentile(samples_ms, 0.50), 2),
        "p95_ms": round(_percentile(samples_ms, 0.95), 2),
        "min_ms": round(min(samples_ms), 2),
        "n": len(samples_ms),
    }


def run(config_name: str, iters: int, warmup: int) -> dict:
    cfg = CONFIGS[config_name]
    _configure_env(cfg)

    from benchmarks.bench_latency import generate_plain_text
    from headroom.transforms.kompress_compressor import (
        HF_MODEL_ID,
        KompressCompressor,
        KompressConfig,
    )

    if cfg["onnx_file"]:
        _assert_artifact_cached(HF_MODEL_ID, cfg["onnx_file"])

    # enable_ccr=False: keep disk/db store I/O out of the timed section.
    compressor = KompressCompressor(KompressConfig(enable_ccr=False))
    texts = {name: generate_plain_text(tok) for name, tok in SIZES.items()}

    for _ in range(warmup):
        compressor.compress(texts["medium"])
    backend, providers = _assert_loaded(cfg, HF_MODEL_ID)

    single = {}
    for name, text in texts.items():
        samples, ratios = [], []
        for _ in range(iters):
            t0 = time.perf_counter()
            r = compressor.compress(text)
            samples.append((time.perf_counter() - t0) * 1000.0)
            ratios.append(r.compression_ratio)
        # Guard: at least one call must have really compressed, not passthrough.
        if min(ratios) >= 1.0:
            sys.exit(f"FAIL: {name} never compressed (ratio>=1.0 every call) — passthrough/error?")
        single[name] = {
            "target_tokens": SIZES[name],
            "words": r.original_tokens,  # KompressResult.original_tokens is a word count
            "min_ratio": round(min(ratios), 3),
            **_stats(samples),
        }

    batched = {}
    for bs in (8, 32):
        batch = [texts["medium"]] * bs
        for _ in range(warmup):
            compressor.compress_batch(batch)
        samples = []
        for _ in range(max(3, iters // 4)):
            t0 = time.perf_counter()
            compressor.compress_batch(batch)
            samples.append((time.perf_counter() - t0) * 1000.0)
        p = _stats(samples)
        p["per_item_ms"] = round(p["mean_ms"] / bs, 2)
        batched[f"batch{bs}"] = p

    return {
        "config": config_name,
        "backend": cfg["backend"],
        "loaded_backend": backend,
        "providers": providers,
        "single_request": single,
        "batched": batched,
    }


def summarize(paths: list[str]) -> None:
    print(f"{'config':<14} {'size':<7} {'p50_ms':>8} {'p95_ms':>8} {'words':>6} {'ratio':>6}")
    for path in paths:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        for size, s in r["single_request"].items():
            print(
                f"{r['config']:<14} {size:<7} {s['p50_ms']:>8} {s['p95_ms']:>8} "
                f"{s['words']:>6} {s['min_ratio']:>6}"
            )


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS))
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--summarize", nargs="+", help="print a table from result JSONs and exit")
    args = ap.parse_args(argv)

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.config:
        sys.exit("--config is required (one of: " + ", ".join(sorted(CONFIGS)) + ")")

    result = run(args.config, args.iters, args.warmup)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
