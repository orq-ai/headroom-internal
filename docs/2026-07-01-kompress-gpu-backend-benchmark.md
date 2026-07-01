# Kompress GPU Backend Benchmark

**Date:** 2026-07-01 · **Hardware:** NVIDIA RTX 6000 Ada (sm_89) · **Stack:**
`Dockerfile.gpu` image, onnxruntime-gpu 1.26, torch 2.12.1+cu126, transformers 5.12.1

## TL;DR

Added an opt-in `onnx_cuda` Kompress backend (ONNX Runtime CUDA execution provider,
fp32 artifact) and benchmarked single-request `compress()` latency against ONNX-CPU
and PyTorch-GPU. **`onnx_cuda` is the fastest and lightest option — 9–47× faster than
CPU and 1.5–1.9× faster than PyTorch-GPU** — so `auto` prefers it whenever a CUDA
device is present. Response speed was the goal; this is the headline number a caller
feels per compress call.

## Results — single-request `compress()` p50 (ms)

| config | small (256 tok) | medium (1k) | large (4k) |
|---|---|---|---|
| onnx_cpu (int8-wo, prod default on CPU) | 69.6 | 376.4 | 1811.2 |
| onnx_cpu_fp32 (control) | 63.1 | 344.3 | 1569.4 |
| pytorch (GPU) | 11.5 | 21.2 | 73.5 |
| **onnx_cuda (fp32)** | **7.9** | **11.5** | **38.5** |

Batched per-item (ms, batch8/batch32): onnx_cpu 413/415 · onnx_cpu_fp32 359/367 ·
pytorch 21.6/20.9 · **onnx_cuda 11.3/12.4**.

Compression was real on every timed call (min ratio 0.76–0.84), not passthrough.

## Analysis

- **onnx_cuda vs onnx_cpu:** ~8.8× / ~33× / ~47× (small/medium/large). The gap widens
  with input size — CPU inference is O(tokens); the GPU amortizes launch overhead.
- **Hardware isolated (onnx_cpu_fp32 vs onnx_cuda):** ~8× / ~30× / ~41×. The fp32-CPU
  control runs the *same artifact* as onnx_cuda, and it lands next to int8-CPU — so
  the win is the GPU, not the precision change from int8 to fp32.
- **onnx_cuda vs pytorch-GPU:** ~1.5× / ~1.8× / ~1.9×. ONNX Runtime's CUDA EP beats
  PyTorch eager on the same GPU at every size, and needs no torch import at runtime.
- **Small inputs:** even at 256 tokens (where GPU launch overhead is most likely to
  lose) onnx_cuda still wins — nothing argues against the GPU default for this model.

**Decision:** keep `auto`→onnx_cuda when the CUDA EP is available (shipped). ONNX-GPU
is the fastest, and lighter than pulling in torch.

## Method

- `benchmarks/bench_kompress_backends.py`, one config per process (backend is fixed at
  model load). Reuses `benchmarks.bench_latency.generate_plain_text` (realistic input)
  and `headroom.perf.analyzer._percentile`.
- Clean-latency conditions forced by the script: `HEADROOM_COMPRESSION_DEADLINE_MS=0`
  (no work truncation) and `KompressConfig(enable_ccr=False)` (no CCR disk I/O in the
  timed section). Warmup excludes model load / graph init.
- Hard-fails guard against silent lies — the numbers can't be from an accidental
  fallback: onnx_cuda must carry `CUDAExecutionProvider` *and* the fp32 artifact must
  be cached (int8-wo's MatMulNBits has no CUDA kernel); pytorch params must be on the
  CUDA device (the label is "pytorch" even on CPU); at least one timed call must really
  compress (ratio < 1.0).

## Runtime findings

1. **`Dockerfile.gpu` needed an `ldconfig` fix for the ONNX CUDA EP.** onnxruntime-gpu
   1.26 loads its cu12/cuDNN9 libs from the bundled `nvidia-*-cu12` wheels, but those
   lib dirs (`site-packages/nvidia/*/lib`) were not on the loader path — so the CUDA EP
   failed to instantiate and silently fell back to CPU while `get_providers()` still
   listed it. A prior "ORT CUDA session init pass" validation was a false positive for
   exactly this reason. Fixed by registering the dirs in the ld.so cache at build time.
   Without it, `auto`→onnx_cuda silently runs on CPU in production.
2. **pytorch bench needed `-e USER=<name>` (benchmark quirk, not a prod bug).** Under
   `docker run --user <uid>` with no matching `/etc/passwd` entry, torch's
   `getpass.getuser()` raised `KeyError: uid not found` while importing `torch._dynamo`
   (pulled in by transformers' ModernBERT via `deepgemm.py`); the retry then
   double-registered the mega-cache "precompile" artifact (the misleading symptom).
   Production runs as `nonroot` (which has a passwd entry), so this never occurs there.

## Reproduce

Image build bakes only int8-wo, so populate a full HF cache and mount it offline:

```bash
# on the box, HF token present:
hf download chopratejas/kompress-v2-base          # all onnx artifacts + safetensors
hf download answerdotai/ModernBERT-base           # encoder backbone + tokenizer

DOCKER_BUILDKIT=1 docker build -f Dockerfile.gpu \
  --secret id=hf_token,src=$HOME/.cache/huggingface/token -t headroom:cu126 .

for cfg in onnx_cpu onnx_cpu_fp32 onnx_cuda pytorch; do
  docker run --rm --gpus all -v "$PWD":/app -w /app -e PYTHONPATH=/app \
    -v "$HOME/.cache/huggingface":/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -e USER=bench --user "$(id -u):$(id -g)" --entrypoint python headroom:cu126 \
    benchmarks/bench_kompress_backends.py --config "$cfg" --out "$cfg.json"
done

docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app --entrypoint python \
  headroom:cu126 benchmarks/bench_kompress_backends.py --summarize *.json
```

Full plan: `docs/superpowers/plans/2026-07-01-kompress-backend-speed-comparison.md`.
