# Kompress Backend Speed Comparison Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to run this task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure Kompress **response speed** (single-request latency) across
execution backends on the RTX 6000 Ada box, and decide which backend headroom
should use on a GPU host. pytorch-CPU excluded.

**Architecture:** One benchmark script (`benchmarks/bench_kompress_backends.py`)
runs ONE config per process (backend is fixed at model load). Run it once per
config, then summarize. It reuses `benchmarks.bench_latency.generate_plain_text`
(realistic input) and `headroom.perf.analyzer._percentile`. Latency is the
headline; batched throughput is secondary.

**Configs (4):**
- `onnx_cpu` — prod default on CPU (int8-wo artifact).
- `onnx_cpu_fp32` — **control**: fp32 on CPU, so the GPU comparison isolates
  hardware from precision (onnx_cpu int8 vs onnx_cuda fp32 changes both at once).
- `onnx_cuda` — ONNX Runtime CUDA EP (fp32 artifact).
- `pytorch` — PyTorch on the GPU.

**Tech Stack:** headroom `KompressCompressor`, onnxruntime-gpu, torch cu126, the
`Dockerfile.gpu` image, the orq research box (`orq-server` skill).

## Global Constraints

- Config selected by `--config`; the script sets `HEADROOM_KOMPRESS_BACKEND`,
  the forced ONNX artifact, and clean-latency env itself. One process per config.
- `benchmarks/` and `scripts/` are **NOT** in the runtime image (it copies only
  site-packages + the `headroom` binary + the baked hf-cache). The repo MUST be
  bind-mounted in: `-v "$PWD":/app -w /app -e PYTHONPATH=/app`.
- Only int8-wo is baked into the image cache. `onnx_cpu_fp32` and `onnx_cuda`
  need the **fp32 artifact pre-downloaded** into a mounted HF cache; the script
  hard-fails if it is absent (no silent int8 fallback).
- Run inside the `Dockerfile.gpu` image with `--gpus all`. Writable output via
  `--user $(id -u):$(id -g)` (runtime user is `nonroot`).
- The script forces `HEADROOM_COMPRESSION_DEADLINE_MS=0` and `enable_ccr=False`
  so neither work-truncation nor CCR disk I/O distorts the timing.
- Same input generator + iteration counts for every config.

## What this measures — and what it does NOT

- **Headline:** single-request `compress()` p50/p95 at ~256/1k/4k tokens = the
  wall time one caller feels. The proxy hot path calls `compress()` per message
  sequentially, so batch=1 is the right unit.
- **Different code paths, on purpose:** for `onnx_cuda`/`pytorch` a single
  `compress()` reroutes through `compress_batch([content])` (GPU batches the
  chunks); `onnx_cpu` runs chunks sequentially. This is what prod does — it is a
  faithful end-to-end comparison, NOT a like-for-like kernel micro-benchmark.
- **Isolated latency, not felt-speed-under-load:** the execution semaphore is
  pinned to 1 for every backend, so this does not capture queueing under
  concurrent proxy traffic. First cut only.
- **Not a quality check:** `min_ratio` is recorded solely as a passthrough guard
  (a run where nothing compressed = swallowed error). It is NOT a compression-
  quality/parity measurement; int8 vs fp32 quality is out of scope here.

---

### Task 1: Verify the harness locally (CPU only, no GPU)

**Files:**
- Verify: `benchmarks/bench_kompress_backends.py`

**Interfaces:**
- Consumes: `KompressCompressor(KompressConfig(enable_ccr=False)).compress(text)`,
  `benchmarks.bench_latency.generate_plain_text(target_tokens)`,
  `headroom.perf.analyzer._percentile(data, pct)`,
  `_kompress_cache[model_id] = (model, tok, backend)`.
- Produces: JSON `{config, backend, loaded_backend, providers, single_request:
  {small|medium|large: {target_tokens, words, min_ratio, mean_ms, p50_ms,
  p95_ms, min_ms, n}}, batched: {batch8|batch32: {..., per_item_ms}}}`.

- [ ] **Step 1: Lint**

Run: `uv run ruff check benchmarks/bench_kompress_backends.py`
Expected: `All checks passed!`

- [ ] **Step 2: Run the CPU config end-to-end (proves the harness)**

Run:
```bash
uv run --all-extras python benchmarks/bench_kompress_backends.py \
  --config onnx_cpu --iters 8 --warmup 2 --out /tmp/onnx_cpu.json
```
Expected: exit 0; JSON with `loaded_backend: "onnx"`; every `single_request.*`
has `min_ratio < 1.0` (real compression) and `p95_ms >= p50_ms`.

- [ ] **Step 3: Commit the script**

```bash
git add benchmarks/bench_kompress_backends.py
git commit -m "test: add Kompress backend response-speed benchmark"
```

### Task 2: Build the GPU image and pre-cache the fp32 artifact on the box

**Files:** none (uses `Dockerfile.gpu`).

- [ ] **Step 1: Build**

```bash
ssh bauke@orq-research-workstation.siberian-pompano.ts.net
# in the headroom-internal checkout, branch bauke/res-1034-headroom-cu126:
DOCKER_BUILDKIT=1 docker build -f Dockerfile.gpu \
  --secret id=hf_token,src=$HOME/.cache/huggingface/token \
  -t headroom:cu126-bench .
```
Expected: build succeeds. `Dockerfile.gpu` asserts `CUDAExecutionProvider` is in
`ort.get_available_providers()` at build time, so a successful build guarantees the
provider is *compiled in* — NOT that it instantiates on GPU at runtime (see Runtime
finding #1: the provider can list yet silently run on CPU). Runtime GPU execution is
verified separately by the benchmark's provider + nvitop checks.

- [ ] **Step 2: Pre-download the fp32 artifact into the host HF cache**

```bash
hf download chopratejas/kompress-v2-base onnx/kompress-fp32.onnx \
  --local-dir-use-symlinks False --local-dir /dev/null 2>/dev/null; \
HF_HOME=$HOME/.cache/huggingface hf download chopratejas/kompress-v2-base \
  onnx/kompress-fp32.onnx
```
Expected: fp32 (601MB) lands in `$HOME/.cache/huggingface`. This cache is mounted
into the runs; the script hard-fails without it.

### Task 3: Run each config in its own container

**Files:** produces `onnx_cpu.json`, `onnx_cpu_fp32.json`, `onnx_cuda.json`,
`pytorch.json` in `$PWD` on the box.

Shared docker flags (repo + HF cache mounted, writable output, GPU):
```bash
DOCKER="docker run --rm --gpus all \
  -v $PWD:/app -w /app -e PYTHONPATH=/app \
  -v $HOME/.cache/huggingface:/hf -e HF_HOME=/hf \
  --user $(id -u):$(id -g) --entrypoint python headroom:cu126-bench \
  benchmarks/bench_kompress_backends.py"
```

- [ ] **Step 1: onnx_cpu (prod, int8-wo)**

`$DOCKER --config onnx_cpu --out onnx_cpu.json`
Expected: `loaded_backend: "onnx"`, exit 0.

- [ ] **Step 2: onnx_cpu_fp32 (control)**

`$DOCKER --config onnx_cpu_fp32 --out onnx_cpu_fp32.json`
Expected: `loaded_backend: "onnx"`, exit 0 (fails fast if fp32 not cached).

- [ ] **Step 3: onnx_cuda (watch nvitop in another shell)**

`$DOCKER --config onnx_cuda --out onnx_cuda.json`
Expected: `loaded_backend: "onnx_cuda"`, `providers` includes
`CUDAExecutionProvider`, exit 0, **GPU-util spike in nvitop**. Any silent
fallback (missing fp32, or CUDA EP absent) exits non-zero with `FAIL:`.

- [ ] **Step 4: pytorch (GPU)**

`$DOCKER --config pytorch --out pytorch.json`
Expected: `loaded_backend: "pytorch"`, exit 0, GPU-util spike. The script
hard-fails if the model landed on CPU (`FAIL: pytorch loaded on device='cpu'`).

### Task 4: Summarize and decide

**Files:** none (reads the four JSONs).

- [ ] **Step 1: Print the table**

```bash
docker run --rm -v $PWD:/app -w /app -e PYTHONPATH=/app \
  --entrypoint python headroom:cu126-bench \
  benchmarks/bench_kompress_backends.py --summarize \
  onnx_cpu.json onnx_cpu_fp32.json onnx_cuda.json pytorch.json
```
Expected: rows of `config size p50_ms p95_ms words ratio`.

- [ ] **Step 2: Decision (numeric, covers the whole outcome space)**

Let `cuda`, `torch`, `cpu` = p50_ms at the **medium (1k)** size (repeat the check
at large; small is informational — GPU launch overhead is expected to lose there).

1. If `cuda <= 1.1 * torch` (ONNX-GPU within 10% of PyTorch-GPU, or faster) AND
   `cuda < cpu` → **keep `auto`→onnx_cuda** (already shipped). onnx_cuda is the
   GPU default.
2. Else if `torch < 0.9 * cuda` (PyTorch-GPU >10% faster than ONNX-GPU) →
   **change the GPU default to pytorch**; keep `onnx_cuda` as an option. This
   requires editing the `auto` branch (kompress_compressor.py) — the current
   `auto`→onnx_cuda was shipped ahead of this data and must be revised.
3. Else (within noise of each other) → keep the shipped `auto`→onnx_cuda for its
   lighter runtime (no torch import), note the tie.

Answer the sub-question "does GPU help a ~150M model at all" from
`onnx_cpu_fp32` vs `onnx_cuda` (same fp32 artifact, hardware isolated): if
`cuda >= cpu_fp32`, GPU does not help this model at 1k tokens — say so.

- [ ] **Step 3: Record the result**

Paste the table + decision into PR #43 (or its Linear ticket). If the decision
changes the GPU default, open a follow-up to edit the `auto` branch and update
the `cuda-gpu-stack-constraints` memory.

---

## Self-Review — does this achieve the goal (response speed)?

- **Headline = single-request `compress()` p50/p95** across three token sizes on
  the real prod code path = "how fast is one response." ✅
- **No silent lies:** hard-fail on onnx_cuda→CPU nodes (fp32-cached + provider
  assert), pytorch→CPU device, and passthrough (min_ratio guard); plus nvitop. ✅
- **Confounds handled:** deadline disabled, CCR off, fp32-cpu control isolates
  hardware from precision, warmup excludes load/graph-init. ✅
- **Honest scope:** batch-path difference, isolated-vs-load latency, and
  "not a quality check" all stated above rather than papered over. ✅
- **Decision rule covers the whole outcome space** (within-10% / pytorch-wins /
  tie) and names the revert if `auto` was shipped wrong. ✅
- **Runnable:** repo bind-mounted (image lacks `benchmarks/`), fp32 pre-cached,
  writable output as the host user. ✅

**Gaps accepted:** absolute ms are box-specific (RTX 6000 Ada, sm_89) — the
ranking transfers, not the values. Concurrency-under-load is a separate, later
measurement.

---

## Results (2026-07-01, RTX 6000 Ada sm_89, ORT 1.26, torch 2.12.1+cu126)

Single-request `compress()` p50 (ms), deadline off, CCR off:

| config | small (256) | medium (1k) | large (4k) | min_ratio |
|---|---|---|---|---|
| onnx_cpu (int8-wo, prod) | 69.6 | 376.4 | 1811.2 | 0.77–0.80 |
| onnx_cpu_fp32 (control) | 63.1 | 344.3 | 1569.4 | 0.76–0.80 |
| pytorch (GPU) | 11.5 | 21.2 | 73.5 | 0.76–0.81 |
| **onnx_cuda (fp32)** | **7.9** | **11.5** | **38.5** | 0.76–0.84 |

Batched per-item (ms): onnx_cpu 413/415 · onnx_cpu_fp32 359/367 · pytorch 21.6/20.9 · **onnx_cuda 11.3/12.4** (batch8/32).

**onnx_cuda vs onnx_cpu:** ~8.8× (small), ~33× (medium), ~47× (large).
**onnx_cuda vs onnx_cpu_fp32 (hardware-isolated):** ~8× / ~30× / ~41×. The fp32-CPU
control ≈ int8-CPU, so the win is the GPU, not precision. GPU decisively helps this
~150M model, and the gap widens with input size (CPU is O(tokens); GPU amortizes).
**onnx_cuda vs pytorch-GPU:** ~1.5× / ~1.8× / ~1.9× — ONNX Runtime CUDA beats PyTorch
eager on the same GPU at every size.

**Decision: keep `auto`→onnx_cuda (shipped).** Decision-rule branch 1 confirmed with
data — `cuda <= 1.1*torch` (it is faster than torch) AND `cuda < cpu`. onnx_cuda is
the fastest option AND the lightest (no torch import at runtime). No revert needed.

### Two runtime findings (the hard-fail guards earned their keep)

1. **`Dockerfile.gpu` needs the cu12 lib dirs on the loader path for the ONNX CUDA EP
   (real prod gap).** onnxruntime-gpu 1.26 can't locate the cu12/cuDNN9 libs (in
   `site-packages/nvidia/*/lib`) at runtime, so it silently falls back to CPU — the
   prior "ORT CUDA session init" validation was a false positive (session created,
   ran CPU). While debugging, `LD_LIBRARY_PATH=$(ls -d …/nvidia/*/lib | tr '\n' :)`
   confirmed the fix. **Shipped fix: register those dirs in the ld.so cache — write
   `/etc/ld.so.conf.d/nvidia-cu12.conf` + `ldconfig` as root before the `USER` switch
   (Dockerfile.gpu runtime stage). Without it, `auto`→onnx_cuda silently runs CPU in
   prod.** (`ldconfig` over `ENV LD_LIBRARY_PATH` — persists in the image, no per-run
   env needed.)
2. **pytorch bench needed `-e USER=<name>` (benchmark quirk, not a prod bug).**
   Under `--user <uid>` with no `/etc/passwd` entry, torch's `getpass.getuser()`
   raised `KeyError: uid not found` while importing `torch._dynamo` (pulled in by
   transformers' ModernBERT → `deepgemm.py @torch._dynamo.allow_in_graph`); the
   retry then double-registered the mega-cache "precompile" artifact — the confusing
   symptom. `-e USER=bench` fixed it (getpass checks env before `pwd`). Prod runs as
   `nonroot` (has a passwd entry), so this does not affect production.

Full-cache offline runs: host `~/.cache/huggingface` populated with the kompress repo
(all onnx artifacts + safetensors) + `answerdotai/ModernBERT-base`, mounted at `/hf`
with `HF_HUB_OFFLINE=1` (the image bakes only int8-wo).
