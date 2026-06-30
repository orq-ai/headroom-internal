"""Assert a uv.lock resolves a real cu126 (CUDA 12.6) torch with no CUDA-13 stack.

Usage: python scripts/verify_cu126_lock.py uv.lock

headroom keeps torch>=2.12.1 (upstream CVE pin), which cu128 (caps at 2.11)
can't satisfy, so it uses the cu126 channel — the only CUDA-12 PyTorch index
with torch 2.12.x. This guard runs in CI so a future `uv lock` can't silently
drift torch back to the PyPI CUDA-13 wheels, which would mismatch the
onnxruntime-gpu<1.27 (libcudart.so.12) pin.

Checks (each a hard exit(1), so it gates under `python -O` too):
  1. a torch entry resolves from the cu126 index, and
  2. its version carries the `+cu126` local tag (cu126 also hosts +cpu wheels), and
  3. NO package in the lock is a CUDA-13 wheel — name ending `-cu13`, or an
     `nvidia-*` / `cuda-*` package at version `13.*`.
"""

import pathlib
import sys
import tomllib


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


if len(sys.argv) != 2:
    fail("usage: verify_cu126_lock.py <uv.lock>")

packages = tomllib.loads(pathlib.Path(sys.argv[1]).read_text()).get("package", [])

# A cross-platform lock has multiple torch entries (cu126 for linux, PyPI for
# darwin/MPS). Select the cu126 one explicitly rather than relying on ordering.
cu = [
    p for p in packages
    if p.get("name") == "torch" and "cu126" in p.get("source", {}).get("registry", "")
]
if not cu:
    fail("no cu126 torch entry")
version = cu[0].get("version", "")
if "+cu126" not in version:  # the cu126 index also hosts +cpu wheels
    fail(f"cu126-index torch is not a cu126 build (+cpu slipped in?): {version}")

# Scan the WHOLE lock: most of the CUDA-13 stack is unsuffixed and transitive
# (nvidia-cublas==13.x, cuda-toolkit==13.x), not under torch's direct deps.
bad = [
    f"{p['name']} {p.get('version', '')}" for p in packages
    if p.get("name", "").endswith("-cu13")
    or (p.get("name", "").startswith(("nvidia-", "cuda-")) and p.get("version", "").startswith("13."))
]
if bad:
    fail(f"CUDA-13 packages present: {bad}")

print(f"OK: torch {version} from cu126, no CUDA-13 packages")
