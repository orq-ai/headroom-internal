"""Assert a uv.lock resolves a real cu126 (CUDA 12.6) torch with no CUDA-13 stack.

Usage: python scripts/verify_cu126_lock.py uv.lock

headroom keeps torch>=2.12.1 (upstream CVE pin), which cu128 (caps at 2.11)
can't satisfy, so it uses the cu126 channel — the only CUDA-12 PyTorch index
with torch 2.12.x. This guard runs in CI so a future `uv lock` can't silently
drift torch back to the PyPI CUDA-13 wheels, which would mismatch the
onnxruntime-gpu<1.27 (libcudart.so.12) pin.

Checks (each a hard exit(1), so it gates under `python -O` too):
  1. a torch entry resolves from the cu126 index, and
  2. EVERY cu-tagged torch entry carries the `+cu126` local tag — a stray
     `+cu130`/`+cpu` torch alongside the good one would otherwise slip past, and
  3. the cu126 torch meets the `>=2.12.1` CVE floor (the reason cu126 exists), and
  4. NO package in the lock is a CUDA-13 wheel — name ending `-cu13`, version
     carrying a `+cu13x` tag, or an `nvidia-*` / `cuda-*` package at major 13.

stdlib-only (tomllib) so the CI lock-guard job needs no `uv sync` / pip install.
"""

import pathlib
import sys

import tomllib

# The CVE floor that motivates cu126 (see module docstring). Bump alongside the
# `torch>=...` pin in pyproject.toml.
TORCH_FLOOR = (2, 12, 1)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _release(version: str) -> tuple[int, ...]:
    """Numeric release tuple from a version string, ignoring epoch/local/pre tags.

    "2.12.1+cu126" -> (2, 12, 1); "13.0.1" -> (13, 0, 1); "13" -> (13,).
    Non-numeric components stop the parse (good enough for a drift gate).
    """
    base = version.split("+", 1)[0].split("!", 1)[-1]
    out = []
    for part in base.split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        fail("usage: verify_cu126_lock.py <uv.lock>")

    path = pathlib.Path(argv[1])
    try:
        packages = tomllib.loads(path.read_text()).get("package", [])
    except FileNotFoundError:
        fail(f"lock file not found: {path}")
    except tomllib.TOMLDecodeError as exc:
        fail(f"lock file is not valid TOML ({path}): {exc}")

    torch = [p for p in packages if p.get("name") == "torch"]
    if not torch:
        fail("no torch entry in lock")

    # No cu-tagged torch may be anything but cu126. The PyPI darwin/MPS torch
    # has no `+cu` tag and is fine; a `+cu130`/`+cu128`/`+cpu` build is not.
    wrong = [
        p.get("version", "")
        for p in torch
        if "+cu" in p.get("version", "") and "+cu126" not in p.get("version", "")
    ]
    if wrong:
        fail(f"non-cu126 CUDA torch present: {wrong}")

    # A cross-platform lock has multiple torch entries (cu126 for linux, PyPI for
    # darwin/MPS). Select the cu126 one by its index registry/url.
    def _from_cu126(p: dict) -> bool:
        src = p.get("source", {})
        # uv writes `registry` for an index URL; newer forms may use `index`.
        return "cu126" in (src.get("registry", "") or src.get("index", ""))

    cu = [p for p in torch if _from_cu126(p)]
    if not cu:
        fail("no cu126 torch entry")
    version = cu[0].get("version", "")
    if "+cu126" not in version:  # the cu126 index also hosts +cpu wheels
        fail(f"cu126-index torch is not a cu126 build (+cpu slipped in?): {version}")
    if _release(version) < TORCH_FLOOR:
        floor = ".".join(map(str, TORCH_FLOOR))
        fail(f"cu126 torch {version} is below the CVE floor torch>={floor}")

    # Scan the WHOLE lock: most of the CUDA-13 stack is unsuffixed and transitive
    # (nvidia-cublas==13.x, cuda-toolkit==13.x), not under torch's direct deps.
    bad = [
        f"{p['name']} {p.get('version', '')}"
        for p in packages
        if p.get("name", "").endswith("-cu13")
        or "+cu13" in p.get("version", "")
        or (
            p.get("name", "").startswith(("nvidia-", "cuda-"))
            and _release(p.get("version", ""))[:1] == (13,)
        )
    ]
    if bad:
        fail(f"CUDA-13 packages present: {bad}")

    print(f"OK: torch {version} from cu126 (>= {'.'.join(map(str, TORCH_FLOOR))}), no CUDA-13 packages")


if __name__ == "__main__":
    main(sys.argv)
