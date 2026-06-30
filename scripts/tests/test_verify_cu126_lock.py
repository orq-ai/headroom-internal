"""Tests for verify_cu126_lock.py — the cu126 lock-drift guard.

Drives the script over synthetic uv.lock fixtures via subprocess (matching the
sibling guard tests) and asserts exit code + message for every drift mode the
docstring claims to catch.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "verify_cu126_lock.py"

# A minimal, valid linux+darwin torch resolution: cu126 for linux, PyPI for mac.
GOOD_LOCK = """
[[package]]
name = "torch"
version = "2.12.1+cu126"
source = { registry = "https://download.pytorch.org/whl/cu126" }

[[package]]
name = "torch"
version = "2.12.1"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "nvidia-cudnn-cu12"
version = "9.5.1.17"
source = { registry = "https://download.pytorch.org/whl/cu126" }
"""


def run(tmp_path: Path, lock_text: str, *, arg: str | None = None) -> subprocess.CompletedProcess:
    lock = tmp_path / "uv.lock"
    lock.write_text(textwrap.dedent(lock_text))
    return subprocess.run(
        [sys.executable, str(SCRIPT), arg if arg is not None else str(lock)],
        capture_output=True,
        text=True,
    )


def test_good_lock_passes(tmp_path: Path):
    r = run(tmp_path, GOOD_LOCK)
    assert r.returncode == 0, r.stderr
    assert "OK: torch 2.12.1+cu126" in r.stdout


def test_no_torch_fails(tmp_path: Path):
    r = run(tmp_path, '[[package]]\nname = "numpy"\nversion = "2.0.0"\n')
    assert r.returncode == 1
    assert "no torch entry" in r.stderr


def test_torch_from_pypi_only_fails(tmp_path: Path):
    r = run(tmp_path, """
        [[package]]
        name = "torch"
        version = "2.12.1"
        source = { registry = "https://pypi.org/simple" }
    """)
    assert r.returncode == 1
    assert "no cu126 torch entry" in r.stderr


def test_cpu_wheel_from_cu126_fails(tmp_path: Path):
    r = run(tmp_path, """
        [[package]]
        name = "torch"
        version = "2.12.1+cpu"
        source = { registry = "https://download.pytorch.org/whl/cu126" }
    """)
    assert r.returncode == 1
    # +cpu has no "+cu" tag, so it's the "no cu126 torch entry" path.
    assert r.stderr


def test_second_cu130_torch_is_caught(tmp_path: Path):
    """The bypass the review found: a good cu126 torch + a stray cu130 sibling."""
    r = run(tmp_path, GOOD_LOCK + """
        [[package]]
        name = "torch"
        version = "2.13.0+cu130"
        source = { registry = "https://download.pytorch.org/whl/cu130" }
    """)
    assert r.returncode == 1
    assert "non-cu126 CUDA torch" in r.stderr


def test_below_cve_floor_fails(tmp_path: Path):
    """A valid cu126 wheel that is too old still fails (the CVE-floor reason)."""
    r = run(tmp_path, """
        [[package]]
        name = "torch"
        version = "2.6.0+cu126"
        source = { registry = "https://download.pytorch.org/whl/cu126" }
    """)
    assert r.returncode == 1
    assert "below the CVE floor" in r.stderr


def test_cuda13_suffixed_dep_fails(tmp_path: Path):
    r = run(tmp_path, GOOD_LOCK + """
        [[package]]
        name = "nvidia-cudnn-cu13"
        version = "9.1.0"
        source = { registry = "https://pypi.org/simple" }
    """)
    assert r.returncode == 1
    assert "CUDA-13 packages present" in r.stderr


def test_cuda13_unsuffixed_transitive_fails(tmp_path: Path):
    """Unsuffixed CUDA-13 transitive (the subtle case the whole-lock scan exists for)."""
    r = run(tmp_path, GOOD_LOCK + """
        [[package]]
        name = "nvidia-cublas"
        version = "13.0.1"
        source = { registry = "https://pypi.org/simple" }
    """)
    assert r.returncode == 1
    assert "CUDA-13 packages present" in r.stderr


def test_missing_file_clean_error(tmp_path: Path):
    r = run(tmp_path, GOOD_LOCK, arg=str(tmp_path / "nope.lock"))
    assert r.returncode == 1
    assert "not found" in r.stderr
    assert "Traceback" not in r.stderr


def test_bad_toml_clean_error(tmp_path: Path):
    r = run(tmp_path, "this is = = not toml [[[")
    assert r.returncode == 1
    assert "not valid TOML" in r.stderr
    assert "Traceback" not in r.stderr


def test_no_arg_fails(tmp_path: Path):
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "usage" in r.stderr


@pytest.mark.parametrize("index_key", ["registry", "index"])
def test_both_source_key_forms(tmp_path: Path, index_key: str):
    """uv may write the index as `registry` (URL) or `index` (named) — match both."""
    r = run(tmp_path, f"""
        [[package]]
        name = "torch"
        version = "2.12.1+cu126"
        source = {{ {index_key} = "https://download.pytorch.org/whl/cu126" }}
    """)
    assert r.returncode == 0, r.stderr
