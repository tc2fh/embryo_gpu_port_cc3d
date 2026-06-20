"""
Repair the pixi env's broken `_ctypes` import (which also breaks torch, imageio, and CC3D runs).

Symptom:
    ImportError: DLL load failed while importing _ctypes: The specified module could not be found.

Root cause:
    Python 3.12 came from the *defaults* channel (repo.anaconda.com/pkgs/main); its `_ctypes.pyd`
    is built to load `ffi.dll`. But `libffi` 3.5.2 was solved from *conda-forge*, which ships the
    very same library as `ffi-8.dll`. `channel-priority = "disabled"` in pixi.toml allowed Python
    and libffi to come from incompatible channels -> name mismatch -> _ctypes can't find its dep.

Fix (this script):
    Alias `<env>/DLLs/ffi.dll` -> a copy of `<env>/Library/bin/ffi-8.dll`. The two are ABI-identical
    libffi 8; `_ctypes.pyd` is loaded with LOAD_WITH_ALTERED_SEARCH_PATH so a copy placed next to it
    in DLLs/ resolves at startup with no PATH changes. Idempotent; safe to re-run after `pixi install`.

DURABLE fix (recommended, not done automatically because it triggers an env re-solve that could
perturb the working CC3D/torch install): make Python and libffi channel-consistent, e.g. pin Python
from conda-forge (so its `_ctypes.pyd` expects conda-forge's `ffi-8.dll`), or enable strict channel
priority with conda-forge primary. Verify afterwards with `pixi run python -c "import ctypes, torch"`.

Usage:
    pixi run python gpu_port/fix_env_ffi.py
"""
import shutil
import sys
from pathlib import Path


def main() -> int:
    prefix = Path(sys.prefix)
    src = prefix / "Library" / "bin" / "ffi-8.dll"
    dst = prefix / "DLLs" / "ffi.dll"

    if dst.exists():
        print(f"OK: {dst} already present")
    elif not src.exists():
        print(f"ERROR: source DLL not found: {src}")
        print("       libffi may be missing entirely; run `pixi install` first.")
        return 1
    else:
        shutil.copy2(src, dst)
        print(f"FIXED: copied {src.name} -> {dst}")

    # Verify the fix actually resolves the import.
    try:
        import ctypes  # noqa: F401
        print("VERIFIED: `import ctypes` works")
        return 0
    except Exception as e:  # pragma: no cover
        print(f"STILL BROKEN: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
