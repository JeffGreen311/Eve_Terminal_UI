"""Compatibility shim package for legacy ``eve_core`` imports.

This repository stores the core modules under ``eve_core_assets``.  The GUI
imports them from ``eve_core``; adding the assets directory to the package path
allows those imports to resolve without changing every call site.
"""

from pathlib import Path

# Make ``eve_core.<module>`` resolve against ``eve_core_assets/<module>.py``.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "eve_core_assets"
if _ASSETS_DIR.is_dir():
    __path__.append(str(_ASSETS_DIR))  # type: ignore[name-defined]
