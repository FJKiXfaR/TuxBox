"""Resolve the Wine prefix directory that the title's process sees.

Proton appends a `pfx` component to the data path the launcher passes as
WINEPREFIX, so host-side code that shares cache directories with the runtime
must resolve the same directory.
"""
from pathlib import Path


def runtime_prefix(env):
    prefix = Path(env["WINEPREFIX"])
    return prefix / "pfx" if (prefix / "pfx").is_dir() else prefix
