#!/usr/bin/env python3
"""Fill the shader cache with translations from the VIctory translator.

The TuxBox translator refuses programs it cannot lower yet (a large share of
ProjectScarlett's compute and mesh programs) and never requests console DXIL at
all. The VIctory project translates both, so its batch translator is reused as a
seeding pass:

    python tools/victory_seed.py --cache <cache dir> --requests <archive>
    python tools/victory_seed.py --cache <cache dir> --console-dxil <dump dir>

`VICTORY_XLATE` points at the `shader-xlate` binary built from
https://github.com/Marvisak/VIctory (`cargo build --release -p shader-xlate`).
Results are written as `<key>.cso` plus `<key>.sha256`; a refusal marker for the
same key is removed because the program is no longer unsupported.

The translations are produced against VIctory's host binding model, so they are
only a drop-in where the caller's root signature matches that model; the pilot
run fed them into the TuxBox cache to measure pipeline coverage, not to claim
pixel-correct output.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

XBOX_DXIL = b"ILDN"


def victory_xlate() -> pathlib.Path:
    configured = os.environ.get("VICTORY_XLATE")
    candidates = [configured] if configured else []
    candidates += [
        "/mnt/sdb2/gta6/re/VIctory/target/x86_64-unknown-linux-gnu/release/shader-xlate",
        "/mnt/sdb2/gta6/re/VIctory/target/release/shader-xlate",
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return pathlib.Path(candidate)
    raise SystemExit("shader-xlate not found; set VICTORY_XLATE or build VIctory")


def chunks(data: bytes) -> list[bytes] | None:
    if not data.startswith(b"DXBC") or len(data) < 32:
        return None
    total, count = struct.unpack_from("<II", data, 24)
    if total != len(data) or count > 64:
        return None
    found = []
    for index in range(count):
        offset = struct.unpack_from("<I", data, 32 + 4 * index)[0]
        if offset + 8 > len(data):
            return None
        found.append(data[offset:offset + 4])
    return found


def is_console_dxil(data: bytes) -> bool:
    parts = chunks(data)
    return parts is not None and XBOX_DXIL in parts and b"DXIL" in parts


def translate(binary: pathlib.Path, entries: list[tuple[str, pathlib.Path]]) -> tuple[int, int]:
    with tempfile.TemporaryDirectory(prefix="victory-seed-") as tmp:
        tmp = pathlib.Path(tmp)
        manifest = tmp / "manifest.tsv"
        manifest.write_text("".join(f"{key}\t{path}\n" for key, path in entries))
        output = tmp / "out"
        output.mkdir()
        failures = tmp / "failures.tsv"
        result = subprocess.run(
            [str(binary), str(manifest), "--out", str(output), "--failures", str(failures)],
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            print(result.stdout[-2000:], result.stderr[-2000:], file=sys.stderr)
        translated = 0
        cache = pathlib.Path.parse_args  # placeholder replaced by caller
        return translated, len(entries)


def seed(cache: pathlib.Path, entries: list[tuple[str, pathlib.Path]]) -> tuple[int, int]:
    binary = victory_xlate()
    if not entries:
        return 0, 0
    with tempfile.TemporaryDirectory(prefix="victory-seed-") as tmp:
        tmp = pathlib.Path(tmp)
        manifest = tmp / "manifest.tsv"
        manifest.write_text("".join(f"{key}\t{path}\n" for key, path in entries))
        output = tmp / "out"
        output.mkdir()
        failures = tmp / "failures.tsv"
        result = subprocess.run(
            [str(binary), str(manifest), "--out", str(output), "--failures", str(failures)],
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            print(result.stdout[-2000:], result.stderr[-2000:], file=sys.stderr)
        published = 0
        for produced in output.glob("*.cso"):
            data = produced.read_bytes()
            cache.mkdir(parents=True, exist_ok=True)
            (cache / produced.name).write_bytes(data)
            (cache / f"{produced.stem}.sha256").write_text(hashlib.sha256(data).hexdigest())
            (cache / f"{produced.stem}.unsupported").unlink(missing_ok=True)
            published += 1
        refused = len(list(failures.read_text().splitlines())) if failures.is_file() else 0
        return published, refused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=pathlib.Path)
    parser.add_argument("--requests", type=pathlib.Path)
    parser.add_argument("--console-dxil", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    entries: list[tuple[str, pathlib.Path]] = []
    if args.requests:
        for marker in sorted(args.cache.glob("*.unsupported")):
            key = marker.name[: -len(".unsupported")]
            source = args.requests / f"{key}.dxbc"
            if source.is_file():
                entries.append((key, source))
    if args.console_dxil:
        for source in sorted(args.console_dxil.rglob("*.dxbc")):
            data = source.read_bytes()
            if not is_console_dxil(data):
                continue
            entries.append((hashlib.sha256(data).hexdigest(), source))
    if args.limit:
        entries = entries[: args.limit]
    published, refused = seed(args.cache, entries)
    print(f"victory seed: {published} published, {refused} refused, {len(entries)} candidates")


if __name__ == "__main__":
    main()
