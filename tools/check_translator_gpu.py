#!/usr/bin/env python3
"""Run every self-contained GPU check of the shader translator.

Each check assembles or builds synthetic console programs, translates them,
runs the result on the GPU and compares every output word with a reference
model. No game data is used. Builds the translator's example emitters first.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools/shader-compiler/Cargo.toml"
TARGET = "x86_64-unknown-linux-gnu"

# name: (tool, emitter example, needs llvm-mc, needs dxc and dxv, extra arguments)
CHECKS = {
    "alu": ("check_basic_alu_gpu", "packed_integer", True, True, ()),
    "ordinary-integer": ("check_ordinary_integer_gpu", "packed_integer", True, True, ()),
    "packed-integer": ("check_packed_integer_gpu", "packed_integer", True, True, ()),
    "packed-saturate": ("check_packed_integer_gpu", "packed_integer", True, True, ("--clamp",)),
    "packed-chain": ("check_packed_chain_gpu", "packed_integer", True, True, ()),
    "packed-chain-saturate": ("check_packed_chain_gpu", "packed_integer", True, True, ("--include-saturate",)),
    "packed-chain-unoptimized": ("check_packed_chain_gpu", "packed_integer", True, True, ("--optimization", "O0")),
    "sad-mad64": ("check_sad_mad64_gpu", "packed_integer", True, True, ()),
    "f64": ("check_f64_exact_gpu", "packed_integer", True, True, ()),
    "swizzle": ("check_compute_swizzle", "compute_swizzle", False, False, ()),
    **{f"wave-extrema-{width}": ("check_wave_extrema", "wave_extrema", False, True, ("--width", str(width)))
       for width in (32, 64)},
    **{f"wave-guard-{width}-{variant}": ("check_wave_guard", "wave_guard", False, False,
                                         ("--width", str(width), "--variant", variant))
       for width in (32, 64) for variant in ("original", "chain", "chain-zero")},
    "pixel-heap-offsets": ("check_pixel_heap_offsets", "pixel_heap_offsets", False, True, ()),
}


def build_emitters():
    """Build the example emitters and return the directory holding them."""
    output = ROOT / "build/shader-compiler"
    command = ["cargo", "build", "--release", "--locked", "--target", TARGET, "--manifest-path", str(MANIFEST),
               "--target-dir", str(output)]
    for example in sorted({check[1] for check in CHECKS.values()}):
        command += ["--example", example]
    subprocess.run(command, check=True, cwd=ROOT)
    return output / TARGET / "release/examples"


def command_for(name, emitters, tools, host, work):
    tool, emitter, assembles, compiles, extra = CHECKS[name]
    command = [sys.executable, str(ROOT / "tools" / f"{tool}.py"), *extra, "--emitter", str(emitters / emitter)]
    if assembles:
        command += ["--llvm-mc", str(tools["llvm-mc"])]
    if compiles:
        command += ["--dxc", str(tools["dxc"]), "--dxv", str(tools["dxv"])]
    for option, value in host.items():
        command += [f"--{option}", str(value)]
    return command + ["--work", str(work / name), "--output", str(work / f"{name}.json")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("provider", "proton", "prefix", "compiler-dir", "dxc", "work"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--llvm-mc", default="llvm-mc")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--only", choices=tuple(CHECKS), action="append")
    a = ap.parse_args(argv)
    llvm_mc = shutil.which(a.llvm_mc)
    if not llvm_mc:
        ap.error("install llvm-mc from LLVM, or pass --llvm-mc")
    dxc = a.dxc.resolve()
    tools = {"llvm-mc": llvm_mc, "dxc": dxc, "dxv": dxc.with_name("dxv")}
    if not (dxc.is_file() and tools["dxv"].is_file()):
        ap.error("--dxc must name a DXC with dxv beside it")
    work = a.work.resolve()
    if work.exists():
        ap.error("choose a fresh work directory")
    work.mkdir(parents=True)
    emitters = build_emitters()
    host = {"provider": a.provider.resolve(), "proton": a.proton.resolve(), "prefix": a.prefix.resolve(),
            "compiler-dir": a.compiler_dir.resolve(), "adapter": a.adapter.lower()}
    results = []
    for name in a.only or CHECKS:
        started = monotonic()
        log = work / f"{name}.log"
        with log.open("w") as stream:
            try:
                status = subprocess.run(command_for(name, emitters, tools, host, work), cwd=ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                status = "timed out"
        lines = log.read_text(errors="replace").strip().splitlines()
        summary = lines[-1][:300] if lines else ""
        results.append({"check": name, "passed": status == 0, "seconds": round(monotonic() - started, 1),
                        "summary": summary})
        print(f"{'PASS' if status == 0 else 'FAIL'}  {name}  {summary}", flush=True)
        if status != 0:
            print("\n".join(lines[-12:]), flush=True)
    passed = all(row["passed"] for row in results)
    (work / "summary.json").write_text(json.dumps(
        {"schema": 1, "adapter": host["adapter"], "passed": passed, "checks": results}, indent=2) + "\n")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
