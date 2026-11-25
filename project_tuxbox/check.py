"""Run every check a change needs before it can be merged.

The first group is what CI runs. The second builds every provider and guest and
runs Lanes and the other guest fixtures on the GPU, which CI cannot do, so each
contributor runs it and reports the result.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
def host_target():
    info = subprocess.check_output(["rustc", "-vV"], text=True)
    return next(line.removeprefix("host: ") for line in info.splitlines() if line.startswith("host: "))


def code_steps(python, target):
    cargo = ["cargo", "test", "--locked", "--target", target]
    return [
        ("Python tests", [python, "-m", "unittest", "discover", "-s", "tests", "-q"]),
        ("Shader compiler tests", [*cargo, "--manifest-path", "tools/shader-compiler/Cargo.toml"]),
        ("DXBC tests", [*cargo, "-p", "tuxbox-dxbc", "--features", "native-bindings"]),
        ("XG layout tests", [*cargo, "-p", "tuxbox-xg-layout"]),
        ("D3D12 Windows typecheck", ["cargo", "check", "--locked", "--target", "x86_64-pc-windows-msvc",
                                     "-p", "tuxbox-d3d12", "-p", "tuxbox-bvh"]),
    ]


def gpu_steps(python, output, host_options):
    # No `--only`: every suite, Lanes included.
    return [("Build", [python, "build.py"]),
            ("Lanes and fixtures on the GPU",
             [python, "tuxbox.py", "verify", "--output", str(output), *host_options])]


def benchmark_readings(path):
    """The `name: value` timings of the Lanes benchmark recorded in a results file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in data.get("checks", ()):
        if row.get("gate") == "lanes-benchmark" and row.get("passed"):
            return {name: float(value) for name, value in re.findall(r"(\w+_us)=([0-9.]+)", row["receipt"])}
    return {}


def compare_benchmarks(baseline, current):
    """Lines showing each timing against the baseline. Lower is better."""
    old, new = benchmark_readings(baseline), benchmark_readings(current)
    shared = [name for name in new if old.get(name)]
    if not shared:
        return ["No benchmark readings to compare."]
    return [f"  {name:20s} {old[name]:9.1f} -> {new[name]:9.1f} us  {new[name] / old[name] - 1:+.0%}"
            for name in shared]


def run_checks(*, gpu=True, host_options=(), baseline=None, run=subprocess.run, report=print):
    """Run the steps in order and return a process exit code.

    Every step runs even after a failure, except that the GPU check is skipped
    when a build it depends on failed.
    """
    output = ROOT / "build/check/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = code_steps(sys.executable, host_target())
    if gpu:
        steps += gpu_steps(sys.executable, output, list(host_options))
    results = []
    build_failed = False
    for name, command in steps:
        if name.endswith("on the GPU") and build_failed:
            results.append((name, "skipped: a build failed", 0.0))
            continue
        report(f"== {name}")
        started = monotonic()
        passed = run(command, cwd=ROOT, env=os.environ.copy()).returncode == 0
        build_failed |= name == "Build" and not passed
        results.append((name, "passed" if passed else "FAILED", monotonic() - started))
    report("\nSummary")
    for name, result, seconds in results:
        report(f"  {result:8s} {name} ({seconds:.0f}s)")
    failed = any(result != "passed" for _, result, _ in results)
    if gpu and baseline and output.is_file():
        report(f"\nBenchmark against {baseline}")
        for line in compare_benchmarks(baseline, output):
            report(line)
    if not gpu:
        report("\nThe GPU checks were not run. A change cannot be merged until Lanes passes on a GPU.")
    elif not failed:
        report(f"\nAll checks passed. Attach the GPU results to the pull request: {output}")
    return 1 if failed else 0
