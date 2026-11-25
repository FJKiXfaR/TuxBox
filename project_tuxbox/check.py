"""Run every check a change needs before it can be merged.

The first group is what CI runs. The second builds Lanes and runs it on the
GPU, which CI cannot do, so each contributor runs it and reports the result.
"""
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
LANES_SUITES = ("lanes", "lanes-selftest", "lanes-shader-cache")
# What Lanes loads, in build order.
LANES_TARGETS = ("xmem", "d3d12", "gameinput", "lanes")


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
    steps = [(f"Build {target}", [python, "build.py", target]) for target in LANES_TARGETS]
    verify = [python, "tuxbox.py", "verify", "--output", str(output), *host_options]
    for suite in LANES_SUITES:
        verify += ["--only", suite]
    steps.append(("Lanes on the GPU", verify))
    return steps


def run_checks(*, gpu=True, host_options=(), run=subprocess.run, report=print):
    """Run the steps in order and return a process exit code.

    Every step runs even after a failure, except that the GPU check is skipped
    when a build it depends on failed.
    """
    output = ROOT / "build/check/lanes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = code_steps(sys.executable, host_target())
    if gpu:
        steps += gpu_steps(sys.executable, output, list(host_options))
    results = []
    build_failed = False
    for name, command in steps:
        if name == "Lanes on the GPU" and build_failed:
            results.append((name, "skipped: a build failed", 0.0))
            continue
        report(f"== {name}")
        started = monotonic()
        passed = run(command, cwd=ROOT, env=os.environ.copy()).returncode == 0
        build_failed |= name.startswith("Build ") and not passed
        results.append((name, "passed" if passed else "FAILED", monotonic() - started))
    report("\nSummary")
    for name, result, seconds in results:
        report(f"  {result:8s} {name} ({seconds:.0f}s)")
    failed = any(result != "passed" for _, result, _ in results)
    if not gpu:
        report("\nThe GPU checks were not run. A change cannot be merged until Lanes passes on a GPU.")
    elif not failed:
        report(f"\nAll checks passed. Attach the Lanes results to the pull request: {output}")
    return 1 if failed else 0
