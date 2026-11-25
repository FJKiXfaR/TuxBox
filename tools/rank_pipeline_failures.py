#!/usr/bin/env python3
"""Rank why ProjectScarlett pipelines failed, by how many pipelines each reason cost.

The provider writes tuxbox_shader_cache/pipeline-failures/<run>.tsv under the
prefix's temporary folder. Shader preparation and live translation keep a
refusal reason beside each unsupported shader in tuxbox_shader_cache/runtime.
Both stay on your machine; the printed summary holds no paths or shader data.
"""
import argparse
from collections import Counter
from pathlib import Path
import re


NO_REASON = "no translator refusal recorded"
HOST_REJECTED = "translated, then rejected by the driver or runtime"
NO_ANSWER = "translation missing with no recorded reason"
ROOTLESS = "root signature missing"


def normalize(reason):
    """Group reasons that differ only in offsets, registers or encoded words."""
    reason = reason.split("\n", 1)[0].strip()
    reason = re.sub(r" words \[.*", "", reason)
    reason = re.sub(r"0x[0-9a-f]+", "0x_", reason)
    reason = re.sub(r"\bs\d+\b", "sN", reason)
    return re.sub(r"InlineWord\(\d+\)", "InlineWord(_)", reason)


def load_reasons(caches):
    """digest -> reason. Newer cache folders win over older ones."""
    reasons = {}
    for cache in sorted(caches, key=lambda path: path.stat().st_mtime):
        for path in cache.glob("*.reason"):
            reasons[path.stem] = normalize(path.read_text(encoding="utf-8", errors="replace"))
    return reasons


def rank(lines, reasons):
    """Return (pipelines per reason, distinct shaders per reason, pipelines per outcome)."""
    pipelines, shaders, outcomes = Counter(), {}, Counter()
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 4 or not fields[0].isdigit():
            continue
        count, outcome, _, stages = int(fields[0]), *fields[1:]
        outcomes[outcome] += count
        items = [item.partition("=") for item in stages.split()]
        used = {digest for name, _, digest in items if name.endswith(".used")}
        found = [(reasons[digest], digest) for _, _, digest in items
                 if digest in reasons and digest not in used]
        # A pipeline is counted once, under its first refused stage. A request
        # without a root signature says less than the rooted one beside it.
        found.sort(key=lambda entry: ROOTLESS in entry[0])
        if found:
            reason = found[0][0]
        elif any(name.endswith(".missing") for name, _, _ in items):
            reason = NO_ANSWER
        else:
            reason = HOST_REJECTED if used else NO_REASON
        pipelines[reason] += count
        distinct = shaders.setdefault(reason, set())
        if found:
            distinct.add(found[0][1])
    return pipelines, {reason: len(found) for reason, found in shaders.items()}, outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("cache", type=Path, help="the prefix's tuxbox_shader_cache folder")
    parser.add_argument("--run", type=Path, help="a failure file (default: the newest)")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    run = args.run or max((args.cache / "pipeline-failures").glob("*.tsv"),
                          key=lambda path: path.stat().st_mtime, default=None)
    if run is None or not run.is_file():
        parser.error("no pipeline failure file found; run the game first")
    reasons = load_reasons(path for path in (args.cache / "runtime").iterdir() if path.is_dir())
    pipelines, shaders, outcomes = rank(run.read_text(encoding="utf-8").splitlines(), reasons)
    total = sum(pipelines.values())
    print(f"{total} pipelines not delivered: "
          + ", ".join(f"{count} {outcome}" for outcome, count in outcomes.most_common()))
    print("pipelines   share  shaders  reason")
    for reason, count in pipelines.most_common(args.top):
        print(f"{count:9d}  {100 * count / total:5.1f}%  {shaders[reason]:7d}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
