#!/usr/bin/env python3
"""Check that local integration work is still present in this tree.

Upstream merges have repeatedly replaced whole files, silently dropping
compatibility work. This script fails when one of the local pieces listed
below is missing so a merge cannot regress a tested behaviour unnoticed.

It only reads source text; no game files or addresses are involved.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

CHECKS = [
    ("launcher: real prefix resolution for shader caches", "project_tuxbox/prefix_paths.py", "def runtime_prefix("),
    ("launcher: preparation uses the real prefix", "project_tuxbox/scarlett_prep.py", "runtime_prefix(host.env)"),
    ("launcher: live worker uses the real prefix", "project_tuxbox/runtime_shaders.py", "runtime_prefix(env)"),
    ("d3d12: console-native compute guard", "runtime/d3d12/src/device/compute.rs", "host_compatible(bytes)"),
    ("d3d12: learned refused translated DXIL prefixes", "runtime/d3d12/src/device/compute.rs", "DRIVER_REFUSED_PREFIXES"),
    ("d3d12: oversized host code limit", "runtime/d3d12/src/device/compute.rs", "TUXBOX_MAX_HOST_CODE_BYTES"),
    ("d3d12: console-native stream stage guard", "runtime/d3d12/src/device/stream.rs", "console_native_stage"),
    ("d3d12: owned pipeline stream for background compile", "runtime/d3d12/src/device/stream.rs", "OwnedStream"),
    ("d3d12: descriptor heap type mapping", "runtime/d3d12/src/device.rs", "fn host_heap_type"),
    ("d3d12: undecodable descriptor handles are not forwarded", "runtime/d3d12/src/device/descriptor_copy.rs", "descriptor copy skipped"),
    ("d3d12: lost swap chain is recreated", "runtime/d3d12/src/commands/present.rs", "presentation access lost"),
    ("d3d12: present redirects to the drawn target", "runtime/d3d12/src/commands/present.rs", "redirect::source"),
    ("d3d12: redirect records bound targets", "runtime/d3d12/src/commands/graphics.rs", "redirect::bound"),
    ("d3d12: present sampling follows redirect", "runtime/d3d12/src/diagnostics.rs", "crate::present::redirect::enabled()"),
    ("d3d12: raw present frame dump", "runtime/d3d12/src/present.rs", "TUXBOX_DUMP_PRESENT_FRAMES"),
    ("d3d12: console DXIL cache-only path", "runtime/d3d12/src/device/shader_cache.rs", "is_console_dxil"),
    ("d3d12: generic recoverable assert rule", "runtime/d3d12/src/survey.rs", "bytes.first() == Some(&0xcc)"),
    ("d3d12: optional self exit for cache merge", "runtime/d3d12/src/survey.rs", "install_self_exit"),
    ("d3d12: SSE4a emulation registered", "runtime/d3d12/src/survey.rs", "crate::sse4a::handle_illegal_instruction"),
    ("d3d12: SSE4a EXTRQ/INSERTQ emulation", "runtime/d3d12/src/sse4a.rs", "INSERTQ"),
    ("gameinput: synthetic pad", "runtime/gameinput/src/sample.rs", "TUXBOX_SYNTH_PAD_A_MS"),
    # The translator was split into modules by upstream 9e491ff; the wave-size
    # decision now lives in hlsl/driver/pipeline.rs.
    ("xshader: forced wave32 knob", "tools/shader-compiler/src/hlsl/driver/pipeline.rs", "XSHADER_FORCE_WAVE32"),
    ("launcher: refusal retry switch", "project_tuxbox/runtime_shaders.py", "TUXBOX_RETRY_REFUSALS"),
    ("launcher: cache seeding opt-out", "project_tuxbox/runtime_shaders.py", "TUXBOX_CACHE_NO_SEED"),
    ("launcher: compile-policy DXIL split", "project_tuxbox/dxil_store.py", "dxil-unoptimised"),
    ("d3d12: present scans several targets", "runtime/d3d12/src/present.rs", "TUXBOX_SCAN_TARGETS"),
    ("d3d12: title RVA in diagnostics", "runtime/d3d12/src/survey.rs", "fn title_rva"),
    ("xgameruntime: API trace switch", "runtime/xgameruntime/src/services.rs", "TUXBOX_TRACE_API"),
]

missing = []
for label, relative, marker in CHECKS:
    path = ROOT / relative
    if not path.is_file() or marker not in path.read_text(errors="replace"):
        missing.append((label, relative, marker))

if missing:
    print("local integration work is missing after a merge:")
    for label, relative, marker in missing:
        print(f"  - {label}: {relative} lacks {marker!r}")
    sys.exit(1)
print(f"local integration intact: {len(CHECKS)} markers present")
