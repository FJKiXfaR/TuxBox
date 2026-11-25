# Local integration work

This branch carries compatibility work on top of upstream. Upstream merges have
replaced whole files more than once and silently dropped pieces of it, so the
tree keeps a check for every piece:

```sh
python tools/check_local_integration.py
```

Run it after every merge. It fails when a marker disappears.

## Launcher

- `project_tuxbox/prefix_paths.py` resolves the prefix directory the title
  actually runs in (Proton appends `pfx` to the data path the launcher passes as
  `WINEPREFIX`). Shader preparation, the live shader worker, and the shared
  shader caches use it, so the worker and the title agree on one directory.
- Shader preparation and refusal bookkeeping stay usable across runs instead of
  preparing an empty request archive.

## d3d12 provider

- Console-native bytecode never reaches the host shader compiler; programs
  without a host translation or with a refused translation get a stand-in.
- Known-crashing translated programs are refused by digest, and translated
  programs above a size limit are replaced instead of blocking the driver.
- Graphics and compute pipeline creation copy their descriptor and shader data
  and compile behind `TUXBOX_PIPELINE_DEADLINE`; the title receives a stand-in
  pipeline instead of losing frames to a multi-second driver compile.
- Descriptor heaps: the console's additional view heap type maps to the host
  view heap, and descriptor handles that cannot be decoded are logged and
  skipped instead of being forwarded to the driver.
- Present: a lost swap chain is recreated, the frame is taken from the render
  target the title drew into, and `TUXBOX_DUMP_PRESENT_FRAMES=<dir>` writes the
  sampled resource for inspection.
- Illegal instructions: `sse4a.rs` emulates `EXTRQ`/`INSERTQ` next to the
  existing `clwb`/`clzero`/`mwaitx` helpers; a debug assert whose continuation is
  a direct jump is recovered generically, without a hardcoded title address.
- `TUXBOX_SELF_EXIT_AT_MS=<ms>` ends the process through `ExitProcess` so the
  driver's pipeline cache is merged on exit; a killed process leaves it
  unmerged and every later run recompiles from scratch.

## GameInput

- `TUXBOX_SYNTH_PAD_A_MS` (`<ms>` or `<start>,<period>`) presents a virtual pad
  pressing A, which lets an unattended run pass an account prompt.

## Shader translator

- `XSHADER_FORCE_WAVE32=1` lowers a compute program's required wave size from 64
  to 32 so it can be created on adapters whose Vulkan subgroup size is 32. The
  translation is approximate for operations that depend on the 64th lane.

## Merge routine

1. `git fetch origin`
2. `git merge origin/main` (resolve in favour of upstream structure but keep the
   markers listed above; the checker is authoritative).
3. `python tools/check_local_integration.py`
4. `python build.py d3d12` and run a smoke launch.
