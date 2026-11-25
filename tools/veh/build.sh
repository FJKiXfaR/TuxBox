#!/bin/bash
# Build the xmem.dll VEH proxy that emulates AMD-only x86 instructions for the
# Xbox Series title on an Intel host. Requires the Windows SDK headers and the
# LLVM toolchain; paths are overridable from the environment.
set -eu
S=${XWIN_SDK_ROOT:-/mnt/sdb2/gta6/xwin-sdk/tux}
CC=${CC:-clang-cl}
LLD=${LLD:-lld-link}
cd "$(dirname "$0")"
"$CC" --target=x86_64-pc-windows-msvc /nologo /c /O2 /GS- \
  -I"$S"/crt/include -I"$S"/sdk/include/ucrt -I"$S"/sdk/include/um -I"$S"/sdk/include/shared \
  vehshim.c /Fovehshim.obj
"$LLD" /nologo /DLL /OUT:xmem.dll /DEF:xmem.def /NODEFAULTLIB /SUBSYSTEM:WINDOWS \
  /MACHINE:X64 /ENTRY:DllMain /BASE:0x180000000 /DYNAMICBASE \
  /LIBPATH:"$S"/sdk/lib/um/x86_64 /LIBPATH:"$S"/sdk/lib/ucrt/x86_64 \
  vehshim.obj kernel32.lib user32.lib
