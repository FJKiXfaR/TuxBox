# VEH proxy (xmem.dll)

Emulates the AMD-only x86 instructions the Xbox Series build of GTA VI executes
on an Intel host (an Intel i7-9850H has no SSE4a, CLZERO, CLWB, MONITORX/MWAITX):

- SSE4a, decoded per AMD APM: `66 0F 78 /0 ib ib` = EXTRQ imm, `66 0F 79 /r` =
  EXTRQ reg, `F2 0F 78 /r ib ib` = INSERTQ imm, `F2 0F 79 /r` = INSERTQ reg.
  The mandatory prefix selects the operation; the opcode selects the form.
  (An earlier revision had these swapped; both logged title crashes at
  RVA 0x2E7FF1 and 0x455BAC7 are INSERTQ-immediate, per the APM and iced-x86.)
- CLZERO (`0F 01 FC`), CLWB/CLFLUSHOPT (`66 0F AE /6`, `/7`, NOP-patched),
  MONITORX/MWAITX (NOP), PREFETCHW (hint), MOVNTSS/MOVNTSD (non-temporal
  scalar stores), MCOMMIT, WBNOINVD.
- The proxy also forwards the XMem*/TuxBox* exports of the real provider via
  `xmem.def` and skips the title's no-progress fatal assert, recording a thread
  snapshot for diagnosis.
