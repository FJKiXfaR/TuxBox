import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory
import unittest

from tools.reference.scarlett_worker_fixture import container, pack


def native_shader(code):
    shader = container(code, resources=[(7, 0, 0, 0)])
    # Extend the fixture's two-chunk directory with a root table for UAV u0.
    root = b"RTS0" + pack([64, 1, 1, 24, 0, 0, 0, 0, 0, 36, 1, 44, 1, 1, 0, 0, 0])
    first, second = struct.unpack_from("<II", shader, 32)
    root_at = len(shader) + 4
    directory = pack([root_at + len(root), 3, first + 4, second + 4, root_at])
    return shader[:24] + directory + shader[first:] + root


class FloatToIntegerTests(unittest.TestCase):
    def test_rounding_at_half_and_integral_boundaries(self):
        self.check_conversion(0x7e0818ff, [
            (0x3effffff, 0), (0x3f000000, 1), (0xbf000000, 0),
            (0xbf000001, -1), (0xbfc00000, -1), (0x40200000, 3),
            (0x4b000001, 8388609), (0xcb000001, -8388609),
        ])

    def test_saturating_signed_and_unsigned_conversion(self):
        # gfx1030 V_CVT_I32_F32 and V_CVT_U32_F32. The expected bits include
        # the largest representable f32 below each positive integer boundary.
        cases = [
            (0x4effffff, 0x7fffff80, 0x7fffff80),
            (0x4f000000, 0x7fffffff, 0x80000000),
            (0x4f7fffff, 0x7fffffff, 0xffffff00),
            (0x4f800000, 0x7fffffff, 0xffffffff),
            (0xcf000001, 0x80000000, 0),
            (0x7f800000, 0x7fffffff, 0xffffffff),
            (0xff800000, 0x80000000, 0),
            (0x7fc00001, 0, 0), (0xff800001, 0, 0),
            (0xbfffffff, -1, 0), (0x3fffffff, 1, 1),
        ]
        self.check_conversion(0x7e0810ff, [(bits, s) for bits, s, _ in cases])
        self.check_conversion(0x7e080eff, [(bits, u) for bits, _, u in cases])

    def check_conversion(self, instruction, cases):
        xshader = os.environ.get("TUXBOX_XSHADER") or shutil.which("xshader")
        dxc = os.environ.get("DXC") or shutil.which("dxc")
        if not (xshader and dxc):
            self.skipTest("set TUXBOX_XSHADER and DXC to run the native rounding check")

        # gfx1030: load the UAV descriptor, convert a literal, store v4 at lane * 4.
        load = [0xf4080402, 0xfa000000, 0xbf8cc07f]
        store = [0x340a0082, 0xe0701000, 0x80040405, 0xbf810000]
        with TemporaryDirectory() as directory:
            trial = Path(directory)
            for bits, expected in cases:
                with self.subTest(instruction=hex(instruction), bits=hex(bits)):
                    code = load + [instruction, bits] + store
                    (trial / "native.dxbc").write_bytes(native_shader(code))
                    for command in (
                        [xshader, trial / "native.dxbc", "--hlsl-out", trial / "shader.hlsl"],
                        [dxc, "-T", "cs_6_0", "-E", "main", "-Fc", trial / "shader.ll",
                         "-Fo", trial / "shader.cso", trial / "shader.hlsl"],
                    ):
                        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    # DXC folds the conversion before the shader writes its only output.
                    ir = (trial / "shader.ll").read_text()
                    values = re.findall(r"call void @dx\.op\.bufferStore\.i32\([^\n]+?, i32 undef, i32 (-?\d+),", ir)
                    self.assertEqual([int(value) & 0xffffffff for value in values], [expected & 0xffffffff])
