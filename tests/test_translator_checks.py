from pathlib import Path
import unittest

from tools.check_translator_gpu import CHECKS, ROOT, command_for


class TranslatorCheckListTests(unittest.TestCase):
    def test_every_listed_tool_and_emitter_exists(self):
        examples = ROOT / "tools/shader-compiler/examples"
        for name, (tool, emitter, *_) in CHECKS.items():
            self.assertTrue((ROOT / "tools" / f"{tool}.py").is_file(), name)
            self.assertTrue((examples / f"{emitter}.rs").is_file(), name)

    def test_passes_each_tool_only_what_it_takes(self):
        tools = {"llvm-mc": "mc", "dxc": "dxc", "dxv": "dxv"}
        for name in CHECKS:
            command = command_for(name, Path("/emitters"), tools, {"adapter": "10de:2705"}, Path("/work"))
            source = Path(command[1]).read_text(encoding="utf-8")
            for option in (word for word in command[2:] if word.startswith("--")):
                self.assertIn(option.removeprefix("--"), source, f"{name} {option}")


if __name__ == "__main__":
    unittest.main()
