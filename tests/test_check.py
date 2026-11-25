from types import SimpleNamespace
import unittest
from unittest import mock

from project_tuxbox import check
from project_tuxbox.verify import LANES_SCENARIOS, SUITES


class CheckCommandTests(unittest.TestCase):
    def run_checks(self, failing=(), **options):
        commands, lines = [], []

        def run(command, **_):
            commands.append(command)
            return SimpleNamespace(returncode=1 if any(word in command for word in failing) else 0)

        with mock.patch.object(check, "host_target", return_value="x86_64-unknown-linux-gnu"):
            code = check.run_checks(run=run, report=lines.append, **options)
        return code, commands, "\n".join(lines)

    def test_runs_code_checks_then_lanes_on_the_gpu(self):
        code, commands, report = self.run_checks(host_options=["--backend", "proton"])
        self.assertEqual(code, 0)
        verify = commands[-1]
        self.assertIn("verify", verify)
        for suite in check.LANES_SUITES:
            self.assertIn(suite, verify)
            self.assertIn(suite, SUITES)
        self.assertEqual(verify[verify.index("--backend") + 1], "proton")
        self.assertIn("Attach the Lanes results", report)

    def test_a_failure_still_runs_the_remaining_checks(self):
        code, commands, report = self.run_checks(failing=("unittest",))
        self.assertEqual(code, 1)
        self.assertIn("verify", commands[-1])
        self.assertIn("FAILED   Python tests", report)

    def test_a_failed_build_skips_the_gpu_run(self):
        code, commands, report = self.run_checks(failing=("d3d12",))
        self.assertEqual(code, 1)
        self.assertFalse(any("verify" in command for command in commands))
        self.assertIn("skipped: a build failed", report)

    def test_without_a_gpu_says_the_change_is_not_ready(self):
        code, commands, report = self.run_checks(gpu=False)
        self.assertEqual(code, 0)
        self.assertFalse(any("verify" in command for command in commands))
        self.assertIn("cannot be merged until Lanes passes", report)

    def test_the_cache_scenario_is_checked_by_its_own_suite(self):
        self.assertNotIn("xbox-shader", LANES_SCENARIOS)
        self.assertIn("lanes-shader-cache", check.LANES_SUITES)


if __name__ == "__main__":
    unittest.main()
