from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.rank_pipeline_failures import HOST_REJECTED, NO_ANSWER, NO_REASON, main, normalize, rank


class RankPipelineFailures(unittest.TestCase):
    def test_reasons_group_over_offsets_and_registers(self):
        self.assertEqual(normalize("wave reduction at 0x3a74 may diverge\nmore"),
                         "wave reduction at 0x_ may diverge")
        self.assertEqual(normalize("mimg at 0x2e4: s32 not a sampler (InlineWord(146)) words [1]"),
                         "mimg at 0x_: sN not a sampler (InlineWord(_))")

    def test_pipelines_count_once_under_their_first_refused_stage(self):
        rows = ["count\toutcome\tpath\tstages",
                "5\t0x80004001\tslot86\tMS=aa PS=bb",
                "2\tstand-in\tslot62\tCS=aa",
                "3\t0x80070057\tslot86\tVS=cc PS=dd",
                "garbage"]
        pipelines, shaders, outcomes = rank(rows, {"aa": "first", "bb": "second"})
        self.assertEqual(pipelines, {"first": 7, NO_REASON: 3})
        self.assertEqual(shaders, {"first": 1, NO_REASON: 0})
        self.assertEqual(outcomes, {"0x80004001": 5, "stand-in": 2, "0x80070057": 3})

    def test_translation_lookups_name_the_refused_request_or_the_host(self):
        rows = ["count\toutcome\tpath\tstages",
                "5\t0x80004001\tslot86\tMS=raw MS.missing=rootless MS.missing=rooted",
                "3\t0x80070057\tslot86\tMS=raw MS.used=good",
                "2\t0x80004001\tslot62\tCS=other CS.missing=unknown"]
        reasons = {"rootless": "mesh root signature missing", "rooted": "real reason",
                   "good": "stale reason from an older translator"}
        pipelines, _, _ = rank(rows, reasons)
        self.assertEqual(pipelines, {"real reason": 5, HOST_REJECTED: 3, NO_ANSWER: 2})

    def test_newest_run_is_ranked_from_a_cache_folder(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "pipeline-failures").mkdir()
            (cache / "runtime/one").mkdir(parents=True)
            (cache / "runtime/one/aa.reason").write_text("unsupported thing")
            (cache / "pipeline-failures/1-2.tsv").write_text(
                "count\toutcome\tpath\tstages\n4\t0x80004001\tslot62\tCS=aa\n")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(cache)]), 0)
        self.assertIn("4 pipelines not delivered: 4 0x80004001", output.getvalue())
        self.assertIn("100.0%        1  unsupported thing", output.getvalue())


if __name__ == "__main__":
    unittest.main()
