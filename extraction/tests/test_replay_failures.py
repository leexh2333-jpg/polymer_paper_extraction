import tempfile
import unittest
from pathlib import Path

from tools.replay_failures import discover_cases


class ReplayFailureDiscoveryTests(unittest.TestCase):
    def test_default_skips_failure_with_success_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ref = root / "reference_no_0000001"
            ref.mkdir()
            (ref / "stage1_failure.json").write_text("{}", encoding="utf-8")
            (ref / "stage1_mentions.json").write_text("{}", encoding="utf-8")
            (ref / "stage2_failure.json").write_text("{}", encoding="utf-8")

            cases = discover_cases([root])

        self.assertEqual([(case.ref_no, case.stage) for case in cases], [
            ("reference_no_0000001", "stage2"),
        ])

    def test_include_resolved_keeps_historical_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ref = root / "reference_no_0000001"
            ref.mkdir()
            (ref / "stage1_failure.json").write_text("{}", encoding="utf-8")
            (ref / "stage1_mentions.json").write_text("{}", encoding="utf-8")

            cases = discover_cases([root], include_resolved=True)

        self.assertEqual([(case.ref_no, case.stage) for case in cases], [
            ("reference_no_0000001", "stage1"),
        ])


if __name__ == "__main__":
    unittest.main()
