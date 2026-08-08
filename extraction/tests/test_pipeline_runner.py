import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pipeline_runner


class PipelineRunnerTests(unittest.TestCase):
    def test_preview_flag_is_forwarded_to_batch_runner(self) -> None:
        args = pipeline_runner.build_parser().parse_args([
            "--input-dir",
            ".",
            "--preview",
        ])

        command = pipeline_runner.build_extraction_command(
            args,
            processed_output=Path("processed"),
            ref_nos=["reference_no_0000001"],
        )

        self.assertIn("--preview", command)

    def test_dry_run_selects_documents_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdfs"
            input_dir.mkdir()
            for index in (2, 1):
                (input_dir / f"reference_no_{index:07d}.pdf").write_bytes(b"pdf")

            with patch.object(pipeline_runner, "run_command") as run_command:
                result = pipeline_runner.main(
                    ["--input-dir", str(input_dir), "--max-documents", "1", "--dry-run"]
                )

            self.assertEqual(result, 0)
            run_command.assert_not_called()

    def test_pipeline_passes_same_selection_through_all_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdfs"
            input_dir.mkdir()
            (input_dir / "reference_no_0000001.pdf").write_bytes(b"pdf")
            mineru_output = root / "mineru"
            organized_root = root / "organized"
            processed_output = root / "processed"
            stats = {"errors": [], "warnings": [], "blocks_count": 1, "images_copied": 0}

            with (
                patch.object(pipeline_runner, "run_command") as run_command,
                patch.object(pipeline_runner, "reorganize_paper", return_value=stats) as reorganize,
                patch.object(
                    pipeline_runner,
                    "transform_paper",
                    return_value=processed_output / "documents" / "reference_no_0000001_document.json",
                ) as transform,
            ):
                result = pipeline_runner.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--mineru-output",
                        str(mineru_output),
                        "--organized-root",
                        str(organized_root),
                        "--processed-output",
                        str(processed_output),
                        "--ref-no",
                        "reference_no_0000001",
                        "--skip-meta",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(run_command.call_count, 2)
            ocr_command = run_command.call_args_list[0].args[0]
            extraction_command = run_command.call_args_list[1].args[0]
            self.assertIn("reference_no_0000001", ocr_command)
            self.assertIn("reference_no_0000001", extraction_command)
            reorganize.assert_called_once_with(
                mineru_output.resolve(), organized_root.resolve(), "reference_no_0000001"
            )
            self.assertEqual(transform.call_args.args[3], "reference_no_0000001")

    def test_pipeline_stops_after_mineru_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "reference_no_0000001.pdf").write_bytes(b"pdf")

            with (
                patch.object(
                    pipeline_runner,
                    "run_command",
                    side_effect=pipeline_runner.PipelineRunnerError("ocr failed"),
                ),
                patch.object(pipeline_runner, "reorganize_paper") as reorganize,
                patch.object(pipeline_runner, "transform_paper") as transform,
            ):
                with self.assertRaisesRegex(pipeline_runner.PipelineRunnerError, "ocr failed"):
                    pipeline_runner.main(["--input-dir", str(input_dir), "--skip-meta"])

            reorganize.assert_not_called()
            transform.assert_not_called()

    def test_metadata_initialization_failure_stops_without_skip_meta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "reference_no_0000001.pdf").write_bytes(b"pdf")

            with (
                patch.object(
                    pipeline_runner,
                    "ConfiguredMetaExtractor",
                    side_effect=RuntimeError("missing config"),
                ),
                patch.object(pipeline_runner, "transform_paper") as transform,
            ):
                with self.assertRaisesRegex(
                    pipeline_runner.PipelineRunnerError,
                    "--skip-meta",
                ):
                    pipeline_runner.main([
                        "--input-dir",
                        str(input_dir),
                        "--skip-ocr",
                        "--skip-organize",
                        "--skip-extraction",
                    ])

            transform.assert_not_called()

    def test_metadata_extraction_failure_stops_without_skip_meta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdfs"
            input_dir.mkdir()
            (input_dir / "reference_no_0000001.pdf").write_bytes(b"pdf")
            output = root / "reference_no_0000001_document.json"
            output.write_text(
                json.dumps({
                    "paper": {
                        "metadata_extraction": {
                            "status": "failed",
                            "error_type": "SchemaError",
                        }
                    }
                }),
                encoding="utf-8",
            )

            with (
                patch.object(
                    pipeline_runner,
                    "ConfiguredMetaExtractor",
                    return_value=Mock(),
                ),
                patch.object(
                    pipeline_runner,
                    "transform_paper",
                    return_value=output,
                ),
            ):
                with self.assertRaisesRegex(
                    pipeline_runner.PipelineRunnerError,
                    "元数据抽取失败.*--skip-meta",
                ):
                    pipeline_runner.main([
                        "--input-dir",
                        str(input_dir),
                        "--skip-ocr",
                        "--skip-organize",
                        "--skip-extraction",
                    ])

    def test_batch_id_is_resolved_after_mineru_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "pdfs"
            input_dir.mkdir()
            mineru_output = root / "mineru"
            mineru_output.mkdir()

            def finish_mineru(command, *, label):
                self.assertEqual(label, "MinerU")
                status = {
                    "data": {
                        "extract_result": [
                            {"file_name": "reference_no_0000001.pdf", "state": "done"}
                        ]
                    }
                }
                (mineru_output / "batch_remote_status.json").write_text(
                    json.dumps(status), encoding="utf-8"
                )

            with (
                patch.object(pipeline_runner, "run_command", side_effect=finish_mineru),
                patch.object(
                    pipeline_runner,
                    "reorganize_paper",
                    return_value={"errors": []},
                ) as reorganize,
            ):
                result = pipeline_runner.main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--mineru-output",
                        str(mineru_output),
                        "--batch-id",
                        "remote",
                        "--skip-transform",
                        "--skip-extraction",
                    ]
                )

            self.assertEqual(result, 0)
            reorganize.assert_called_once_with(
                mineru_output.resolve(),
                pipeline_runner.PROJECT_ROOT / "wenxian",
                "reference_no_0000001",
            )


if __name__ == "__main__":
    unittest.main()
