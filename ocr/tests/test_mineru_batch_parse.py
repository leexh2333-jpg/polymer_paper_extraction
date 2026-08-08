import argparse
import os
import tempfile
import unittest

import requests
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from mineru_batch_parse import (
    build_manifest,
    extract_and_delete_zip,
    load_env_file,
    make_data_id,
    safe_extract_zip,
    select_pdfs,
    upload_files,
)


class MineruBatchParseTests(unittest.TestCase):
    def test_build_manifest_records_ocr_options_without_credentials(self) -> None:
        args = argparse.Namespace(
            model_version="vlm",
            ocr=True,
            language="en",
            page_ranges="1-3",
            disable_formula=False,
            disable_table=True,
            extra_formats=["html"],
        )
        manifest = build_manifest(
            "batch-1",
            Path("input"),
            [Path("input") / "paper.pdf"],
            args,
        )

        self.assertEqual(manifest["batch_id"], "batch-1")
        self.assertTrue(manifest["ocr_enabled"])
        self.assertEqual(manifest["language"], "en")
        self.assertEqual(manifest["page_ranges"], "1-3")
        self.assertTrue(manifest["enable_formula"])
        self.assertFalse(manifest["enable_table"])
        self.assertEqual(manifest["extra_formats"], ["html"])
        self.assertNotIn("MINERU_API_KEY", manifest)
        self.assertNotIn("upload_urls", manifest)

    def test_load_env_file_does_not_overwrite_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text('MINERU_API_KEY="from-file"\n', encoding="utf-8")
            with patch.dict(os.environ, {"MINERU_API_KEY": "existing"}, clear=False):
                load_env_file(env_path)
                self.assertEqual(os.environ["MINERU_API_KEY"], "existing")

    def test_make_data_id_is_valid_and_bounded(self) -> None:
        data_id = make_data_id(Path("含 空格@符号" + "x" * 200 + ".pdf"))
        self.assertLessEqual(len(data_id), 128)
        self.assertRegex(data_id, r"^[A-Za-z0-9_.-]+$")

    def test_upload_files_retries_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "demo.pdf"
            pdf_path.write_bytes(b"pdf-data")
            response = Mock(status_code=200)
            session = Mock()
            session.put.side_effect = [requests.ReadTimeout("temporary timeout"), response]
            with patch("mineru_batch_parse.time.sleep") as sleep:
                upload_files(session, [pdf_path], ["https://upload.example/demo.pdf"])
            self.assertEqual(session.put.call_count, 2)
            sleep.assert_called_once_with(1)
            response.raise_for_status.assert_called_once_with()

    def test_extract_and_delete_zip_removes_archive_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "result.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("full.md", "parsed")
            output_dir = root / "result"
            extract_and_delete_zip(zip_path, output_dir)
            self.assertFalse(zip_path.exists())
            self.assertEqual((output_dir / "full.md").read_text(), "parsed")

    def test_safe_extract_zip_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "bad.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(RuntimeError):
                safe_extract_zip(zip_path, root / "output")

    def test_select_pdfs_applies_max_after_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("reference_no_0000003.pdf", "reference_no_0000001.pdf"):
                (root / name).write_bytes(b"pdf")

            selected = select_pdfs(root, max_documents=1)

            self.assertEqual([path.name for path in selected], ["reference_no_0000001.pdf"])

    def test_select_pdfs_preserves_explicit_order_and_rejects_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("reference_no_0000001.pdf", "reference_no_0000002.pdf"):
                (root / name).write_bytes(b"pdf")

            selected = select_pdfs(
                root,
                ref_nos=["reference_no_0000002", "reference_no_0000001.pdf"],
            )
            self.assertEqual(
                [path.stem for path in selected],
                ["reference_no_0000002", "reference_no_0000001"],
            )
            with self.assertRaises(FileNotFoundError):
                select_pdfs(root, ref_nos=["reference_no_9999999"])


if __name__ == "__main__":
    unittest.main()


