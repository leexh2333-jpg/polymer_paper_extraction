import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from transform_mineru_to_standard import (
    CODE_ROOT,
    DEFAULT_PIPELINE_CONFIG,
    META_PROMPT_PATH,
    DocumentGateError,
    MetaExtractionResponse,
    transform_paper,
)


META_CONFIDENCE = {
    "score": 0.9,
    "field_scores": {
        "doi": 0.9,
        "title": 0.9,
        "authors": 0.9,
        "journal": 0.9,
        "year": 0.9,
    },
    "uncertain_fields": [],
    "evidence_basis": ["explicit_text", "exact_evidence_span"],
    "uncertainty_codes": [],
}


class FakeMetaExtractor:
    def extract(self, prompt: str, source_text: str) -> MetaExtractionResponse:
        if "以下 6 个字段" not in prompt:
            raise AssertionError("prompt 未加载")
        if "Demo Polymer Paper" not in source_text:
            raise AssertionError("首页文本未传入")
        return MetaExtractionResponse(
            data={
                "doi": "10.1000/demo",
                "title": "Demo Polymer Paper",
                "authors": ["A. Author"],
                "journal": "Polymer Journal",
                "year": 2026,
                "confidence": META_CONFIDENCE,
            },
            provider="test",
            model="fake-model",
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "billable_input_tokens": 100,
                "total_tokens": 120,
            },
            cost={
                "status": "calculated",
                "currency": "CNY",
                "input_per_million": "2",
                "output_per_million": "10",
                "input_cost": "0.0002",
                "output_cost": "0.0002",
                "total_cost": Decimal("0.0004"),
            },
        )


class NeverCalledMetaExtractor:
    def extract(self, prompt: str, source_text: str) -> MetaExtractionResponse:
        raise AssertionError("complete 元数据缓存不应重复调用")


class TransformMineruTests(unittest.TestCase):
    def test_runtime_resources_resolve_inside_code_root(self) -> None:
        self.assertEqual(
            META_PROMPT_PATH,
            CODE_ROOT / "extraction" / "prompts" / "meta_extract.md",
        )
        self.assertEqual(
            DEFAULT_PIPELINE_CONFIG,
            CODE_ROOT / "extraction" / "config" / "pipeline.yaml",
        )
        self.assertTrue(META_PROMPT_PATH.is_file())
        self.assertTrue(DEFAULT_PIPELINE_CONFIG.is_file())

    def _prepare_fixture(
        self,
        root: Path,
        *,
        state: str = "done",
    ) -> tuple[Path, Path, Path, str]:
        ref_no = "reference_no_0000001"
        mineru_output = root / "mineru_output"
        organized_root = root / "wenxian"
        processed_output = root / "processed_data"
        paper_dir = mineru_output / ref_no
        images_dir = paper_dir / f"{ref_no}_images"
        images_dir.mkdir(parents=True)

        blocks = [
            {
                "type": "text",
                "text": "Demo Polymer Paper",
                "text_level": 1,
                "page_idx": 0,
                "bbox": [1, 2, 30, 4],
            },
            {
                "type": "text",
                "text": "Value $x$ is reported.",
                "page_idx": 0,
                "bbox": [1, 5, 30, 9],
            },
            {
                "type": "table",
                "table_caption": ["Table 1. Demo"],
                "table_body": "<table><tr><td>old</td></tr></table>",
                "img_path": "images/table.jpg",
                "page_idx": 0,
                "bbox": [1, 10, 30, 20],
            },
            {
                "type": "equation",
                "text": "$$x=1$$",
                "page_idx": 0,
                "bbox": [1, 21, 30, 24],
            },
            {
                "type": "header",
                "text": "Polymer Journal 1 (2026)",
                "page_idx": 0,
                "bbox": [1, 0, 30, 1],
            },
        ]
        v2 = [[
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Demo Polymer Paper"}],
                    "level": 1,
                },
                "bbox": [1, 2, 30, 4],
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Value "},
                        {"type": "equation_inline", "content": "x"},
                        {"type": "text", "content": " is reported."},
                    ]
                },
                "bbox": [1, 5, 30, 9],
            },
            {
                "type": "table",
                "content": {"table_content": []},
                "bbox": [1, 10, 30, 20],
            },
            {
                "type": "equation_interline",
                "content": {"equation_interline_content": "$$x=1$$"},
                "bbox": [1, 21, 30, 24],
            },
        ]]
        (paper_dir / "uuid_content_list.json").write_text(
            json.dumps(blocks),
            encoding="utf-8",
        )
        (paper_dir / "uuid_content_list_v2.json").write_text(
            json.dumps(v2),
            encoding="utf-8",
        )
        table_html = "<table>\n<tr><td>new</td></tr>\n</table>"
        (paper_dir / f"{ref_no}.md").write_text(
            f"# Demo Polymer Paper\n\n{table_html}\n\n![]({ref_no}_images/table.jpg)",
            encoding="utf-8",
        )
        (paper_dir / "uuid_origin.pdf").write_bytes(b"%PDF-1.4")
        (images_dir / "table.jpg").write_bytes(b"image")

        batch_id = "batch-1"
        (mineru_output / f"batch_{batch_id}_manifest.json").write_text(
            json.dumps({
                "batch_id": batch_id,
                "files": [str(root / f"{ref_no}.pdf")],
                "model_version": "vlm",
                "ocr_enabled": False,
                "language": None,
                "page_ranges": None,
                "enable_formula": True,
                "enable_table": True,
                "extra_formats": [],
            }),
            encoding="utf-8",
        )
        (mineru_output / f"batch_{batch_id}_status.json").write_text(
            json.dumps({
                "code": 0,
                "data": {
                    "batch_id": batch_id,
                    "extract_result": [{
                        "file_name": f"{ref_no}.pdf",
                        "state": state,
                        "full_zip_url": "https://secret.example/result.zip",
                    }],
                },
            }),
            encoding="utf-8",
        )

        organized = organized_root / ref_no
        (organized / "images").mkdir(parents=True)
        (organized / "content.json").write_text(
            json.dumps({
                "document_id": ref_no,
                "source": "mineru",
                "blocks": [
                    blocks[0],
                    blocks[1],
                    {**blocks[2], "img_path": "images/table1.jpg"},
                    blocks[3],
                    blocks[4],
                ],
            }),
            encoding="utf-8",
        )
        (organized / "content_v2.json").write_text(json.dumps(v2), encoding="utf-8")
        (organized / f"{ref_no}.md").write_text(table_html, encoding="utf-8")
        (organized / "origin.pdf").write_bytes(b"%PDF-1.4")
        (organized / "images" / "table1.jpg").write_bytes(b"image")
        return mineru_output, organized_root, processed_output, ref_no

    def test_transform_writes_structured_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mineru_output, organized_root, processed_output, ref_no = (
                self._prepare_fixture(root)
            )

            output = transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                ref_no,
                meta_extractor=FakeMetaExtractor(),
            )
            document = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(document["document_id"], ref_no)
            self.assertEqual(document["paper"]["metadata_status"], "complete")
            self.assertEqual(document["paper"]["metadata_extraction"]["model"], "fake-model")
            self.assertEqual(
                document["paper"]["metadata_extraction"]["confidence"]["score"],
                0.9,
            )
            self.assertEqual(
                document["paper"]["metadata_extraction"]["cost"]["total_cost"],
                "0.0004",
            )
            self.assertEqual(document["ocr"]["status"], "done")
            self.assertNotIn("full_zip_url", json.dumps(document))
            self.assertEqual(
                [element["element_type"] for element in document["elements"]],
                ["title", "text", "table", "equation"],
            )
            self.assertEqual(document["elements"][1]["text"], "Value $x$ is reported.")
            self.assertEqual(
                document["elements"][2]["table_body"],
                "<table>\n<tr><td>new</td></tr>\n</table>",
            )
            self.assertEqual(
                document["elements"][2]["image_path"],
                f"wenxian/{ref_no}/images/table1.jpg",
            )
            self.assertEqual(document["elements"][3]["equation_kind"], "display")

    def test_missing_meta_extractor_does_not_block_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mineru_output, organized_root, processed_output, ref_no = (
                self._prepare_fixture(root)
            )

            output = transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                ref_no,
            )
            document = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(document["paper"]["metadata_status"], "failed")
            self.assertIsNone(document["paper"]["title"])
            self.assertIn(
                "metadata_extraction_failed",
                {warning["code"] for warning in document["warnings"]},
            )

    def test_complete_metadata_is_reused_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mineru_output, organized_root, processed_output, ref_no = (
                self._prepare_fixture(root)
            )
            output = transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                ref_no,
                meta_extractor=FakeMetaExtractor(),
            )

            transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                ref_no,
                meta_extractor=NeverCalledMetaExtractor(),
            )
            cached = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(cached["paper"]["metadata_status"], "complete")

            transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                ref_no,
                force_meta=True,
                meta_extractor=FakeMetaExtractor(),
            )
            forced = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(forced["paper"]["metadata_status"], "complete")

    def test_non_done_status_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mineru_output, organized_root, processed_output, ref_no = (
                self._prepare_fixture(root, state="failed")
            )

            with self.assertRaises(DocumentGateError):
                transform_paper(
                    mineru_output,
                    organized_root,
                    processed_output,
                    ref_no,
                )


if __name__ == "__main__":
    unittest.main()
