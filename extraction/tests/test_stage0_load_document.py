import json
import sys
import tempfile
import unittest
from pathlib import Path


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTRACTION_ROOT))

from schema.polymer_schema import SourceDocument, Stage0Document
from stages.stage0_load_document import (
    Stage0Error,
    load_document_elements,
    run_stage0,
)


def element(
    block_id: str,
    element_type: str,
    index: int,
    **extra: object,
) -> dict[str, object]:
    return {
        "block_id": block_id,
        "page_id": index // 10,
        "block_index": index,
        "element_type": element_type,
        "bbox": [1, 2, 3, 4],
        "alignment_status": "matched",
        **extra,
    }


def source_document(elements: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "document_id": "reference_no_0000001",
        "paper": {
            "ref_no": "reference_no_0000001",
            "pdf_filename": "uuid_origin.pdf",
            "source_pdf_path": "mineru_output/reference_no_0000001/uuid_origin.pdf",
            "organized_pdf_path": "wenxian/reference_no_0000001/origin.pdf",
            "doi": None,
            "title": "Demo",
            "authors": ["A. Author"],
            "journal": "Journal",
            "year": 2026,
            "metadata_status": "partial",
            "metadata_extraction": {"status": "success"},
        },
        "source_files": {},
        "ocr": {"status": "done"},
        "elements": elements,
        "warnings": [{"code": "source_warning", "message": "kept"}],
    }


class Stage0Tests(unittest.TestCase):
    def test_run_stage0_writes_null_doi(self) -> None:
        raw = source_document([
            element("P_0_0", "text", 0, text="Demo polymer"),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            document_path = root / "reference_no_0000001_document.json"
            document_path.write_text(
                json.dumps(raw),
                encoding="utf-8",
            )

            output_path, cached = run_stage0(document_path, root / "output")
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertFalse(cached)
        self.assertIn("doi", written["paper"])
        self.assertIsNone(written["paper"]["doi"])

    def test_rebuilds_sections_and_filters_references(self) -> None:
        raw = source_document([
            element(
                "P_0_0",
                "title",
                0,
                text="Demo",
                title_level=1,
                section="DocumentTitle",
            ),
            element("P_0_1", "title", 1, text="SUMMARY:", title_level=2),
            element("P_0_2", "text", 2, text="Abstract text"),
            element("P_0_3", "title", 3, text="2. Experimental", title_level=2),
            element("P_0_4", "title", 4, text="Materials", title_level=3),
            element("P_0_5", "text", 5, text="Method text"),
            element("P_0_6", "title", 6, text="Results and Discussion", title_level=2),
            element("T_0_7", "table", 7, table_body="<table></table>"),
            element("P_0_8", "title", 8, text="Summary", title_level=2),
            element("P_0_9", "text", 9, text="Conclusion text"),
            element("R_1_10", "references", 10, text="Reference entry"),
            element("P_1_11", "title", 11, text="References", title_level=2),
            element(
                "E_1_12",
                "equation",
                12,
                text="$$x=1$$",
                equation_kind="display",
            ),
            element(
                "E_1_13",
                "equation",
                13,
                text="$x$",
                equation_kind="inline",
            ),
        ])
        source = SourceDocument.model_validate(raw)

        result = load_document_elements(source)
        dumped = result.model_dump(mode="json")
        Stage0Document.model_validate(dumped)
        by_id = {item["block_id"]: item for item in dumped["elements"]}

        self.assertNotIn("R_1_10", by_id)
        self.assertEqual(by_id["P_0_0"]["section"], "DocumentTitle")
        self.assertEqual(by_id["P_0_2"]["section"], "Abstract")
        self.assertEqual(by_id["P_0_4"]["section"], "Methods")
        self.assertEqual(by_id["P_0_5"]["section"], "Methods")
        self.assertEqual(by_id["T_0_7"]["section"], "Results")
        self.assertEqual(by_id["P_0_9"]["section"], "Conclusion")
        self.assertEqual(by_id["P_1_11"]["section"], "References")
        self.assertEqual(by_id["E_1_12"]["equation_kind"], "display")
        self.assertNotIn("E_1_13", by_id)
        self.assertEqual(
            {warning["code"] for warning in dumped["warnings"]},
            {
                "source_warning",
                "inline_equation_skipped",
                "table_grid_unavailable",
            },
        )
        self.assertEqual(dumped["schema_version"], "1.1")

    def test_table_cells_have_stable_grid_coordinates(self) -> None:
        raw = source_document([
            element(
                "T_0_0",
                "table",
                0,
                table_body=(
                    "<table><tr><th rowspan='2'>Sample</th>"
                    "<th colspan='2'>Value</th></tr>"
                    "<tr><th>A</th><th>B</th></tr>"
                    "<tr><td>S1</td><td>1</td><td>2</td></tr></table>"
                ),
            ),
        ])

        result = load_document_elements(SourceDocument.model_validate(raw))
        table = result.elements[0]
        cells = {cell.text: cell for cell in table.table_cells or []}

        self.assertEqual(cells["Sample"].cell_id, "T_0_0:r0000:c0000")
        self.assertEqual(cells["Sample"].row_span, 2)
        self.assertEqual(cells["Value"].column_span, 2)
        self.assertEqual(cells["A"].column_index, 1)
        self.assertEqual(cells["B"].column_index, 2)
        self.assertEqual(cells["S1"].row_index, 2)

    def test_real_heading_aliases_are_mapped(self) -> None:
        raw = source_document([
            element("P_0_0", "title", 0, text="Synopsis", title_level=2),
            element("P_0_1", "text", 1, text="Abstract text"),
            element(
                "P_0_2",
                "title",
                2,
                text="EXPERIMENTAL COMPARISON",
                title_level=2,
            ),
            element("P_0_3", "text", 3, text="Result text"),
            element(
                "P_0_4",
                "title",
                4,
                text="SUMMARY AND CONCLUSIONS",
                title_level=2,
            ),
        ])

        result = load_document_elements(SourceDocument.model_validate(raw))
        by_id = {
            item.block_id: item
            for item in result.elements
        }

        self.assertEqual(by_id["P_0_1"].section, "Abstract")
        self.assertEqual(by_id["P_0_3"].section, "Results")
        self.assertEqual(by_id["P_0_4"].section, "Conclusion")
    def test_duplicate_block_id_fails_source_schema(self) -> None:
        duplicate = element("P_0_0", "text", 0, text="one")
        raw = source_document([duplicate, {**duplicate, "text": "two"}])

        with self.assertRaises(ValueError):
            SourceDocument.model_validate(raw)

    def test_missing_table_body_fails_stage0(self) -> None:
        raw = source_document([element("T_0_0", "table", 0)])
        source = SourceDocument.model_validate(raw)

        with self.assertRaises(Stage0Error):
            load_document_elements(source)


if __name__ == "__main__":
    unittest.main()
