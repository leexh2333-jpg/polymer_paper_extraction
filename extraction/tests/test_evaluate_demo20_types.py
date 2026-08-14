import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.evaluate_demo20_types import (
    EvaluationError,
    _xlsx_sheet_rows,
    assert_frozen_ground_truth,
    evaluate_predictions,
    load_ground_truth,
)


class Demo20TypeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_workbook(self, path: Path) -> None:
        content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>"""
        workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="sample_export" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
        relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Target="worksheets/sheet1.xml"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
</Relationships>"""
        values = [
            "reference_no", "polymer_id", "sample_id", "sample_json",
            "1", "BD000001", "0000001-001", json.dumps({
                "sample_id": "0000001-001",
                "polymer_id": "BD000001",
                "polymer_type": "Blend",
                "material_type": ["Compound"],
            }),
        ]
        shared = "".join(f"<si><t>{value}</t></si>" for value in values)
        shared_strings = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"{shared}</sst>"
        )
        sheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <sheetData>
  <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>
  <row r="2"><c r="A2" t="s"><v>4</v></c><c r="B2" t="s"><v>5</v></c><c r="C2" t="s"><v>6</v></c><c r="D2" t="s"><v>7</v></c></row>
 </sheetData>
</worksheet>"""
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr("xl/_rels/workbook.xml.rels", relationships)
            archive.writestr("xl/sharedStrings.xml", shared_strings)
            archive.writestr("xl/worksheets/sheet1.xml", sheet)

    def test_xlsx_reader_returns_full_sample_json(self) -> None:
        workbook = self.root / "gt.xlsx"
        self._write_workbook(workbook)

        rows = _xlsx_sheet_rows(workbook, "sample_export")

        self.assertEqual(rows[0]["reference_no"], "1")
        self.assertIn('"polymer_id": "BD000001"', rows[0]["sample_json"])

    def test_gt_loader_enumerates_bd_json_by_structure(self) -> None:
        gt_dir = self.root / "gt"
        doc_dir = gt_dir / "reference_no_0000001"
        doc_dir.mkdir(parents=True)
        (doc_dir / "BD000001_001.json").write_text(json.dumps({
            "sample_id": "0000001-001",
            "polymer_id": "BD000001",
            "polymer_type": "Blend",
            "material_type": ["Compound"],
        }), encoding="utf-8")
        (doc_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        workbook = self.root / "gt.xlsx"
        self._write_workbook(workbook)

        records, manifest = load_ground_truth(
            ["reference_no_0000001"], gt_dir, workbook
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["polymer_id"], "BD000001")
        self.assertEqual(manifest["documents"]["reference_no_0000001"]["rows"], 1)

    def test_frozen_assertions_fail_loudly(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "GT 加载异常"):
            assert_frozen_ground_truth({"sample_rows": 221})

    def test_blend_document_detection_reports_entity_and_sample_scopes(self) -> None:
        ref_no = "reference_no_0000001"
        prediction_dir = self.root / "predictions"
        document_dir = prediction_dir / ref_no
        document_dir.mkdir(parents=True)
        (document_dir / "final.json").write_text(json.dumps({
            "polymer_entities": [{"polymer_type": "copolymer"}],
            "samples": [{
                "polymer_type": "polymer_blend",
                "material_type": "compound",
            }],
        }), encoding="utf-8")
        records = [{
            "document_id": ref_no,
            "polymer_id": "BD000001",
            "polymer_type": "Blend",
            "material_type": ["Compound"],
        }]

        result = evaluate_predictions(records, [ref_no], prediction_dir)

        self.assertEqual(result["blend_document_detection"]["recall"], 0.0)
        self.assertEqual(
            result["blend_document_detection_entity_or_sample"]["recall"],
            1.0,
        )
        self.assertEqual(
            result["per_document"][ref_no]["predicted_sample_polymer_types"],
            {"Blend": 1},
        )


if __name__ == "__main__":
    unittest.main()
