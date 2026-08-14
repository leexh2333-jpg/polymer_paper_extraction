"""Evaluate demo20 polymer/material types without entity-name matching.

The frozen PolyInfo ground truth deliberately loads sample records by JSON
structure instead of filename prefix.  This keeps ``BD*.json`` blend records in
scope and makes the exact GT inputs auditable in the result JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF_LIST = PROJECT_ROOT / "preview" / "demo_latest_20_refs.txt"
DEFAULT_GT_DIR = PROJECT_ROOT / "polyinfo数据" / "有doi"
DEFAULT_GT_WORKBOOK = (
    PROJECT_ROOT / "polyinfo数据" / "sample_exprot_34" / "sample_exprot_34.xlsx"
)
GT_SHEET = "sample_export"
IGNORED_GT_JSON = {"progress_state.json", "run_manifest.json"}

EXPECTED_GT = {
    "sample_rows": 222,
    "unique_polymer_records": 137,
    "polymer_type_rows": {
        "Homopolymer": 139,
        "Copolymer": 28,
        "Blend": 55,
    },
    "material_type_labels": {
        "Neat resin": 181,
        "Compound": 37,
        "Composite": 8,
    },
    "material_type_label_total": 226,
}

POLYMER_TYPE_MAP = {
    "homopolymer": "Homopolymer",
    "copolymer": "Copolymer",
    "polymer_blend": "Blend",
}
MATERIAL_TYPE_MAP = {
    "neat_resin": "Neat resin",
    "compound": "Compound",
    "composite": "Composite",
    "inorganic_polymer": "Inorganic polymer",
}


class EvaluationError(RuntimeError):
    """The frozen evaluation inputs or predictions are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        raise EvaluationError(f"无效的 XLSX 单元格引用：{cell_reference}")
    index = 0
    for character in letters.group(0):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _xlsx_sheet_rows(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    """Read a flat XLSX sheet using only the Python standard library."""

    namespaces = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "p": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("m:si", namespaces):
                shared_strings.append("".join(
                    node.text or "" for node in item.findall(".//m:t", namespaces)
                ))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id: str | None = None
        for sheet in workbook.findall("m:sheets/m:sheet", namespaces):
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{namespaces['r']}}}id")
                break
        if relationship_id is None:
            raise EvaluationError(f"工作簿缺少 sheet：{sheet_name}")

        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target: str | None = None
        for relation in relationships.findall("p:Relationship", namespaces):
            if relation.attrib.get("Id") == relationship_id:
                target = relation.attrib.get("Target")
                break
        if target is None:
            raise EvaluationError(f"无法解析 sheet 路径：{sheet_name}")
        target = target.replace("\\", "/").lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheet_root = ET.fromstring(archive.read(sheet_path))

        matrix: list[list[Any]] = []
        for row in sheet_root.findall("m:sheetData/m:row", namespaces):
            values: dict[int, Any] = {}
            for cell in row.findall("m:c", namespaces):
                reference = cell.attrib.get("r", "")
                column = _column_index(reference)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", namespaces)
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//m:t", namespaces)
                    )
                elif value_node is None:
                    value = None
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text or "0")]
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    raw = value_node.text or ""
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                values[column] = value
            if values:
                width = max(values) + 1
                matrix.append([values.get(index) for index in range(width)])

    if not matrix:
        return []
    headers = [str(value or "").strip() for value in matrix[0]]
    return [
        {
            header: row[index] if index < len(row) else None
            for index, header in enumerate(headers)
            if header
        }
        for row in matrix[1:]
    ]


def _normalized_ref_no(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("reference_no_"):
        return text
    try:
        return f"reference_no_{int(float(text)):07d}"
    except ValueError as exc:
        raise EvaluationError(f"无效 reference_no：{value!r}") from exc


def load_ref_nos(path: Path) -> list[str]:
    refs = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    refs = [ref for ref in refs if ref]
    if len(refs) != len(set(refs)):
        raise EvaluationError("ref-list 包含重复文档")
    return refs


def _record_material_types(record: dict[str, Any]) -> list[str]:
    value = record.get("material_type")
    if value is None:
        return []
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def load_ground_truth(
    ref_nos: Iterable[str],
    gt_dir: Path,
    workbook_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = list(ref_nos)
    ref_set = set(refs)
    records: list[dict[str, Any]] = []
    loaded_files: list[dict[str, Any]] = []
    per_document_sources: dict[str, list[str]] = defaultdict(list)

    missing_from_directory: set[str] = set()
    for ref_no in refs:
        doc_dir = gt_dir / ref_no
        if not doc_dir.is_dir():
            missing_from_directory.add(ref_no)
            continue
        for path in sorted(doc_dir.glob("*.json")):
            if path.name in IGNORED_GT_JSON:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"GT JSON 无效：{path}") from exc
            if not isinstance(record, dict) or "sample_id" not in record:
                continue
            if "polymer_id" not in record:
                raise EvaluationError(f"GT sample 缺少 polymer_id：{path}")
            records.append({"document_id": ref_no, **record})
            resolved = str(path.resolve())
            per_document_sources[ref_no].append(resolved)
            loaded_files.append({"path": resolved, "sha256": _sha256(path)})

    workbook_rows = _xlsx_sheet_rows(workbook_path, GT_SHEET)
    workbook_counts: Counter[str] = Counter()
    for row in workbook_rows:
        ref_no = _normalized_ref_no(row.get("reference_no"))
        if ref_no not in ref_set or ref_no not in missing_from_directory:
            continue
        raw_json = row.get("sample_json")
        if not isinstance(raw_json, str):
            raise EvaluationError(f"{ref_no} 的 sample_json 不是字符串")
        try:
            record = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{ref_no} 的 sample_json 无效") from exc
        if not isinstance(record, dict):
            raise EvaluationError(f"{ref_no} 的 sample_json 不是对象")
        record.setdefault("sample_id", row.get("sample_id"))
        record.setdefault("polymer_id", row.get("polymer_id"))
        if not record.get("sample_id") or not record.get("polymer_id"):
            raise EvaluationError(f"{ref_no} 的工作簿记录缺少 sample/polymer ID")
        records.append({"document_id": ref_no, **record})
        workbook_counts[ref_no] += 1

    unresolved = sorted(missing_from_directory - set(workbook_counts))
    if unresolved:
        raise EvaluationError("GT 缺少文档：" + ", ".join(unresolved))
    workbook_resolved = str(workbook_path.resolve())
    if workbook_counts:
        loaded_files.append({"path": workbook_resolved, "sha256": _sha256(workbook_path)})
        for ref_no in workbook_counts:
            per_document_sources[ref_no].append(workbook_resolved)

    per_document_rows = Counter(record["document_id"] for record in records)
    manifest = {
        "ref_list": [str(ref) for ref in refs],
        "sheet": GT_SHEET,
        "files": loaded_files,
        "documents": {
            ref_no: {
                "rows": per_document_rows[ref_no],
                "sources": per_document_sources[ref_no],
            }
            for ref_no in refs
        },
    }
    return records, manifest


def summarize_ground_truth(records: list[dict[str, Any]]) -> dict[str, Any]:
    polymer_rows = Counter(str(record.get("polymer_type")) for record in records)
    material_labels: Counter[str] = Counter()
    unique_polymers: dict[tuple[str, str], str] = {}
    for record in records:
        document_id = str(record["document_id"])
        polymer_id = str(record["polymer_id"])
        polymer_type = str(record.get("polymer_type"))
        key = (document_id, polymer_id)
        existing = unique_polymers.setdefault(key, polymer_type)
        if existing != polymer_type:
            raise EvaluationError(f"同一 GT polymer_id 类型冲突：{key}")
        material_labels.update(_record_material_types(record))
    return {
        "sample_rows": len(records),
        "unique_polymer_records": len(unique_polymers),
        "polymer_type_rows": dict(sorted(polymer_rows.items())),
        "polymer_type_unique": dict(sorted(Counter(unique_polymers.values()).items())),
        "material_type_labels": dict(sorted(material_labels.items())),
        "material_type_label_total": sum(material_labels.values()),
    }


def assert_frozen_ground_truth(summary: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in EXPECTED_GT.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise EvaluationError(
            "GT 加载异常：" + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_predictions(
    records: list[dict[str, Any]],
    ref_nos: Iterable[str],
    prediction_dir: Path,
) -> dict[str, Any]:
    refs = list(ref_nos)
    gt_polymer_by_doc: dict[str, dict[str, str]] = defaultdict(dict)
    gt_material_by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        ref_no = str(record["document_id"])
        gt_polymer_by_doc[ref_no][str(record["polymer_id"])] = str(
            record.get("polymer_type")
        )
        gt_material_by_doc[ref_no].update(_record_material_types(record))

    predicted_polymer: Counter[str] = Counter()
    predicted_material: Counter[str] = Counter()
    entity_total = sample_total = 0
    entity_abstain = sample_abstain = 0
    predicted_blend_entity_docs: set[str] = set()
    predicted_blend_sample_docs: set[str] = set()
    per_document: dict[str, Any] = {}
    for ref_no in refs:
        final_path = prediction_dir / ref_no / "final.json"
        if not final_path.is_file():
            raise EvaluationError(f"缺少预测文件：{final_path}")
        try:
            final = json.loads(final_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"预测 JSON 无效：{final_path}") from exc
        entities = final.get("polymer_entities") or []
        samples = final.get("samples") or []
        doc_polymer: Counter[str] = Counter()
        doc_sample_polymer: Counter[str] = Counter()
        doc_material: Counter[str] = Counter()
        for entity in entities:
            entity_total += 1
            raw_type = entity.get("polymer_type")
            if raw_type is None:
                entity_abstain += 1
                continue
            label = POLYMER_TYPE_MAP.get(str(raw_type), str(raw_type))
            predicted_polymer[label] += 1
            doc_polymer[label] += 1
        for sample in samples:
            sample_total += 1
            raw_polymer_type = sample.get("polymer_type")
            if raw_polymer_type is not None:
                polymer_label = POLYMER_TYPE_MAP.get(
                    str(raw_polymer_type), str(raw_polymer_type)
                )
                doc_sample_polymer[polymer_label] += 1
            raw_type = sample.get("material_type")
            if raw_type is None:
                sample_abstain += 1
                continue
            label = MATERIAL_TYPE_MAP.get(str(raw_type), str(raw_type))
            predicted_material[label] += 1
            doc_material[label] += 1
        if doc_polymer["Blend"]:
            predicted_blend_entity_docs.add(ref_no)
        if doc_sample_polymer["Blend"]:
            predicted_blend_sample_docs.add(ref_no)
        per_document[ref_no] = {
            "gt_unique_polymer_types": dict(sorted(Counter(
                gt_polymer_by_doc[ref_no].values()
            ).items())),
            "predicted_entity_types": dict(sorted(doc_polymer.items())),
            "predicted_sample_polymer_types": dict(
                sorted(doc_sample_polymer.items())
            ),
            "gt_material_type_labels": dict(sorted(gt_material_by_doc[ref_no].items())),
            "predicted_sample_material_types": dict(sorted(doc_material.items())),
            "prediction_path": str(final_path.resolve()),
            "prediction_sha256": _sha256(final_path),
        }

    gt_blend_docs = {
        ref_no
        for ref_no, values in gt_polymer_by_doc.items()
        if "Blend" in values.values()
    }
    predicted_blend_any_docs = (
        predicted_blend_entity_docs | predicted_blend_sample_docs
    )

    def blend_detection(predicted_docs: set[str]) -> dict[str, Any]:
        true_positive_docs = gt_blend_docs & predicted_docs
        return {
            "gt_documents": sorted(gt_blend_docs),
            "predicted_documents": sorted(predicted_docs),
            "true_positive_documents": sorted(true_positive_docs),
            "precision": _safe_ratio(len(true_positive_docs), len(predicted_docs)),
            "recall": _safe_ratio(len(true_positive_docs), len(gt_blend_docs)),
        }

    gt_summary = summarize_ground_truth(records)
    return {
        "prediction_dir": str(prediction_dir.resolve()),
        "totals": {
            "predicted_entities": entity_total,
            "predicted_samples": sample_total,
            "entity_type_abstentions": entity_abstain,
            "sample_material_type_abstentions": sample_abstain,
        },
        "rates": {
            "entity_type_abstention": _safe_ratio(entity_abstain, entity_total),
            "sample_material_type_abstention": _safe_ratio(sample_abstain, sample_total),
            "entity_to_gt_unique_granularity": _safe_ratio(
                entity_total, gt_summary["unique_polymer_records"]
            ),
            "sample_to_gt_row_granularity": _safe_ratio(
                sample_total, gt_summary["sample_rows"]
            ),
        },
        "class_counts": {
            "gt_unique_polymer_types": gt_summary["polymer_type_unique"],
            "predicted_entity_types": dict(sorted(predicted_polymer.items())),
            "gt_material_type_labels": gt_summary["material_type_labels"],
            "predicted_sample_material_types": dict(sorted(predicted_material.items())),
        },
        "majority_baselines": {
            "polymer_type_homopolymer": _safe_ratio(
                gt_summary["polymer_type_unique"].get("Homopolymer", 0),
                gt_summary["unique_polymer_records"],
            ),
            "material_type_neat_resin": _safe_ratio(
                gt_summary["material_type_labels"].get("Neat resin", 0),
                gt_summary["sample_rows"],
            ),
        },
        "blend_document_detection": blend_detection(
            predicted_blend_entity_docs
        ),
        "blend_document_detection_entity_or_sample": blend_detection(
            predicted_blend_any_docs
        ),
        "per_document": per_document,
    }


def build_result(
    *,
    ref_list: Path,
    gt_dir: Path,
    workbook_path: Path,
    prediction_dir: Path,
) -> dict[str, Any]:
    ref_nos = load_ref_nos(ref_list)
    records, manifest = load_ground_truth(ref_nos, gt_dir, workbook_path)
    gt_summary = summarize_ground_truth(records)
    assert_frozen_ground_truth(gt_summary)
    manifest["ref_list_path"] = str(ref_list.resolve())
    manifest["ref_list_sha256"] = _sha256(ref_list)
    manifest["assertions"] = EXPECTED_GT
    return {
        "schema_version": "demo20_type_evaluation.v1",
        "ground_truth_manifest": manifest,
        "ground_truth_summary": gt_summary,
        "evaluation": evaluate_predictions(records, ref_nos, prediction_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测 demo20 聚合物/材料类型")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref-list", type=Path, default=DEFAULT_REF_LIST)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--gt-workbook", type=Path, default=DEFAULT_GT_WORKBOOK)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_result(
        ref_list=args.ref_list.resolve(),
        gt_dir=args.gt_dir.resolve(),
        workbook_path=args.gt_workbook.resolve(),
        prediction_dir=args.prediction_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写入评测结果：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
