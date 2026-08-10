"""Preview-only Stage 4R：从明确缺失的表格格子确定性补入性质值。

默认只写 stage4_properties.recovery_preview.json，不覆盖 Stage 4。使用 --apply
时会先备份原文件，再写回 stage4_properties.json。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from schema.polymer_schema import Stage0Document, Stage4Document
from stages.table_grid import table_cells_for
from stages.table_recall_audit import audit_documents

STAGE_ID = "stage4r_table_recovery"
IMPLEMENTATION_VERSION = "0.1.0"
_ID_RE = re.compile(r"^uprop(\d+)$")


def _norm(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _cell_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        locator = value.get("table_locator")
        if isinstance(locator, Mapping) and isinstance(locator.get("cell_id"), str):
            result.add(locator["cell_id"])
        for child in value.values():
            result.update(_cell_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_cell_ids(child))
    return result


def next_unresolved_number(stage4: Mapping[str, Any]) -> int:
    numbers = []
    for item in stage4.get("unresolved_properties") or []:
        match = _ID_RE.fullmatch(str(item.get("unresolved_id") or ""))
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _sample_entity_map(stage3: Mapping[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    samples = [item for item in stage3.get("samples") or [] if isinstance(item, Mapping)]
    mapping = {
        str(item["sample_id"]): str(item["refers_to_entity"])
        for item in samples
        if item.get("sample_id") and item.get("refers_to_entity")
    }
    return mapping, [dict(item) for item in samples]


def _entity_aliases(stage2: Mapping[str, Any]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for item in stage2.get("polymer_entities") or []:
        if not isinstance(item, Mapping) or not item.get("entity_id"):
            continue
        entity_id = str(item["entity_id"])
        for key in ("canonical_name", "normalized_name", "display_name"):
            if item.get(key):
                aliases[entity_id].add(str(item[key]))
        for value in item.get("source_names") or []:
            if value:
                aliases[entity_id].add(str(value))
    return aliases


def _table_entities(stage4: Mapping[str, Any], sample_entities: Mapping[str, str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in [
        *(stage4.get("properties") or []),
        *(stage4.get("unresolved_properties") or []),
        *(stage4.get("property_series") or []),
    ]:
        if not isinstance(item, Mapping):
            continue
        entity = item.get("entity_id") or sample_entities.get(str(item.get("sample_id") or ""))
        if not entity:
            continue
        for cell_id in _cell_ids(item):
            result[cell_id.split(":", 1)[0]].add(str(entity))
    return result



def _row_label_candidates(
    cells: Sequence[Any],
    row_index: int,
    column_index: int,
) -> list[str]:
    """取该行位于目标列左侧的所有单元格文本，作为样品标签候选。

    审计器的 row_headers 会把"看起来像数值"的格子剔掉，而不少论文的样品编码
    恰好是纯数字加连字符（0-2、0-2-0-6、0-4-0-10），于是真正的样品标签被当成
    数值滤掉，infer_entity_id 拿到空的 row_headers 只能报 entity_ambiguous。
    这里绕过该过滤，直接按网格取左侧标签列。
    """
    return [
        cell.text.strip()
        for cell in sorted(
            (
                cell
                for cell in cells
                if cell.row_index <= row_index < cell.row_index + cell.row_span
                and cell.column_index < column_index
                and cell.text.strip()
            ),
            key=lambda item: item.column_index,
        )
    ]


def infer_entity_id(
    *,
    row_headers: Sequence[str],
    row_index: int | None = None,
    table_id: str,
    valid_entity_ids: set[str],
    samples: Sequence[Mapping[str, Any]],
    entity_aliases: Mapping[str, set[str]] | None = None,
    table_entities: Mapping[str, set[str]],
    row_label_candidates: Sequence[str] | None = None,
) -> tuple[str | None, str]:
    row_parts = [_norm(value) for value in row_headers if _norm(value)]
    aliases_by_entity: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        entity = str(sample.get("refers_to_entity") or "")
        if entity not in valid_entity_ids:
            continue
        for label in (sample.get("sample_label_raw"), sample.get("polymer_name")):
            if label_norm := _norm(label):
                aliases_by_entity[entity].add(label_norm)
    for entity, aliases in (entity_aliases or {}).items():
        if entity not in valid_entity_ids:
            continue
        aliases_by_entity[entity].update(
            alias_norm for alias in aliases if (alias_norm := _norm(alias))
        )

    exact_matches = {
        entity
        for entity, aliases in aliases_by_entity.items()
        if any(row_part == alias for row_part in row_parts for alias in aliases)
    }
    if len(exact_matches) == 1:
        return next(iter(exact_matches)), "row_label_exact_entity_alias"

    # 仅保留最长的包含匹配，避免 0-2 与 0-2-0-I 同时命中造成假歧义。
    best_length = 0
    best_matches: set[str] = set()
    for entity, aliases in aliases_by_entity.items():
        for row_part in row_parts:
            for alias in aliases:
                if len(alias) < 3 or not (alias in row_part or row_part in alias):
                    continue
                score = min(len(alias), len(row_part))
                if score > best_length:
                    best_length = score
                    best_matches = {entity}
                elif score == best_length:
                    best_matches.add(entity)
    row_text = _norm(" ".join(row_headers))
    mixture_markers = ("blend", "mixture", "composite", "共混", "混合", "复合")
    if len(best_matches) == 1 and not any(marker in row_text for marker in mixture_markers):
        return next(iter(best_matches)), "row_label_longest_entity_alias"

    sample_entity_ids = {
        str(sample.get("refers_to_entity"))
        for sample in samples
        if str(sample.get("refers_to_entity") or "") in valid_entity_ids
    }
    if not row_parts and len(sample_entity_ids) == 1:
        return next(iter(sample_entity_ids)), "document_samples_single_entity"

    existing = set(table_entities.get(table_id) or set()) & valid_entity_ids
    if not row_parts and len(existing) == 1:
        return next(iter(existing)), "table_existing_entity"
    if len(valid_entity_ids) == 1:
        return next(iter(valid_entity_ids)), "document_single_entity"

    # 兜底：审计器把纯数字样品编码（0-2-0-6）当数值滤掉了，用未过滤的
    # 左侧标签列重试一次严格匹配。只认精确相等，不做包含匹配 —— 包含匹配在
    # 这类编码上极易假阳（0-2 是 0-2-0-6 的子串）。
    extra_parts = [_norm(value) for value in (row_label_candidates or []) if _norm(value)]
    extra_parts = [part for part in extra_parts if part not in set(row_parts)]
    if extra_parts:
        extra_exact = {
            entity
            for entity, aliases in aliases_by_entity.items()
            if any(part == alias for part in extra_parts for alias in aliases)
        }
        if len(extra_exact) == 1:
            return next(iter(extra_exact)), "row_cell_exact_entity_alias"

    return None, "entity_ambiguous" if valid_entity_ids else "entity_not_found"


def _row_sentence(cells: Sequence[Any], row_index: int) -> str:
    row = sorted(
        (cell for cell in cells if cell.row_index == row_index and cell.text.strip()),
        key=lambda cell: cell.column_index,
    )
    return " | ".join(cell.text.strip() for cell in row)


def build_unresolved_property(
    *,
    unresolved_id: str,
    entity_id: str,
    cell_report: Mapping[str, Any],
    table: Any,
    cells: Sequence[Any],
) -> dict[str, Any]:
    column_headers = [str(item) for item in cell_report.get("column_headers") or [] if str(item).strip()]
    row_headers = [str(item) for item in cell_report.get("row_headers") or [] if str(item).strip()]
    # Preview 首版使用审计器已经确认的规范性质名，避免多层表头中的前序
    # 数据值或占位符被误写成 property_name_raw。
    property_name = str(cell_report["property_name_normalized"]).strip()
    row_index = int(cell_report["row_index"])
    column_index = int(cell_report["column_index"])
    value_raw = str(cell_report["text"]).strip()
    source_sentence = _row_sentence(cells, row_index) or value_raw
    bbox = list(table.bbox) if table.bbox is not None else None
    return {
        "unresolved_id": unresolved_id,
        "entity_id": entity_id,
        "property_name_raw": property_name,
        "value_raw": value_raw,
        "reason": "sample_ambiguous",
        "evidence": [{
            "block_id": table.block_id,
            "page": table.page,
            "bbox": bbox,
            "source_type": "table",
            "source_sentence": source_sentence,
            "table_locator": {
                "table_id": table.block_id,
                "cell_id": cell_report["cell_id"],
                "row_index": row_index,
                "column_index": column_index,
                "row_label": " / ".join(row_headers) or None,
                "column_label": " / ".join(column_headers) or None,
                "cell_value": value_raw,
            },
        }],
    }


def recover_document(
    stage0: Stage0Document,
    stage2: Mapping[str, Any],
    stage3: Mapping[str, Any],
    stage4: Mapping[str, Any],
    *,
    threshold: float = 0.8,
) -> tuple[Stage4Document, dict[str, Any]]:
    original = Stage4Document.model_validate(stage4)
    audit = audit_documents(stage0, stage4, threshold=threshold)
    valid_entities = {
        str(item.get("entity_id"))
        for item in stage2.get("polymer_entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }
    sample_entities, samples = _sample_entity_map(stage3)
    entity_aliases = _entity_aliases(stage2)
    table_entities = _table_entities(stage4, sample_entities)
    tables = {item.block_id: item for item in stage0.elements if item.type == "table"}
    existing_cells = _cell_ids(stage4)
    next_number = next_unresolved_number(stage4)
    additions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for table_report in audit.get("tables") or []:
        table_id = str(table_report["table_id"])
        table = tables.get(table_id)
        if table is None:
            continue
        cells = table_cells_for(table)
        for cell in table_report.get("missing_property_cells") or []:
            cell_id = str(cell["cell_id"])
            if cell_id in existing_cells:
                continue
            if not str(cell.get("role_reason") or "").startswith("property_header:"):
                skipped.append({
                    "cell_id": cell_id,
                    "table_id": table_id,
                    "property_name_normalized": cell.get("property_name_normalized"),
                    "value_raw": cell.get("text"),
                    "reason": "unsafe_non_header_property_hint",
                })
                continue
            entity_id, basis = infer_entity_id(
                row_headers=cell.get("row_headers") or [],
                row_index=cell.get("row_index"),
                table_id=table_id,
                valid_entity_ids=valid_entities,
                samples=samples,
                entity_aliases=entity_aliases,
                table_entities=table_entities,
                row_label_candidates=_row_label_candidates(
                    cells,
                    int(cell["row_index"]),
                    int(cell["column_index"]),
                ),
            )
            if entity_id is None:
                skipped.append({
                    "cell_id": cell_id,
                    "table_id": table_id,
                    "property_name_normalized": cell.get("property_name_normalized"),
                    "value_raw": cell.get("text"),
                    "reason": basis,
                })
                continue
            item = build_unresolved_property(
                unresolved_id=f"uprop{next_number:03d}",
                entity_id=entity_id,
                cell_report=cell,
                table=table,
                cells=cells,
            )
            next_number += 1
            additions.append(item)
            item["_entity_resolution_basis"] = basis
            existing_cells.add(cell_id)

    payload = original.model_dump(mode="json")
    payload["unresolved_properties"].extend(
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in additions
    )
    if additions:
        payload["warnings"].append({
            "code": "stage4r_preview_table_recovery",
            "message": f"Stage 4R Preview 确定性补入 {len(additions)} 个表格性质格；样品归属保留 unresolved。",
            "recovered_cell_count": len(additions),
            "skipped_ambiguous_count": len(skipped),
        })
    merged = Stage4Document.model_validate(payload)
    report = {
        "stage": STAGE_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "document_id": stage0.document_id,
        "audit_summary": audit.get("summary"),
        "recovered_count": len(additions),
        "skipped_ambiguous_count": len(skipped),
        "recovered": [
            {
                "unresolved_id": item["unresolved_id"],
                "entity_id": item["entity_id"],
                "property_name_raw": item["property_name_raw"],
                "value_raw": item["value_raw"],
                "cell_id": item["evidence"][0]["table_locator"]["cell_id"],
                "entity_resolution_basis": item["_entity_resolution_basis"],
            }
            for item in additions
        ],
        "skipped_ambiguous": skipped,
    }
    return merged, report


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _same_file_content(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()


def _prepare_stage4_input(
    *,
    stage4_path: Path,
    preview_path: Path,
    report_path: Path,
    backup_path: Path,
    apply: bool,
    force: bool,
    in_place: bool,
) -> tuple[Path, bool]:
    already_applied = apply and in_place and _same_file_content(stage4_path, preview_path)
    if not force and report_path.is_file() and preview_path.is_file():
        if not apply or already_applied:
            return stage4_path, True
    if not apply or not in_place:
        return stage4_path, False
    if already_applied and backup_path.is_file():
        return backup_path, False
    shutil.copy2(stage4_path, backup_path)
    return stage4_path, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ref-no", required=True)
    parser.add_argument("--config", type=Path, help="兼容批处理器；Stage 4R 不读取模型配置")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = (args.output_root or input_root).expanduser().resolve()
    ref_dir = args.ref_no if args.ref_no.startswith("reference_no_") else f"reference_no_{args.ref_no}"
    source = input_root / ref_dir
    target = output_root / ref_dir
    target.mkdir(parents=True, exist_ok=True)
    stage4_path = source / "stage4_properties.json"
    preview_path = target / "stage4_properties.recovery_preview.json"
    report_path = target / "stage4r_recovery.json"
    backup_path = target / "stage4_properties.pre_recovery.json"
    stage4_input_path, cached = _prepare_stage4_input(
        stage4_path=stage4_path,
        preview_path=preview_path,
        report_path=report_path,
        backup_path=backup_path,
        apply=bool(args.apply),
        force=bool(args.force),
        in_place=source.resolve() == target.resolve(),
    )
    if cached:
        print(f"[cached] {ref_dir}: Stage 4R Preview 已存在")
        return 0
    stage0 = Stage0Document.model_validate_json((source / "stage0_blocks.json").read_text(encoding="utf-8"))
    stage2 = _load_json(source / "stage2_entities.json")
    stage3 = _load_json(source / "stage3_process.json")
    stage4 = _load_json(stage4_input_path)
    merged, report = recover_document(stage0, stage2, stage3, stage4, threshold=args.threshold)
    _write_json(report_path, report)
    _write_json(preview_path, merged.model_dump(mode="json"))
    if args.apply:
        destination = target / "stage4_properties.json"
        _write_json(destination, merged.model_dump(mode="json"))
    print(json.dumps({
        "document_id": merged.document_id,
        "recovered_count": report["recovered_count"],
        "skipped_ambiguous_count": report["skipped_ambiguous_count"],
        "preview_path": str(preview_path),
        "applied": bool(args.apply),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
