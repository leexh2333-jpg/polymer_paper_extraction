"""Stage 6：离线校验 Stage 0-5 并生成 final.json。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from llm_client import DEFAULT_CONFIG_PATH, load_pipeline_config
from reports.render_extraction_html import render_extraction_html
from schema.polymer_schema import (
    CostSummary,
    Evidence,
    FinalCharacterization,
    FinalDocument,
    FinalEvidence,
    FinalMaterialMention,
    FinalMeasurementCondition,
    FinalPolymerEntity,
    FinalProcessStep,
    FinalPropertySeries,
    FinalPropertySeriesCoordinate,
    FinalPropertySeriesPoint,
    FinalPropertyObservation,
    FinalSample,
    FinalStage5PropertyObservation,
    FinalUnresolvedPropertyObservation,
    CompletenessMetric,
    QualityMetrics,
    MeasurementContext,
    Stage0Document,
    StageBilling,
    StageCost,
    Stage1Document,
    Stage2Document,
    Stage3Document,
    Stage4Document,
    Stage5Document,
    Stage6Validation,
    ValidationIssue,
    ValidationSummary,
)
from stages.table_grid import resolve_table_locator
from stages.stage4_property import (
    _element_source_text,
    _resolve_surface_text,
    write_json_atomic,
)


STAGE_ID = "stage6_validate_merge"
SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "upload_url",
    "download_url",
}
SECRET_VALUE_RE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{16,}|"
    r"\b(?:sk|ark)-[A-Za-z0-9_-]{20,})",
    flags=re.IGNORECASE,
)
OCR_PROVENANCE_FIELDS = (
    "engine",
    "batch_id",
    "model_version",
    "ocr_enabled",
    "language",
    "page_ranges",
    "enable_formula",
    "enable_table",
    "extra_formats",
    "status",
    "manifest_file",
    "status_file",
)
POLYMER_NAME_CONTAMINATION_RE = re.compile(
    r"(?:\b(?:dried|precipitated|crosslinked|vulcanized|film|sheet|"
    r"specimen|sample)\b|\b\d+(?:\.\d+)?\s*(?:phr|mm|cm|µm|μm)\b)",
    flags=re.IGNORECASE,
)


class Stage6Error(RuntimeError):
    """Stage 6 输入或写盘失败。"""


def _load_model(path: Path, model: type[BaseModel], label: str) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise Stage6Error(f"无法读取 {label}：{path}") from exc
    except ValidationError as exc:
        raise Stage6Error(f"{label} 未通过 Schema：{path.name}") from exc


def _add_issue(
    issues: list[ValidationIssue],
    *,
    stage: str,
    code: str,
    message: str,
    object_id: str | None = None,
) -> None:
    issue = ValidationIssue(
        stage=stage,
        code=code,
        message=message,
        object_id=object_id,
    )
    key = (issue.stage, issue.code, issue.message, issue.object_id)
    existing = {
        (item.stage, item.code, item.message, item.object_id)
        for item in issues
    }
    if key not in existing:
        issues.append(issue)


def _normalize_upstream_warning(
    warning: dict[str, Any],
    default_stage: str,
) -> ValidationIssue:
    stage = str(warning.get("stage") or default_stage).strip()
    code = str(warning.get("code") or "upstream_warning").strip()
    message = str(
        warning.get("message") or "上游阶段报告 warning"
    ).strip()
    object_id_value = warning.get("object_id")
    object_id = (
        str(object_id_value).strip()
        if object_id_value is not None
        else None
    )
    return ValidationIssue(
        stage=stage or default_stage,
        code=code or "upstream_warning",
        message=message or "上游阶段报告 warning",
        object_id=object_id or None,
    )


def _evidence_key(evidence: Evidence) -> str:
    return json.dumps(
        evidence.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_for_comparison(value: str) -> str:
    normalized = value
    replacements = {
        r"\$\pm\$": "±",
        r"\pm": "±",
        r"\mp": "∓",
        r"^{\circ}": "°",
        r"^\circ": "°",
        r"\mathrm{C}": "C",
        r"\delta": "δ",
        r"\chi": "χ",
        r"\eta": "η",
        "$": "",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _is_stable_blank_table_cell(block: Any, locator: dict[str, Any]) -> bool:
    stable_fields = (
        locator.get("cell_id"),
        locator.get("row_index"),
        locator.get("column_index"),
    )
    matching_cells = [
        cell
        for cell in (block.table_cells or [])
        if (
            getattr(cell, "cell_id", None),
            getattr(cell, "row_index", None),
            getattr(cell, "column_index", None),
        )
        == stable_fields
    ]
    return (
        len(matching_cells) == 1
        and not getattr(matching_cells[0], "text", "").strip()
    )


class EvidenceRegistry:
    def __init__(
        self,
        block_map: dict[str, Any],
        errors: list[ValidationIssue],
        warnings: list[ValidationIssue],
    ) -> None:
        self.block_map = block_map
        self.errors = errors
        self.warnings = warnings
        self._ids: dict[str, str] = {}
        self.items: list[FinalEvidence] = []
        self._validated: set[str] = set()
        self._stable_warnings: set[str] = set()

    def add(
        self,
        evidence: Evidence,
        *,
        stage: str,
        object_id: str,
    ) -> str:
        evidence = self._with_stable_table_locator(
            evidence,
            stage=stage,
            object_id=object_id,
        )
        key = _evidence_key(evidence)
        if key not in self._validated:
            self._validate(
                evidence,
                stage=stage,
                object_id=object_id,
            )
            self._validated.add(key)
        evidence_id = self._ids.get(key)
        if evidence_id is None:
            evidence_id = f"ev{len(self.items) + 1:03d}"
            self._ids[key] = evidence_id
            self.items.append(FinalEvidence(
                evidence_id=evidence_id,
                **evidence.model_dump(mode="python"),
            ))
        return evidence_id

    def _with_stable_table_locator(
        self,
        evidence: Evidence,
        *,
        stage: str,
        object_id: str,
    ) -> Evidence:
        block = self.block_map.get(evidence.block_id)
        if (
            block is None
            or block.type != "table"
            or evidence.table_locator is None
        ):
            return evidence
        stable = resolve_table_locator(block, evidence.table_locator)
        if stable is not None:
            return evidence.model_copy(update={"table_locator": stable})
        if (
            evidence.table_locator.get("cell_value") is None
            and _is_stable_blank_table_cell(block, evidence.table_locator)
        ):
            return evidence
        warning_key = _evidence_key(evidence)
        if block.table_cells and warning_key not in self._stable_warnings:
            self._stable_warnings.add(warning_key)
            _add_issue(
                self.warnings,
                stage=stage,
                code="table_locator_stable_cell_unresolved",
                message=(
                    "table_locator 无法唯一解析到稳定 cell_id；"
                    "保留原行列文字定位"
                ),
                object_id=object_id,
            )
        return evidence

    def add_many(
        self,
        evidence_items: Iterable[Evidence],
        *,
        stage: str,
        object_id: str,
    ) -> list[str]:
        return [
            self.add(item, stage=stage, object_id=object_id)
            for item in evidence_items
        ]

    def _validate(
        self,
        evidence: Evidence,
        *,
        stage: str,
        object_id: str,
    ) -> None:
        block = self.block_map.get(evidence.block_id)
        if block is None:
            _add_issue(
                self.errors,
                stage=stage,
                code="unknown_evidence_block",
                message=f"Evidence 引用了未知 block：{evidence.block_id}",
                object_id=object_id,
            )
            return
        if evidence.page != block.page:
            _add_issue(
                self.errors,
                stage=stage,
                code="evidence_page_mismatch",
                message="Evidence.page 与 Stage 0 block 不一致",
                object_id=object_id,
            )
        if evidence.bbox != block.bbox:
            _add_issue(
                self.errors,
                stage=stage,
                code="evidence_bbox_mismatch",
                message="Evidence.bbox 与 Stage 0 block 不一致",
                object_id=object_id,
            )
        if evidence.source_type != block.type:
            _add_issue(
                self.errors,
                stage=stage,
                code="evidence_type_mismatch",
                message="Evidence.source_type 与 Stage 0 block 不一致",
                object_id=object_id,
            )
        source = _element_source_text(block)
        if (
            evidence.source_sentence not in source
            and _normalize_for_comparison(evidence.source_sentence)
            not in _normalize_for_comparison(source)
        ):
            _add_issue(
                self.errors,
                stage=stage,
                code="evidence_not_in_source",
                message="Evidence.source_sentence 不是 Stage 0 原文子串",
                object_id=object_id,
            )
        locator = evidence.table_locator
        if locator is None:
            return
        if block.type != "table":
            _add_issue(
                self.errors,
                stage=stage,
                code="table_locator_on_non_table",
                message="非 table Evidence 不得包含 table_locator",
                object_id=object_id,
            )
            return
        required = ("table_id", "row_label", "column_label")
        if any(
            not isinstance(locator.get(field), str)
            or not locator[field].strip()
            for field in required
        ):
            _add_issue(
                self.errors,
                stage=stage,
                code="invalid_table_locator",
                message="table_locator 缺少行、列或单元格定位字段",
                object_id=object_id,
            )
            return
        cell_value = locator.get("cell_value")
        if cell_value is None:
            if not _is_stable_blank_table_cell(block, locator):
                _add_issue(
                    self.errors,
                    stage=stage,
                    code="invalid_table_locator",
                    message="空单元格 locator 缺少可解析的稳定定位字段",
                    object_id=object_id,
                )
                return
        elif not isinstance(cell_value, str) or not cell_value.strip():
            _add_issue(
                self.errors,
                stage=stage,
                code="invalid_table_locator",
                message="table_locator.cell_value 必须为非空字符串或 null",
                object_id=object_id,
            )
            return
        if locator["table_id"] != evidence.block_id:
            _add_issue(
                self.errors,
                stage=stage,
                code="table_id_mismatch",
                message="table_locator.table_id 必须等于 block_id",
                object_id=object_id,
            )
        for field in ("row_label", "column_label"):
            if _resolve_surface_text(source, locator[field]) is None:
                _add_issue(
                    self.errors,
                    stage=stage,
                    code="table_locator_not_in_source",
                    message=f"table_locator.{field} 不是表格原文",
                    object_id=object_id,
                )
        if isinstance(cell_value, str) and (
            _resolve_surface_text(source, cell_value) is None
        ):
            _add_issue(
                self.errors,
                stage=stage,
                code="table_locator_not_in_source",
                message="table_locator.cell_value 不是表格原文",
                object_id=object_id,
            )


def _without_evidence(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="python", exclude={"evidence"})


def _finalize_condition_quantity(
    quantity: Any,
    registry: EvidenceRegistry,
    *,
    stage: str,
    object_id: str,
) -> Any:
    if quantity is None:
        return None
    evidence_ids = list(dict.fromkeys([
        *quantity.evidence_ids,
        *registry.add_many(
            quantity.evidence,
            stage=stage,
            object_id=object_id,
        ),
    ]))
    return quantity.model_copy(update={
        "evidence": [],
        "evidence_ids": evidence_ids,
    })


def _measurement_context(
    condition: Any,
    registry: EvidenceRegistry,
    *,
    stage: str,
    object_id: str,
) -> MeasurementContext:
    existing_other_ids = getattr(
        condition, "other_condition_evidence_ids", {}
    )
    other_condition_evidence_ids = {
        key: list(dict.fromkeys([
            *existing_other_ids.get(key, []),
            *registry.add_many(
                evidence,
                stage=stage,
                object_id=object_id,
            ),
        ]))
        for key, evidence in condition.other_condition_evidence.items()
    }
    for key, evidence_ids in existing_other_ids.items():
        other_condition_evidence_ids.setdefault(key, list(evidence_ids))
    return MeasurementContext(
        temperature=_finalize_condition_quantity(
            condition.temperature, registry, stage=stage, object_id=object_id
        ),
        frequency=_finalize_condition_quantity(
            condition.frequency, registry, stage=stage, object_id=object_id
        ),
        humidity=_finalize_condition_quantity(
            condition.humidity, registry, stage=stage, object_id=object_id
        ),
        pressure=_finalize_condition_quantity(
            condition.pressure, registry, stage=stage, object_id=object_id
        ),
        wavelength=_finalize_condition_quantity(
            condition.wavelength, registry, stage=stage, object_id=object_id
        ),
        other_conditions=condition.other_conditions,
        other_condition_evidence={},
        other_condition_evidence_ids=other_condition_evidence_ids,
        condition_status=condition.condition_status,
    )


def _scan_sensitive(
    value: Any,
    errors: list[ValidationIssue],
    path: str = "",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).casefold() in SENSITIVE_KEYS:
                _add_issue(
                    errors,
                    stage=STAGE_ID,
                    code="sensitive_field",
                    message=f"输出包含禁止的敏感字段：{child_path}",
                )
            _scan_sensitive(child, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_sensitive(child, errors, f"{path}[{index}]")
    elif isinstance(value, str) and SECRET_VALUE_RE.search(value):
        _add_issue(
            errors,
            stage=STAGE_ID,
            code="sensitive_value",
            message=f"输出疑似包含凭据值：{path}",
        )


def _ocr_provenance(document: Stage0Document) -> dict[str, Any]:
    provenance: dict[str, Any] = {"stage": "ocr"}
    for field in OCR_PROVENANCE_FIELDS:
        if field in document.ocr:
            provenance[field] = document.ocr[field]
    return provenance


def _not_applicable_billing(stage: str) -> StageBilling:
    return StageBilling(
        stage=stage,
        call_count=0,
        cost=StageCost(
            status="not_applicable",
            currency="CNY",
            input_cost=Decimal(0),
            output_cost=Decimal(0),
            total_cost=Decimal(0),
        ),
    )


def _billing_from_provenance(
    stage: str,
    provenance: dict[str, Any],
) -> StageBilling:
    raw_cost = provenance.get("cost")
    cost = (
        StageCost.model_validate(raw_cost)
        if isinstance(raw_cost, dict)
        else StageCost(
            status="unavailable",
            currency="CNY",
        )
    )
    raw_usage = provenance.get("usage")
    call_count = provenance.get("call_count")
    if call_count is None and stage == "stage1_material_mention":
        call_count = provenance.get("chunk_count")
    return StageBilling(
        stage=stage,
        provider=provenance.get("provider"),
        model=provenance.get("model"),
        call_count=call_count,
        usage=raw_usage if isinstance(raw_usage, dict) else None,
        cost=cost,
    )


def _build_cost_summary(
    stage0: Stage0Document,
    stage1: Stage1Document,
    stage2: Stage2Document,
    stage3: Stage3Document,
    stage4: Stage4Document,
    stage5: Stage5Document,
) -> CostSummary:
    metadata = stage0.paper.metadata_extraction
    stages = [
        _not_applicable_billing("stage_minus1_reorganize_mineru"),
        _billing_from_provenance("meta_extract", metadata),
        _not_applicable_billing("stage0_load_document"),
        _billing_from_provenance(
            "stage1_material_mention",
            stage1.provenance.model_dump(mode="python"),
        ),
        _billing_from_provenance(
            "stage2_polymer_entity",
            stage2.provenance.model_dump(mode="python"),
        ),
        _billing_from_provenance(
            "stage3_sample_process",
            stage3.provenance.model_dump(mode="python"),
        ),
        _billing_from_provenance(
            "stage4_property",
            stage4.provenance.model_dump(mode="python"),
        ),
        _billing_from_provenance(
            "stage5_characterization",
            stage5.provenance.model_dump(mode="python"),
        ),
        _not_applicable_billing("stage6_validate_merge"),
        _not_applicable_billing("report_html"),
    ]
    total_cost = sum(
        (
            item.cost.total_cost
            for item in stages
            if item.cost.total_cost is not None
        ),
        start=Decimal(0),
    )
    return CostSummary(
        status=(
            "partial"
            if any(
                item.cost.status == "unavailable"
                for item in stages
            )
            else "complete"
        ),
        stages=stages,
        total_cost=total_cost,
    )


def _metric(complete: int, total: int) -> CompletenessMetric:
    return CompletenessMetric(
        complete=complete,
        total=total,
        ratio=complete / total if total else 1.0,
    )


def _build_quality_metrics(
    stage1: Stage1Document,
    stage2: Stage2Document,
    stage3: Stage3Document,
    stage4: Stage4Document,
    stage5: Stage5Document,
) -> QualityMetrics:
    series_points = [
        point
        for series in stage4.property_series
        for point in series.points
    ]
    covered_series_points = [
        point
        for point in series_points
        if point.coverage_status == "covered"
    ]
    all_properties = [
        *stage4.properties,
        *stage4.unresolved_properties,
        *covered_series_points,
        *stage5.properties,
    ]
    method_properties = [
        item
        for item in [
            *stage4.properties,
            *stage4.unresolved_properties,
        ]
        if item.determination_method_raw is not None
    ]
    sample_entities = {
        item.sample_id: item.refers_to_entity
        for item in stage3.samples
    }
    linked_method_properties = [
        item
        for item in method_properties
        if any(
            (
                characterization.sample_id == item.sample_id
                if hasattr(item, "property_id")
                else (
                    characterization.entity_id
                    or (
                        sample_entities.get(characterization.sample_id)
                        if characterization.sample_id is not None
                        else None
                    )
                )
                == item.entity_id
            )
            and (
                item.property_id
                if hasattr(item, "property_id")
                else item.unresolved_id
            )
            in characterization.derived_property_ids
            for characterization in stage5.characterizations
        )
    ]
    confidence_objects = [
        *stage1.material_mentions,
        *stage2.polymer_entities,
        *stage3.samples,
        *stage3.process_steps,
        *stage4.measurement_conditions,
        *stage4.properties,
        *stage4.unresolved_properties,
        *stage4.property_series,
        *series_points,
        *stage5.characterizations,
        *stage5.properties,
    ]
    return QualityMetrics(
        properties_with_units=_metric(
            sum(item.unit_raw is not None for item in all_properties),
            len(all_properties),
        ),
        standard_process_steps=_metric(
            sum(item.process_type != "other" for item in stage3.process_steps),
            len(stage3.process_steps),
        ),
        stage4_methods_with_characterization=_metric(
            len(linked_method_properties),
            len(method_properties),
        ),
        objects_with_confidence=_metric(
            sum(item.confidence is not None for item in confidence_objects),
            len(confidence_objects),
        ),
        series_points_covered=_metric(
            sum(
                point.coverage_status == "covered"
                for point in series_points
            ),
            sum(
                point.coverage_status in {"covered", "missing"}
                for point in series_points
            ),
        ),
    )


def _validate_document_references(
    stage1: Stage1Document,
    stage2: Stage2Document,
    stage3: Stage3Document,
    stage4: Stage4Document,
    stage5: Stage5Document,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    mention_ids = {
        item.mention_id for item in stage1.material_mentions
    }
    resolved_mentions = {
        mention_id
        for entity in stage2.polymer_entities
        for mention_id in entity.resolved_from_mentions
    }
    unresolved_mentions = set(stage2.unresolved_mention_ids)
    unknown_mentions = (
        resolved_mentions | unresolved_mentions
    ) - mention_ids
    if unknown_mentions:
        _add_issue(
            errors,
            stage="stage2_polymer_entity",
            code="unknown_mention_reference",
            message=f"Stage 2 引用了未知 mention：{sorted(unknown_mentions)}",
        )
    missing_mentions = mention_ids - resolved_mentions - unresolved_mentions
    if missing_mentions:
        _add_issue(
            errors,
            stage="stage2_polymer_entity",
            code="missing_mention_resolution",
            message=f"Stage 1 mention 未被解析或标记 unresolved：{sorted(missing_mentions)}",
        )

    entity_ids = {item.entity_id for item in stage2.polymer_entities}
    resolved_entities = {
        sample.refers_to_entity
        for sample in stage3.samples
        if sample.refers_to_entity is not None
    }
    unresolved_entities = set(stage3.unresolved_entity_ids)
    unknown_entities = (
        resolved_entities | unresolved_entities
    ) - entity_ids
    if unknown_entities:
        _add_issue(
            errors,
            stage="stage3_sample_process",
            code="unknown_entity_reference",
            message=f"Stage 3 引用了未知 entity：{sorted(unknown_entities)}",
        )
    missing_entities = entity_ids - resolved_entities - unresolved_entities
    if missing_entities:
        _add_issue(
            warnings,
            stage="stage3_sample_process",
            code="entity_without_sample_resolution",
            message=f"Entity 未关联 Sample 且未标记 unresolved：{sorted(missing_entities)}",
        )

    sample_ids = {item.sample_id for item in stage3.samples}
    condition_ids = {
        item.condition_id for item in stage4.measurement_conditions
    }
    referenced_conditions = {
        item.measurement_condition_id for item in stage4.properties
    }
    for item in stage4.properties:
        if item.sample_id not in sample_ids:
            _add_issue(
                errors,
                stage="stage4_property",
                code="unknown_sample_reference",
                message=f"Property 引用了未知 sample：{item.sample_id}",
                object_id=item.property_id,
            )
        if item.measurement_condition_id not in condition_ids:
            _add_issue(
                errors,
                stage="stage4_property",
                code="unknown_condition_reference",
                message=(
                    "Property 引用了未知 measurement condition："
                    f"{item.measurement_condition_id}"
                ),
                object_id=item.property_id,
            )
    unused_conditions = condition_ids - referenced_conditions
    if unused_conditions:
        _add_issue(
            warnings,
            stage="stage4_property",
            code="unused_measurement_condition",
            message=f"MeasurementCondition 未被性质引用：{sorted(unused_conditions)}",
        )
    for item in stage4.unresolved_properties:
        if item.entity_id not in entity_ids:
            _add_issue(
                errors,
                stage="stage4_property",
                code="unknown_entity_reference",
                message=f"Unresolved property 引用了未知 entity：{item.entity_id}",
                object_id=item.unresolved_id,
            )

    series_ids = {item.series_id for item in stage4.property_series}
    for item in [*stage4.properties, *stage4.unresolved_properties]:
        references = set(item.series_ids or [])
        if item.series_id is not None:
            references.add(item.series_id)
        unknown_series = sorted(references - series_ids)
        if unknown_series:
            _add_issue(
                errors,
                stage="stage4_property",
                code="unknown_series_reference",
                message=f"Aggregate 引用了未知 series：{unknown_series}",
                object_id=(
                    item.property_id
                    if hasattr(item, "property_id")
                    else item.unresolved_id
                ),
            )
    for item in stage4.property_series:
        if item.sample_id and item.sample_id not in sample_ids:
            _add_issue(
                errors,
                stage="stage4_property",
                code="unknown_sample_reference",
                message=f"PropertySeries 引用了未知 sample：{item.sample_id}",
                object_id=item.series_id,
            )
        if item.entity_id and item.entity_id not in entity_ids:
            _add_issue(
                errors,
                stage="stage4_property",
                code="unknown_entity_reference",
                message=f"PropertySeries 引用了未知 entity：{item.entity_id}",
                object_id=item.series_id,
            )
        for point in item.points:
            if point.sample_id and point.sample_id not in sample_ids:
                _add_issue(
                    errors,
                    stage="stage4_property",
                    code="unknown_sample_reference",
                    message=(
                        "PropertySeries point 引用了未知 sample："
                        f"{point.sample_id}"
                    ),
                    object_id=point.point_id,
                )
            if point.entity_id and point.entity_id not in entity_ids:
                _add_issue(
                    errors,
                    stage="stage4_property",
                    code="unknown_entity_reference",
                    message=(
                        "PropertySeries point 引用了未知 entity："
                        f"{point.entity_id}"
                    ),
                    object_id=point.point_id,
                )

    stage4_property_ids = {item.property_id for item in stage4.properties}
    unresolved_property_ids = {
        item.unresolved_id for item in stage4.unresolved_properties
    }
    stage5_property_ids = {item.property_id for item in stage5.properties}
    characterization_ids = {
        item.characterization_id for item in stage5.characterizations
    }
    all_property_ids = (
        stage4_property_ids
        | unresolved_property_ids
        | stage5_property_ids
    )
    for item in stage5.characterizations:
        characterization_series = set(item.series_ids or [])
        if item.series_id is not None:
            characterization_series.add(item.series_id)
        unknown_series = sorted(characterization_series - series_ids)
        if unknown_series:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_series_reference",
                message=(
                    "Characterization 引用了未知 series："
                    f"{unknown_series}"
                ),
                object_id=item.characterization_id,
            )
        unknown_samples = sorted(set(item.sample_ids or []) - sample_ids)
        if unknown_samples:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_sample_reference",
                message=f"Characterization 引用了未知 samples：{unknown_samples}",
                object_id=item.characterization_id,
            )
        unknown_entities = sorted(set(item.entity_ids or []) - entity_ids)
        if unknown_entities:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_entity_reference",
                message=f"Characterization 引用了未知 entities：{unknown_entities}",
                object_id=item.characterization_id,
            )
        if item.sample_id and item.sample_id not in sample_ids:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_sample_reference",
                message=f"Characterization 引用了未知 sample：{item.sample_id}",
                object_id=item.characterization_id,
            )
        if item.entity_id and item.entity_id not in entity_ids:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_entity_reference",
                message=f"Characterization 引用了未知 entity：{item.entity_id}",
                object_id=item.characterization_id,
            )
        unknown_properties = (
            set(item.derived_property_ids) - all_property_ids
        )
        if unknown_properties:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_property_reference",
                message=(
                    "Characterization 引用了未知 property："
                    f"{sorted(unknown_properties)}"
                ),
                object_id=item.characterization_id,
            )
    for item in stage5.properties:
        unknown_samples = sorted(set(item.sample_ids or []) - sample_ids)
        if unknown_samples:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_sample_reference",
                message=f"Stage 5 property 引用了未知 samples：{unknown_samples}",
                object_id=item.property_id,
            )
        unknown_entities = sorted(set(item.entity_ids or []) - entity_ids)
        if unknown_entities:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_entity_reference",
                message=f"Stage 5 property 引用了未知 entities：{unknown_entities}",
                object_id=item.property_id,
            )
        if item.characterization_id not in characterization_ids:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_characterization_reference",
                message=(
                    "Stage 5 property 引用了未知 Characterization："
                    f"{item.characterization_id}"
                ),
                object_id=item.property_id,
            )
        if item.sample_id and item.sample_id not in sample_ids:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_sample_reference",
                message=f"Stage 5 property 引用了未知 sample：{item.sample_id}",
                object_id=item.property_id,
            )
        if item.entity_id and item.entity_id not in entity_ids:
            _add_issue(
                errors,
                stage="stage5_characterization",
                code="unknown_entity_reference",
                message=f"Stage 5 property 引用了未知 entity：{item.entity_id}",
                object_id=item.property_id,
            )

    produced_samples = {
        sample_id
        for step in stage3.process_steps
        for sample_id in step.output_sample_ids
    }
    referenced_samples = {
        sample_id
        for step in stage3.process_steps
        for sample_id in (*step.input_sample_ids, *step.output_sample_ids)
    } | {
        item.sample_id for item in stage4.properties
    } | {
        item.sample_id
        for item in stage5.characterizations
        if item.sample_id is not None
    } | {
        item.sample_id
        for item in stage5.properties
        if item.sample_id is not None
    }
    for sample in stage3.samples:
        if (
            sample.refers_to_entity is None
            and sample.sample_id not in produced_samples
        ):
            _add_issue(
                warnings,
                stage="stage3_sample_process",
                code="sample_without_source",
                message="Sample 无 entity 来源且不是 ProcessStep 输出",
                object_id=sample.sample_id,
            )
        if sample.sample_id not in referenced_samples:
            _add_issue(
                warnings,
                stage="stage3_sample_process",
                code="orphan_sample",
                message="Sample 未被工艺、性质或表征引用",
                object_id=sample.sample_id,
            )


def _add_quality_warnings(
    stage0: Stage0Document,
    stage3: Stage3Document,
    stage4: Stage4Document,
    stage5: Stage5Document,
    warnings: list[ValidationIssue],
) -> None:
    block_map = {item.block_id: item for item in stage0.elements}
    for step in stage3.process_steps:
        if step.process_type == "other":
            _add_issue(
                warnings,
                stage="stage3_sample_process",
                code="generic_process_type",
                message="ProcessStep 使用了通用 process_type=other",
                object_id=step.step_id,
            )
        block = block_map.get(step.evidence.block_id)
        if block is None:
            continue
        source = _normalize_for_comparison(_element_source_text(block))
        unsupported_keys = [
            key
            for key, value in step.parameters.items()
            if _normalize_for_comparison(value) not in source
        ]
        if unsupported_keys:
            _add_issue(
                warnings,
                stage="stage3_sample_process",
                code="unsupported_specific_value",
                message=(
                    "ProcessStep 参数无法在 evidence 原文定位："
                    f"{sorted(unsupported_keys)}"
                ),
                object_id=step.step_id,
            )

    for sample in stage3.samples:
        if POLYMER_NAME_CONTAMINATION_RE.search(sample.polymer_name):
            _add_issue(
                warnings,
                stage="stage3_sample_process",
                code="polymer_name_contamination",
                message="polymer_name 疑似包含样品状态、形态或配方数值",
                object_id=sample.sample_id,
            )

    for property_item in stage4.properties:
        if property_item.determination_method_raw is None:
            continue
        if not any(
            characterization.sample_id == property_item.sample_id
            and property_item.property_id
            in characterization.derived_property_ids
            for characterization in stage5.characterizations
        ):
            _add_issue(
                warnings,
                stage="stage5_characterization",
                code="missing_characterization",
                message=(
                    "Stage 4 property 已报告测定方法，但缺少同一样品的 "
                    "Characterization 回链"
                ),
                object_id=property_item.property_id,
            )
    sample_entities = {
        item.sample_id: item.refers_to_entity
        for item in stage3.samples
    }
    for property_item in stage4.unresolved_properties:
        if property_item.determination_method_raw is None:
            continue
        if not any(
            (
                characterization.entity_id
                or (
                    sample_entities.get(characterization.sample_id)
                    if characterization.sample_id is not None
                    else None
                )
            )
            == property_item.entity_id
            and property_item.unresolved_id
            in characterization.derived_property_ids
            for characterization in stage5.characterizations
        ):
            _add_issue(
                warnings,
                stage="stage5_characterization",
                code="missing_characterization",
                message=(
                    "Unresolved Stage 4 property 已报告测定方法，"
                    "但缺少同一实体的 Characterization 回链"
                ),
                object_id=property_item.unresolved_id,
            )


def validate_and_merge(
    stage0: Stage0Document,
    stage1: Stage1Document,
    stage2: Stage2Document,
    stage3: Stage3Document,
    stage4: Stage4Document,
    stage5: Stage5Document,
) -> tuple[FinalDocument | None, Stage6Validation]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    document_ids = {
        stage0.document_id,
        stage1.document_id,
        stage2.document_id,
        stage3.document_id,
        stage4.document_id,
        stage5.document_id,
    }
    if len(document_ids) != 1:
        _add_issue(
            errors,
            stage=STAGE_ID,
            code="document_id_mismatch",
            message=f"Stage 0-5 document_id 不一致：{sorted(document_ids)}",
        )

    for warning, default_stage in (
        *((item, "stage0") for item in stage0.warnings),
        *((item, "stage1_material_mention") for item in stage1.warnings),
        *((item, "stage2_polymer_entity") for item in stage2.warnings),
        *((item, "stage3_sample_process") for item in stage3.warnings),
        *((item, "stage4_property") for item in stage4.warnings),
        *((item, "stage5_characterization") for item in stage5.warnings),
    ):
        normalized = _normalize_upstream_warning(warning, default_stage)
        _add_issue(
            warnings,
            stage=normalized.stage,
            code=normalized.code,
            message=normalized.message,
            object_id=normalized.object_id,
        )
    if stage0.paper.metadata_status != "complete":
        _add_issue(
            warnings,
            stage="meta_extract",
            code="paper_metadata_incomplete",
            message=(
                "Paper metadata_status="
                f"{stage0.paper.metadata_status}，需人工复核"
            ),
        )
    if stage0.ocr.get("status") != "done":
        _add_issue(
            errors,
            stage="ocr",
            code="ocr_not_done",
            message=f"OCR status 不是 done：{stage0.ocr.get('status')!r}",
        )

    _validate_document_references(
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        errors,
        warnings,
    )
    _add_quality_warnings(
        stage0,
        stage3,
        stage4,
        stage5,
        warnings,
    )
    quality_metrics = _build_quality_metrics(
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
    )

    block_map = {item.block_id: item for item in stage0.elements}
    registry = EvidenceRegistry(block_map, errors, warnings)
    final_mentions = [
        FinalMaterialMention(
            **_without_evidence(item),
            evidence_ids=[
                registry.add(
                    item.evidence,
                    stage="stage1_material_mention",
                    object_id=item.mention_id,
                )
            ],
        )
        for item in stage1.material_mentions
    ]
    final_entities = [
        FinalPolymerEntity(
            **_without_evidence(item),
            evidence_ids=[
                registry.add(
                    item.evidence,
                    stage="stage2_polymer_entity",
                    object_id=item.entity_id,
                )
            ],
        )
        for item in stage2.polymer_entities
    ]
    final_samples = [
        FinalSample(
            **_without_evidence(item),
            evidence_ids=[
                registry.add(
                    item.evidence,
                    stage="stage3_sample_process",
                    object_id=item.sample_id,
                )
            ],
        )
        for item in stage3.samples
    ]
    final_steps = [
        FinalProcessStep(
            **_without_evidence(item),
            evidence_ids=[
                registry.add(
                    item.evidence,
                    stage="stage3_sample_process",
                    object_id=item.step_id,
                )
            ],
        )
        for item in stage3.process_steps
    ]
    final_conditions = []
    condition_contexts = {}
    condition_fields = {
        "temperature",
        "frequency",
        "humidity",
        "pressure",
        "wavelength",
        "other_condition_evidence",
    }
    for item in stage4.measurement_conditions:
        context = _measurement_context(
            item,
            registry,
            stage="stage4_property",
            object_id=item.condition_id,
        )
        payload = item.model_dump(
            mode="python",
            exclude={"evidence", *condition_fields},
        )
        final_conditions.append(FinalMeasurementCondition(
            **payload,
            temperature=context.temperature,
            frequency=context.frequency,
            humidity=context.humidity,
            pressure=context.pressure,
            wavelength=context.wavelength,
            other_condition_evidence_ids=(
                context.other_condition_evidence_ids
            ),
            evidence_ids=[
                registry.add(
                    item.evidence,
                    stage="stage4_property",
                    object_id=item.condition_id,
                )
            ],
        ))
        condition_contexts[item.condition_id] = context
    final_stage4_properties = [
        FinalPropertyObservation(
            **{
                **_without_evidence(item),
                "measurement_context": _measurement_context(
                    item.measurement_context
                    or condition_contexts.get(item.measurement_condition_id),
                    registry,
                    stage="stage4_property",
                    object_id=item.property_id,
                ),
            },
            evidence_ids=registry.add_many(
                item.evidence,
                stage="stage4_property",
                object_id=item.property_id,
            ),
        )
        for item in stage4.properties
    ]
    final_unresolved = [
        FinalUnresolvedPropertyObservation(
            **{
                **_without_evidence(item),
                "measurement_context": (
                    _measurement_context(
                        item.measurement_context,
                        registry,
                        stage="stage4_property",
                        object_id=item.unresolved_id,
                    )
                    if item.measurement_context is not None
                    else None
                ),
            },
            evidence_ids=registry.add_many(
                item.evidence,
                stage="stage4_property",
                object_id=item.unresolved_id,
            ),
        )
        for item in stage4.unresolved_properties
    ]
    final_series = []
    for item in stage4.property_series:
        points = [
            FinalPropertySeriesPoint(
                **{
                    **point.model_dump(
                        mode="python",
                        exclude={"evidence", "coordinates"},
                    ),
                    "measurement_context": _measurement_context(
                        point.measurement_context,
                        registry,
                        stage="stage4_property",
                        object_id=point.point_id,
                    ),
                },
                coordinates=[
                    FinalPropertySeriesCoordinate(
                        **_without_evidence(coordinate),
                        evidence_ids=[registry.add(
                            coordinate.evidence,
                            stage="stage4_property",
                            object_id=point.point_id,
                        )],
                    )
                    for coordinate in point.coordinates
                ],
                evidence_ids=registry.add_many(
                    point.evidence,
                    stage="stage4_property",
                    object_id=point.point_id,
                ),
            )
            for point in item.points
        ]
        series_payload = item.model_dump(
            mode="python",
            exclude={"evidence", "points"},
        )
        series_payload["measurement_context"] = _measurement_context(
            item.measurement_context,
            registry,
            stage="stage4_property",
            object_id=item.series_id,
        )
        final_series.append(FinalPropertySeries(
            **series_payload,
            points=points,
            evidence_ids=registry.add_many(
                item.evidence,
                stage="stage4_property",
                object_id=item.series_id,
            ),
        ))
    final_characterizations = [
        FinalCharacterization(
            **{
                **_without_evidence(item),
                "measurement_context": (
                    _measurement_context(
                        item.measurement_context,
                        registry,
                        stage="stage5_characterization",
                        object_id=item.characterization_id,
                    )
                    if item.measurement_context is not None
                    else None
                ),
            },
            evidence_ids=registry.add_many(
                item.evidence,
                stage="stage5_characterization",
                object_id=item.characterization_id,
            ),
        )
        for item in stage5.characterizations
    ]
    final_stage5_properties = [
        FinalStage5PropertyObservation(
            **{
                **_without_evidence(item),
                "measurement_context": (
                    _measurement_context(
                        item.measurement_context,
                        registry,
                        stage="stage5_characterization",
                        object_id=item.property_id,
                    )
                    if item.measurement_context is not None
                    else None
                ),
            },
            evidence_ids=registry.add_many(
                item.evidence,
                stage="stage5_characterization",
                object_id=item.property_id,
            ),
        )
        for item in stage5.properties
    ]

    provenance = [
        _ocr_provenance(stage0),
        stage1.provenance.model_dump(mode="json"),
        stage2.provenance.model_dump(mode="json"),
        stage3.provenance.model_dump(mode="json"),
        stage4.provenance.model_dump(mode="json"),
        stage5.provenance.model_dump(mode="json"),
    ]
    checked_counts = {
        "stage0_blocks": len(stage0.elements),
        "material_mentions": len(stage1.material_mentions),
        "polymer_entities": len(stage2.polymer_entities),
        "samples": len(stage3.samples),
        "process_steps": len(stage3.process_steps),
        "measurement_conditions": len(stage4.measurement_conditions),
        "stage4_properties": len(stage4.properties),
        "unresolved_properties": len(stage4.unresolved_properties),
        "property_series": len(stage4.property_series),
        "property_series_points": sum(
            len(item.points) for item in stage4.property_series
        ),
        "characterizations": len(stage5.characterizations),
        "stage5_properties": len(stage5.properties),
        "evidence": len(registry.items),
    }
    status = (
        "failed"
        if errors
        else "passed_with_warnings"
        if warnings
        else "passed"
    )
    validation = Stage6Validation(
        document_id=stage0.document_id,
        status=status,
        error_count=len(errors),
        warning_count=len(warnings),
        errors=errors,
        warnings=warnings,
        checked_counts=checked_counts,
        quality_metrics=quality_metrics,
    )
    if errors:
        return None, validation

    summary = ValidationSummary(
        status=status,
        error_count=0,
        warning_count=len(warnings),
    )
    final = FinalDocument(
        document_id=stage0.document_id,
        paper=stage0.paper,
        material_mentions=final_mentions,
        polymer_entities=final_entities,
        unresolved_mention_ids=stage2.unresolved_mention_ids,
        samples=final_samples,
        process_steps=final_steps,
        unresolved_entity_ids=stage3.unresolved_entity_ids,
        property_observations=[
            *final_stage4_properties,
            *final_stage5_properties,
        ],
        measurement_conditions=final_conditions,
        unresolved_property_observations=final_unresolved,
        property_series=final_series,
        characterizations=final_characterizations,
        evidence=registry.items,
        provenance=provenance,
        warnings=warnings,
        validation_summary=summary,
        cost_summary=_build_cost_summary(
            stage0,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
        ),
        quality_metrics=quality_metrics,
    )
    raw_final = final.model_dump(mode="json")
    _scan_sensitive(raw_final, errors)
    if errors:
        validation = Stage6Validation(
            document_id=stage0.document_id,
            status="failed",
            error_count=len(errors),
            warning_count=len(warnings),
            errors=errors,
            warnings=warnings,
            checked_counts=checked_counts,
            quality_metrics=quality_metrics,
        )
        return None, validation
    return final, validation


def _failed_input_validation(
    document_id: str,
    message: str,
) -> Stage6Validation:
    issue = ValidationIssue(
        stage=STAGE_ID,
        code="invalid_stage_input",
        message=message,
    )
    return Stage6Validation(
        document_id=document_id,
        status="failed",
        error_count=1,
        warning_count=0,
        errors=[issue],
        warnings=[],
        checked_counts={},
    )


def run_stage6(
    document_id: str,
    stage0_path: Path,
    stage1_path: Path,
    stage2_path: Path,
    stage3_path: Path,
    stage4_path: Path,
    stage5_path: Path,
    validation_path: Path,
    final_path: Path,
) -> tuple[Stage6Validation, bool]:
    try:
        stage0 = _load_model(stage0_path, Stage0Document, "Stage 0")
        stage1 = _load_model(stage1_path, Stage1Document, "Stage 1")
        stage2 = _load_model(stage2_path, Stage2Document, "Stage 2")
        stage3 = _load_model(stage3_path, Stage3Document, "Stage 3")
        stage4 = _load_model(stage4_path, Stage4Document, "Stage 4")
        stage5 = _load_model(stage5_path, Stage5Document, "Stage 5")
        final, validation = validate_and_merge(
            stage0,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
        )
    except Stage6Error as exc:
        final = None
        validation = _failed_input_validation(document_id, str(exc))

    write_json_atomic(
        validation_path,
        validation.model_dump(mode="json", exclude_none=True),
    )
    if final is None:
        if final_path.is_file():
            final_path.unlink()
        report_path = final_path.with_name("report.html")
        if report_path.is_file():
            report_path.unlink()
        return validation, False
    payload = final.model_dump(mode="json", exclude_none=True)
    payload["paper"] = final.paper.model_dump(
        mode="json",
        exclude_none=False,
    )
    write_json_atomic(final_path, payload)
    render_extraction_html(
        payload,
        final_path.with_name("report.html"),
        stage0_data=stage0.model_dump(mode="json"),
    )
    return validation, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 Stage 6 校验与合并")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ref-no")
    mode.add_argument("--batch", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    paths = config.get("paths") or {}
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else Path(paths.get("output_dir") or EXTRACTION_ROOT / "output").resolve()
    )
    input_root = (
        args.input_root.expanduser().resolve()
        if args.input_root
        else output_root
    )
    if args.ref_no:
        ref_nos = [args.ref_no]
    else:
        ref_nos = sorted(
            path.parent.name
            for path in input_root.glob(
                "reference_no_*/stage5_characterizations.json"
            )
        )
    if not ref_nos:
        raise Stage6Error(f"未找到 Stage 5 输出：{input_root}")

    failures = 0
    for ref_no in ref_nos:
        base = input_root / ref_no
        output = output_root / ref_no
        validation, published = run_stage6(
            ref_no,
            base / "stage0_blocks.json",
            base / "stage1_mentions.json",
            base / "stage2_entities.json",
            base / "stage3_process.json",
            base / "stage4_properties.json",
            base / "stage5_characterizations.json",
            output / "stage6_validation.json",
            output / "final.json",
        )
        if published:
            print(
                f"[done] {ref_no}: errors={validation.error_count}, "
                f"warnings={validation.warning_count}"
            )
        else:
            failures += 1
            print(
                f"[failed] {ref_no}: errors={validation.error_count}, "
                f"warnings={validation.warning_count}",
                file=sys.stderr,
            )
    print(f"Stage 6 完成：成功 {len(ref_nos) - failures}，失败 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
