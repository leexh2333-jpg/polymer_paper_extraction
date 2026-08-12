"""Preview-only Stage 6 逐对象隔离与引用清扫。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel

from schema.polymer_schema import (
    PreviewPublicationSummary,
    RejectedObject,
    SeriesCoverage,
    ValidationIssue,
)


@dataclass
class PreviewCollections:
    material_mentions: list[Any]
    polymer_entities: list[Any]
    unresolved_mention_ids: list[str]
    samples: list[Any]
    process_steps: list[Any]
    unresolved_entity_ids: list[str]
    measurement_conditions: list[Any]
    property_observations: list[Any]
    unresolved_property_observations: list[Any]
    property_series: list[Any]
    characterizations: list[Any]
    evidence: list[Any]


@dataclass
class PreviewSalvageResult:
    collections: PreviewCollections
    remaining_errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    rejected_objects: list[RejectedObject]
    summary: PreviewPublicationSummary


_COLLECTION_SPECS = (
    ("material_mentions", "material_mention", "stage1_material_mention", "mention_id"),
    ("polymer_entities", "polymer_entity", "stage2_polymer_entity", "entity_id"),
    ("samples", "sample", "stage3_sample_process", "sample_id"),
    ("process_steps", "process_step", "stage3_sample_process", "step_id"),
    (
        "measurement_conditions",
        "measurement_condition",
        "stage4_property",
        "condition_id",
    ),
    ("property_observations", "property", "stage4_or_stage5", "property_id"),
    (
        "unresolved_property_observations",
        "unresolved_property",
        "stage4_property",
        "unresolved_id",
    ),
    ("property_series", "property_series", "stage4_property", "series_id"),
    (
        "characterizations",
        "characterization",
        "stage5_characterization",
        "characterization_id",
    ),
)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _replace(value: Any, **updates: Any) -> Any:
    payload = value.model_dump(mode="python")
    payload.update(updates)
    return type(value).model_validate(payload)


def _replace_series_points(series: Any, points: list[Any]) -> Any:
    counts = Counter(point.coverage_status for point in points)
    expected = counts["covered"] + counts["missing"]
    coverage = SeriesCoverage(
        expected=expected,
        covered=counts["covered"],
        missing=counts["missing"],
        not_applicable=counts["not_applicable"],
        ratio=counts["covered"] / expected if expected else 1.0,
    )
    return _replace(series, points=points, coverage=coverage)


def _object_index(
    collections: PreviewCollections,
) -> dict[str, tuple[str, str, str, Any]]:
    result: dict[str, tuple[str, str, str, Any]] = {}
    for collection_name, object_type, source_stage, id_field in _COLLECTION_SPECS:
        for item in getattr(collections, collection_name):
            result[str(getattr(item, id_field))] = (
                collection_name,
                object_type,
                source_stage,
                item,
            )
            if collection_name == "property_series":
                for point in item.points:
                    result[str(point.point_id)] = (
                        collection_name,
                        "property_series_point",
                        "stage4_property",
                        point,
                    )
    return result


def _counts(collections: PreviewCollections) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for collection_name, object_type, _source_stage, _id_field in _COLLECTION_SPECS:
        values = getattr(collections, collection_name)
        counts[object_type] += len(values)
        if collection_name == "property_series":
            counts["property_series_point"] += sum(
                len(item.points) for item in values
            )
    return dict(sorted(counts.items()))


def _issue_warning(issue: ValidationIssue) -> ValidationIssue:
    return ValidationIssue(
        stage=issue.stage,
        code=f"preview_object_rejected_{issue.code}",
        message=f"Preview 已隔离对象；原错误：{issue.message}",
        object_id=issue.object_id,
    )


def _cleanup_warning(
    *, object_id: str, field: str, removed: Iterable[str]
) -> ValidationIssue:
    removed_values = sorted(set(removed))
    return ValidationIssue(
        stage="stage6_validate_merge",
        code="preview_reference_pruned",
        message=(
            f"Preview 删除 {object_id}.{field} 中的悬空引用：{removed_values}"
        ),
        object_id=object_id,
    )


def salvage_preview(
    collections: PreviewCollections,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> PreviewSalvageResult:
    """隔离可定位到对象的错误，并确定性清扫其余对象的悬空引用。"""

    input_counts = _counts(collections)
    original_index = _object_index(collections)
    issues_by_id: dict[str, list[ValidationIssue]] = defaultdict(list)
    remaining_errors: list[ValidationIssue] = []
    for issue in errors:
        if issue.object_id and issue.object_id in original_index:
            issues_by_id[issue.object_id].append(issue)
        else:
            remaining_errors.append(issue)

    rejected: dict[str, RejectedObject] = {}
    cleanup_count = 0

    def reject(
        object_id: str,
        *,
        code: str,
        message: str,
        raw_object: Any | None = None,
    ) -> None:
        if object_id in rejected:
            current = rejected[object_id]
            if code not in current.error_codes:
                current.error_codes.append(code)
            if message not in current.messages:
                current.messages.append(message)
            return
        indexed = original_index.get(object_id)
        if indexed is None and raw_object is None:
            return
        _collection, object_type, source_stage, indexed_object = indexed or (
            "",
            "unknown",
            "stage6_validate_merge",
            raw_object,
        )
        rejected[object_id] = RejectedObject(
            object_id=object_id,
            object_type=object_type,
            source_stage=source_stage,
            error_codes=[code],
            messages=[message],
            raw_object=_dump(raw_object if raw_object is not None else indexed_object),
        )
        if object_type == "property_series":
            for point in indexed_object.points:
                reject(
                    point.point_id,
                    code="preview_parent_series_rejected",
                    message=f"所属 PropertySeries {object_id} 已被隔离",
                    raw_object=point,
                )

    # unknown_property_reference 是可确定性剪掉的反向链接，不应先拒绝方法对象。
    property_ids = {
        str(item.property_id) for item in collections.property_observations
    } | {
        str(item.unresolved_id)
        for item in collections.unresolved_property_observations
    }
    cleaned_characterizations = []
    repaired_issue_ids: set[int] = set()
    for item in collections.characterizations:
        unknown = sorted(set(item.derived_property_ids) - property_ids)
        if unknown:
            cleaned_characterizations.append(
                _replace(
                    item,
                    derived_property_ids=[
                        value
                        for value in item.derived_property_ids
                        if value in property_ids
                    ],
                )
            )
            cleanup_count += len(unknown)
            warnings.append(
                _cleanup_warning(
                    object_id=item.characterization_id,
                    field="derived_property_ids",
                    removed=unknown,
                )
            )
            for issue in issues_by_id.get(item.characterization_id, []):
                if issue.code == "unknown_property_reference":
                    repaired_issue_ids.add(id(issue))
        else:
            cleaned_characterizations.append(item)
    collections.characterizations = cleaned_characterizations

    for object_id, object_issues in issues_by_id.items():
        remaining = [
            issue for issue in object_issues if id(issue) not in repaired_issue_ids
        ]
        for issue in remaining:
            reject(
                object_id,
                code=issue.code,
                message=issue.message,
            )
            warnings.append(_issue_warning(issue))

    # 先删除直接失败的顶层对象与 point。
    rejected_ids = set(rejected)
    for collection_name, _object_type, _source_stage, id_field in _COLLECTION_SPECS:
        values = getattr(collections, collection_name)
        if collection_name != "property_series":
            setattr(
                collections,
                collection_name,
                [item for item in values if str(getattr(item, id_field)) not in rejected_ids],
            )
            continue
        retained_series = []
        for series in values:
            if series.series_id in rejected_ids:
                continue
            points = [
                point for point in series.points if point.point_id not in rejected_ids
            ]
            if not points:
                reject(
                    series.series_id,
                    code="preview_series_without_valid_points",
                    message="PropertySeries 的全部 points 均被隔离",
                    raw_object=series,
                )
                continue
            retained_series.append(_replace_series_points(series, points))
        collections.property_series = retained_series

    # 引用清扫与依赖拒绝迭代到稳定。
    changed = True
    while changed:
        changed = False
        mention_ids = {item.mention_id for item in collections.material_mentions}
        entity_ids = {item.entity_id for item in collections.polymer_entities}
        sample_ids = {item.sample_id for item in collections.samples}
        condition_ids = {
            item.condition_id for item in collections.measurement_conditions
        }
        series_ids = {item.series_id for item in collections.property_series}
        characterization_ids = {
            item.characterization_id for item in collections.characterizations
        }
        property_ids = {
            item.property_id for item in collections.property_observations
        } | {
            item.unresolved_id
            for item in collections.unresolved_property_observations
        }

        new_entities = []
        for item in collections.polymer_entities:
            mentions = [
                value for value in item.resolved_from_mentions if value in mention_ids
            ]
            if not mentions:
                reject(
                    item.entity_id,
                    code="preview_missing_required_reference",
                    message="PolymerEntity 不再包含有效 mention",
                    raw_object=item,
                )
                changed = True
                continue
            variant_of = item.variant_of if item.variant_of in entity_ids else None
            if mentions != item.resolved_from_mentions or variant_of != item.variant_of:
                cleanup_count += (
                    len(item.resolved_from_mentions) - len(mentions)
                    + int(item.variant_of is not None and variant_of is None)
                )
                item = _replace(
                    item,
                    resolved_from_mentions=mentions,
                    variant_of=variant_of,
                )
            new_entities.append(item)
        collections.polymer_entities = new_entities

        def keep_or_reject(values: list[Any], invalid, label: str) -> list[Any]:
            nonlocal changed
            result = []
            for value in values:
                reason = invalid(value)
                if reason:
                    object_id = next(
                        str(getattr(value, field))
                        for field in (
                            "sample_id", "step_id", "condition_id", "property_id",
                            "unresolved_id", "series_id", "characterization_id",
                        )
                        if hasattr(value, field)
                    )
                    reject(
                        object_id,
                        code="preview_missing_required_reference",
                        message=f"{label}：{reason}",
                        raw_object=value,
                    )
                    changed = True
                else:
                    result.append(value)
            return result

        collections.samples = keep_or_reject(
            collections.samples,
            lambda item: (
                "refers_to_entity 已被隔离"
                if item.refers_to_entity and item.refers_to_entity not in entity_ids
                else None
            ),
            "Sample 引用失效",
        )
        collections.process_steps = keep_or_reject(
            collections.process_steps,
            lambda item: (
                "input/output sample 已被隔离"
                if (set(item.input_sample_ids) | set(item.output_sample_ids)) - sample_ids
                else None
            ),
            "ProcessStep 引用失效",
        )
        collections.property_observations = keep_or_reject(
            collections.property_observations,
            lambda item: (
                "sample 已被隔离"
                if getattr(item, "sample_id", None)
                and item.sample_id not in sample_ids
                else "entity 已被隔离"
                if getattr(item, "entity_id", None)
                and item.entity_id not in entity_ids
                else "samples 已包含被隔离对象"
                if set(getattr(item, "sample_ids", None) or []) - sample_ids
                else "entities 已包含被隔离对象"
                if set(getattr(item, "entity_ids", None) or []) - entity_ids
                else "measurement condition 已被隔离"
                if hasattr(item, "measurement_condition_id")
                and item.measurement_condition_id not in condition_ids
                else "characterization 已被隔离"
                if hasattr(item, "characterization_id")
                and item.characterization_id not in characterization_ids
                else None
            ),
            "Property 引用失效",
        )
        collections.unresolved_property_observations = keep_or_reject(
            collections.unresolved_property_observations,
            lambda item: (
                "entity 已被隔离" if item.entity_id not in entity_ids else None
            ),
            "Unresolved property 引用失效",
        )

        def clean_property_series_references(values: list[Any]) -> list[Any]:
            nonlocal cleanup_count, changed
            result = []
            for item in values:
                references = []
                if getattr(item, "series_id", None):
                    references.append(item.series_id)
                references.extend(getattr(item, "series_ids", None) or [])
                if not references:
                    result.append(item)
                    continue
                retained = [value for value in references if value in series_ids]
                removed = len(references) - len(retained)
                if not retained:
                    object_id = getattr(
                        item,
                        "property_id",
                        getattr(item, "unresolved_id", ""),
                    )
                    reject(
                        object_id,
                        code="preview_missing_required_reference",
                        message="aggregate property 的全部 Series 引用均已被隔离",
                        raw_object=item,
                    )
                    changed = True
                    continue
                updates = (
                    {"series_id": retained[0], "series_ids": None}
                    if len(retained) == 1
                    else {"series_id": None, "series_ids": retained}
                )
                if removed or updates["series_id"] != getattr(item, "series_id", None):
                    cleanup_count += removed
                    item = _replace(item, **updates)
                result.append(item)
            return result

        collections.property_observations = clean_property_series_references(
            collections.property_observations
        )
        collections.unresolved_property_observations = (
            clean_property_series_references(
                collections.unresolved_property_observations
            )
        )

        retained_series = []
        for series in collections.property_series:
            if (
                series.sample_id and series.sample_id not in sample_ids
            ) or (
                series.entity_id and series.entity_id not in entity_ids
            ):
                reject(
                    series.series_id,
                    code="preview_missing_required_reference",
                    message="PropertySeries 主体引用已被隔离",
                    raw_object=series,
                )
                changed = True
                continue
            points = []
            for point in series.points:
                if (
                    point.sample_id and point.sample_id not in sample_ids
                ) or (
                    point.entity_id and point.entity_id not in entity_ids
                ):
                    reject(
                        point.point_id,
                        code="preview_missing_required_reference",
                        message="PropertySeriesPoint 主体引用已被隔离",
                        raw_object=point,
                    )
                    changed = True
                    continue
                points.append(point)
            if not points:
                reject(
                    series.series_id,
                    code="preview_series_without_valid_points",
                    message="PropertySeries 不再包含有效 point",
                    raw_object=series,
                )
                changed = True
                continue
            retained_series.append(
                series
                if len(points) == len(series.points)
                else _replace_series_points(series, points)
            )
        collections.property_series = retained_series

        cleaned_characterizations = []
        for item in collections.characterizations:
            if (
                item.sample_id and item.sample_id not in sample_ids
            ) or (
                item.entity_id and item.entity_id not in entity_ids
            ) or set(item.sample_ids or []) - sample_ids or set(item.entity_ids or []) - entity_ids:
                reject(
                    item.characterization_id,
                    code="preview_missing_required_reference",
                    message="Characterization 主体引用已被隔离",
                    raw_object=item,
                )
                changed = True
                continue
            derived = [value for value in item.derived_property_ids if value in property_ids]
            series = [value for value in (item.series_ids or []) if value in series_ids]
            single_series = item.series_id if item.series_id in series_ids else None
            if single_series is None and len(series) == 1:
                single_series, series = series[0], []
            if len(series) < 2:
                series = []
            removed_count = (
                len(item.derived_property_ids) - len(derived)
                + len(item.series_ids or []) - len(series)
                + int(item.series_id is not None and single_series is None)
            )
            if removed_count:
                cleanup_count += removed_count
                item = _replace(
                    item,
                    derived_property_ids=derived,
                    series_id=single_series,
                    series_ids=series or None,
                )
            cleaned_characterizations.append(item)
        collections.characterizations = cleaned_characterizations

    # 只保留仍被发布对象引用的 Evidence。
    used_evidence: set[str] = set()

    def collect_evidence(value: Any) -> None:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="python")
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_ids" and isinstance(child, list):
                    used_evidence.update(str(item) for item in child)
                else:
                    collect_evidence(child)
        elif isinstance(value, list):
            for child in value:
                collect_evidence(child)

    for collection_name, *_rest in _COLLECTION_SPECS:
        collect_evidence(getattr(collections, collection_name))
    collections.evidence = [
        item for item in collections.evidence if item.evidence_id in used_evidence
    ]

    mention_ids = {item.mention_id for item in collections.material_mentions}
    entity_ids = {item.entity_id for item in collections.polymer_entities}
    unresolved_mentions = [
        value for value in collections.unresolved_mention_ids
        if value in mention_ids
    ]
    unresolved_entities = [
        value for value in collections.unresolved_entity_ids
        if value in entity_ids
    ]
    cleanup_count += (
        len(collections.unresolved_mention_ids) - len(unresolved_mentions)
        + len(collections.unresolved_entity_ids) - len(unresolved_entities)
    )
    collections.unresolved_mention_ids = unresolved_mentions
    collections.unresolved_entity_ids = unresolved_entities

    published_counts = _counts(collections)
    rejected_counts = dict(sorted(Counter(
        item.object_type for item in rejected.values()
    ).items()))
    object_types = set(input_counts) | set(published_counts) | set(rejected_counts)
    conservation_passed = all(
        input_counts.get(object_type, 0)
        == published_counts.get(object_type, 0)
        + rejected_counts.get(object_type, 0)
        for object_type in object_types
    )
    summary = PreviewPublicationSummary(
        input_counts=input_counts,
        published_counts=published_counts,
        rejected_counts=rejected_counts,
        reference_cleanup_count=cleanup_count,
        conservation_passed=conservation_passed,
    )
    return PreviewSalvageResult(
        collections=collections,
        remaining_errors=remaining_errors,
        warnings=warnings,
        rejected_objects=sorted(rejected.values(), key=lambda item: item.object_id),
        summary=summary,
    )
