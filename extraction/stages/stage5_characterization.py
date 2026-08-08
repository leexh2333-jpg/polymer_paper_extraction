"""Stage 5：抽取 Characterization 与结构表征数值。"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMCallCost,
    LLMCallRecord,
    LLMClient,
    LLMJSONResponse,
    LLMRequestError,
    LLMTokenUsage,
    extract_json_object,
    llm_failure_artifact,
    llm_config_cache_payload,
    load_pipeline_config,
    resolve_llm_config,
    resolve_pricing_config,
    summarize_client_calls,
)
from prompt_loader import PromptLoader, RenderedPrompt
from schema.polymer_schema import (
    compact_confidence_payload,
    Characterization,
    CharacterizationCandidate,
    CharacterizationStageResponse,
    Stage0Document,
    Stage0Element,
    Stage2Document,
    Stage3Document,
    Stage4Document,
    Stage5Document,
    Stage5PropertyCandidate,
    Stage5PropertyObservation,
    Stage5Provenance,
)
from stages.stage4_property import (
    _element_source_text,
    _materialize_evidence,
    _materialize_measurement_context,
    _normalize_evidence,
    _normalize_measurement_context,
    _normalize_raw_across_evidence,
    _resolve_surface_text,
    _sha256_json,
    write_json_atomic,
)


STAGE_ID = "stage5_characterization"
OUTPUT_SCHEMA_VERSION = "characterization_schema.v4"
IMPLEMENTATION_VERSION = "1.7.1"
# 1.7.1 Preview 对 Schema 合法但 evidence 语义失败的响应保留候选；
# 无法安全结构化时生成 degraded 空壳，Strict 行为不变。
# 1.7.0 仅增加 Preview 确定性收敛与表级方法证据支持；
# 旧版本成功产物仍满足当前输出契约，故缓存可复用。
COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS = (
    "1.6.6", "1.6.5", "1.6.4", "1.6.3", "1.6.2",
)
DEFAULT_INPUT_SECTIONS = ("Methods", "Results")
DEFAULT_VOCABULARY_PATH = EXTRACTION_ROOT / "config" / "polymer_schema.yaml"
STAGE5_PROPERTY_CATEGORIES = {"composition_structure", "morphology"}

MethodVocabulary = dict[str, tuple[str, ...]]
Stage5PropertyVocabulary = dict[str, tuple[str, frozenset[str]]]
STAGE5_EXACT_METHOD_ALIASES = {
    "FTIR": ("FT-IR", "IR"),
    "SEM": ("scanning electron microscope",),
    "TGA": ("thermogravimetric analyses",),
    "viscometry": ("inherent viscosities",),
}


class Stage5Error(RuntimeError):
    """Stage 5 输入、词表、LLM 响应或输出验证失败。"""


class _FailureReplayClient:
    """仅返回已保存响应，不发起网络请求。"""

    def __init__(
        self,
        *,
        resolved: Any,
        pricing: Any,
        response: LLMJSONResponse,
        record: LLMCallRecord,
        failure_path: Path,
    ) -> None:
        self.resolved = resolved
        self.pricing = pricing
        self.response = response
        self.record = record
        self.failure_path = failure_path
        self.call_history: list[LLMCallRecord] = []
        self.calls = 0

    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        if self.calls:
            raise Stage5Error("failure 响应只允许离线回放一次")
        self.calls += 1
        self.call_history.append(self.record)
        return self.response


def _failure_replay_client(
    failure_path: Path,
    config: dict[str, Any],
) -> _FailureReplayClient:
    if not failure_path.is_file():
        raise Stage5Error(f"缺少 Stage 5 failure 文件：{failure_path}")
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage5Error(f"Stage 5 failure 文件无效：{failure_path}") from exc
    raw = failure.get("raw_response") if isinstance(failure, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
        raise Stage5Error("Stage 5 failure 未保存可回放的 raw response")
    try:
        data = extract_json_object(raw["content"])
    except LLMRequestError as exc:
        raise Stage5Error(
            f"Stage 5 failure raw response 无法解析为 JSON 对象：{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise Stage5Error("Stage 5 failure raw response 必须是 JSON 对象")

    usage_data = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    usage = LLMTokenUsage(
        input_tokens=int(usage_data.get("input_tokens") or 0),
        output_tokens=int(usage_data.get("output_tokens") or 0),
        cache_creation_input_tokens=int(
            usage_data.get("cache_creation_input_tokens") or 0
        ),
        cache_read_input_tokens=int(
            usage_data.get("cache_read_input_tokens") or 0
        ),
    )
    cost_data = raw.get("cost") if isinstance(raw.get("cost"), dict) else None
    cost = (
        LLMCallCost(
            currency=str(cost_data["currency"]),
            input_per_million=Decimal(str(cost_data["input_per_million"])),
            output_per_million=Decimal(str(cost_data["output_per_million"])),
            input_cost=Decimal(str(cost_data["input_cost"])),
            output_cost=Decimal(str(cost_data["output_cost"])),
            total_cost=Decimal(str(cost_data["total_cost"])),
        )
        if cost_data is not None
        else None
    )
    provider = str(raw.get("provider") or "unknown")
    model = str(raw.get("model") or "unknown")
    resolved = resolve_llm_config(config, STAGE_ID)
    pricing = resolve_pricing_config(config, resolved.model)
    response = LLMJSONResponse(
        data=data,
        provider=provider,
        model=model,
        usage=usage,
        cost=cost,
    )
    record = LLMCallRecord(
        provider=provider,
        model=model,
        usage=usage,
        cost=cost,
        usage_available=bool(usage_data),
    )
    return _FailureReplayClient(
        resolved=resolved,
        pricing=pricing,
        response=response,
        record=record,
        failure_path=failure_path,
    )


def _load_model(path: Path, model: type[Any], label: str) -> Any:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        cleaned, _ = compact_confidence_payload(raw)
        return model.model_validate(cleaned)
    except OSError as exc:
        raise Stage5Error(f"无法读取 {label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage5Error(f"{label} JSON 无效：{path}") from exc
    except ValidationError as exc:
        raise Stage5Error(f"{label} 未通过 Schema：{path.name}") from exc


def load_stage0_document(path: Path) -> Stage0Document:
    return _load_model(path, Stage0Document, "Stage 0")


def load_stage2_document(path: Path) -> Stage2Document:
    return _load_model(path, Stage2Document, "Stage 2")


def load_stage3_document(path: Path) -> Stage3Document:
    return _load_model(path, Stage3Document, "Stage 3")


def load_stage4_document(path: Path) -> Stage4Document:
    return _load_model(path, Stage4Document, "Stage 4")


def load_characterization_vocabulary(
    path: Path,
) -> tuple[MethodVocabulary, Stage5PropertyVocabulary, str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Stage5Error(f"表征词表无效：{path}") from exc
    if raw.get("schema_version") != "1.0":
        raise Stage5Error("表征词表 schema_version 必须为 1.0")

    method_entries = raw.get("characterization_methods")
    if not isinstance(method_entries, dict) or not method_entries:
        raise Stage5Error("characterization_methods 必须为非空对象")
    methods: MethodVocabulary = {}
    for name, spec in method_entries.items():
        aliases = spec.get("aliases") if isinstance(spec, dict) else None
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(aliases, list)
            or not aliases
            or not all(
                isinstance(alias, str) and alias.strip() for alias in aliases
            )
        ):
            raise Stage5Error(f"表征方法词表条目无效：{name!r}")
        methods[name.strip()] = tuple(alias.strip() for alias in aliases)

    property_entries = raw.get("stage5_property_vocabulary")
    if not isinstance(property_entries, dict) or not property_entries:
        raise Stage5Error("stage5_property_vocabulary 必须为非空对象")
    properties: Stage5PropertyVocabulary = {}
    for name, spec in property_entries.items():
        category = spec.get("category") if isinstance(spec, dict) else None
        allowed_methods = spec.get("methods") if isinstance(spec, dict) else None
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            or category not in STAGE5_PROPERTY_CATEGORIES
            or not isinstance(allowed_methods, list)
            or not allowed_methods
            or not all(
                isinstance(method, str) and method in methods
                for method in allowed_methods
            )
        ):
            raise Stage5Error(f"Stage 5 性质词表条目无效：{name!r}")
        properties[name] = (category, frozenset(allowed_methods))
    return methods, properties, _sha256_json(raw)


def _upstream_evidence_block_ids(
    entities: Stage2Document,
    process: Stage3Document,
    properties: Stage4Document,
) -> set[str]:
    return {
        entity.evidence.block_id for entity in entities.polymer_entities
    } | {
        sample.evidence.block_id for sample in process.samples
    } | {
        step.evidence.block_id for step in process.process_steps
    } | {
        condition.evidence.block_id
        for condition in properties.measurement_conditions
    } | {
        evidence.block_id
        for item in properties.properties
        for evidence in item.evidence
    } | {
        evidence.block_id
        for item in properties.unresolved_properties
        for evidence in item.evidence
    }


def select_context_blocks(
    document: Stage0Document,
    entities: Stage2Document,
    process: Stage3Document,
    properties: Stage4Document,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 90000,
) -> tuple[list[Stage0Element], list[dict[str, Any]], int]:
    if max_input_chars < 2000:
        raise ValueError("max_input_chars 不得小于 2000")
    element_map = {element.block_id: element for element in document.elements}
    referenced_ids = _upstream_evidence_block_ids(
        entities,
        process,
        properties,
    )
    missing = sorted(referenced_ids - set(element_map))
    if missing:
        raise Stage5Error(f"上游输出引用了未知 block：{missing}")

    section_ids = {
        element.block_id
        for element in document.elements
        if element.section in input_sections
        and element.type in {
            "text",
            "title",
            "table",
            "image",
            "equation",
            "footnote",
        }
        and bool(_element_source_text(element))
    }
    selected_ids = section_ids | referenced_ids
    blocks = [
        element
        for element in document.elements
        if element.block_id in selected_ids
        and bool(_element_source_text(element))
    ]
    context_chars = sum(
        len(_element_source_text(element)) + 220 for element in blocks
    )
    if context_chars > max_input_chars:
        raise Stage5Error(
            f"{document.document_id} Stage 5 上下文 {context_chars} 字符，"
            f"超过 max_input_chars={max_input_chars}"
        )
    warnings: list[dict[str, Any]] = []
    if not section_ids and (entities.polymer_entities or process.samples):
        warnings.append({
            "stage": STAGE_ID,
            "code": "section_fallback",
            "message": (
                "Methods/Results 为空，仅使用上游 evidence block；结果需人工复核"
            ),
        })
    return blocks, warnings, context_chars


def _user_message(
    document_id: str,
    entities: Stage2Document,
    process: Stage3Document,
    properties: Stage4Document,
    blocks: list[Stage0Element],
    methods: MethodVocabulary,
    stage5_properties: Stage5PropertyVocabulary,
    validation_feedback: str | None = None,
) -> str:
    entity_data = [
        {
            "entity_id": item.entity_id,
            "polymer_name": item.polymer_name,
            "source_names": item.source_names,
        }
        for item in entities.polymer_entities
    ]
    sample_data = [
        {
            "sample_id": item.sample_id,
            "sample_kind": item.sample_kind,
            "refers_to_entity": item.refers_to_entity,
            "polymer_name": item.polymer_name,
        }
        for item in process.samples
    ]
    stage4_data = [
        {
            "property_id": item.property_id,
            "sample_id": item.sample_id,
            "entity_id": next(
                (
                    sample.refers_to_entity
                    for sample in process.samples
                    if sample.sample_id == item.sample_id
                ),
                None,
            ),
            "resolution_status": "resolved",
            "property_name_raw": item.property_name_raw,
            "property_name_normalized": item.property_name_normalized,
            "value_raw": item.value_raw,
            "unit_raw": item.unit_raw,
            "determination_method_raw": item.determination_method_raw,
            "observation_group_id": item.observation_group_id,
        }
        for item in properties.properties
    ] + [
        {
            "property_id": item.unresolved_id,
            "sample_id": None,
            "entity_id": item.entity_id,
            "resolution_status": "unresolved",
            "property_name_raw": item.property_name_raw,
            "property_name_normalized": item.property_name_normalized,
            "value_raw": item.value_raw,
            "unit_raw": item.unit_raw,
            "determination_method_raw": item.determination_method_raw,
            "observation_group_id": item.observation_group_id,
        }
        for item in properties.unresolved_properties
    ]
    stage4_series_data = [
        {
            "series_id": item.series_id,
            "sample_id": item.sample_id,
            "entity_id": item.entity_id,
            "sample_resolution_status": item.sample_resolution_status,
            "property_name_raw": item.property_name_raw,
            "property_name_normalized": item.property_name_normalized,
            "determination_method_raw": item.determination_method_raw,
            "point_count": len(item.points),
        }
        for item in properties.property_series
    ]
    method_data = {
        name: {"aliases": list(aliases)}
        for name, aliases in methods.items()
    }
    stage5_property_data = {
        name: {
            "property_category": category,
            "allowed_methods": sorted(allowed_methods),
        }
        for name, (category, allowed_methods) in stage5_properties.items()
    }
    block_data = [
        {
            "block_id": block.block_id,
            "page": block.page,
            "type": block.type,
            "section": block.section,
            "source_text": _element_source_text(block),
        }
        for block in blocks
    ]
    message = (
        f"document_id: {document_id}\n"
        "--- BEGIN UNTRUSTED POLYMER ENTITIES ---\n"
        + json.dumps(entity_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED POLYMER ENTITIES ---\n"
        "--- BEGIN UNTRUSTED SAMPLES ---\n"
        + json.dumps(sample_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED SAMPLES ---\n"
        "--- BEGIN EXISTING STAGE 4 PROPERTIES ---\n"
        + json.dumps(stage4_data, ensure_ascii=False, indent=2)
        + "\n--- END EXISTING STAGE 4 PROPERTIES ---\n"
        "--- BEGIN EXISTING STAGE 4 SERIES ---\n"
        + json.dumps(stage4_series_data, ensure_ascii=False, indent=2)
        + "\n--- END EXISTING STAGE 4 SERIES ---\n"
        "--- BEGIN CONTROLLED CHARACTERIZATION METHODS ---\n"
        + json.dumps(method_data, ensure_ascii=False, indent=2)
        + "\n--- END CONTROLLED CHARACTERIZATION METHODS ---\n"
        "--- BEGIN CONTROLLED STAGE 5 PROPERTY VOCABULARY ---\n"
        + json.dumps(stage5_property_data, ensure_ascii=False, indent=2)
        + "\n--- END CONTROLLED STAGE 5 PROPERTY VOCABULARY ---\n"
        "--- BEGIN UNTRUSTED METHODS AND RESULTS BLOCKS ---\n"
        + json.dumps(block_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED METHODS AND RESULTS BLOCKS ---"
    )
    if validation_feedback:
        message += (
            "\n\n上一次响应未通过校验。请重新输出完整 JSON。"
            f"错误类型：{validation_feedback}"
        )
    return message


def _validation_feedback(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for item in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in item.get("loc") or ())
            parts.append(f"{location}: {item.get('msg', 'validation error')}")
        return "; ".join(parts)[:800]
    return str(error)[:800]


def _method_names_for_raw(
    method_raw: str,
    methods: MethodVocabulary,
) -> set[str]:
    normalized_raw = re.sub(
        r"[^a-z0-9]+",
        " ",
        method_raw.casefold(),
    ).strip()
    if not normalized_raw:
        return set()
    matches: set[str] = set()
    for name, aliases in methods.items():
        for candidate in (
            name,
            *aliases,
            *STAGE5_EXACT_METHOD_ALIASES.get(name, ()),
        ):
            normalized_candidate = re.sub(
                r"[^a-z0-9]+",
                " ",
                candidate.casefold(),
            ).strip()
            padded_raw = f" {normalized_raw} "
            padded_candidate = f" {normalized_candidate} "
            symbolic_alias = any(
                character in candidate for character in "[]{}\\$"
            )
            if symbolic_alias and normalized_candidate != normalized_raw:
                continue
            if (
                normalized_candidate == normalized_raw
                or padded_candidate in padded_raw
                or padded_raw in padded_candidate
            ):
                matches.add(name)
                break
    return matches


def _normalize_characterization(
    item: CharacterizationCandidate,
    block_map: dict[str, Stage0Element],
    methods: MethodVocabulary,
) -> CharacterizationCandidate:
    if item.method_normalized not in methods:
        raise ValueError(
            f"未知 method_normalized：{item.method_normalized}"
        )
    if item.method_normalized not in _method_names_for_raw(
        item.method_raw,
        methods,
    ):
        raise ValueError(
            f"{item.characterization_id}.method_raw 与 "
            "method_normalized 不匹配"
        )
    anchors = [item.method_raw]
    if item.instrument:
        anchors.append(item.instrument)
    anchors.extend(item.parameters.values())
    evidence = [
        _normalize_evidence(
            candidate,
            block_map,
            anchors,
            allow_table_reference=True,
        )
        for candidate in item.evidence
    ]
    method_raw = _normalize_raw_across_evidence(
        item.method_raw,
        evidence,
        block_map,
        f"{item.characterization_id}.method_raw",
    )
    instrument = (
        _normalize_raw_across_evidence(
            item.instrument,
            evidence,
            block_map,
            f"{item.characterization_id}.instrument",
        )
        if item.instrument
        else None
    )
    parameters = {
        key: _normalize_raw_across_evidence(
            value,
            evidence,
            block_map,
            f"{item.characterization_id}.parameters.{key}",
        )
        for key, value in item.parameters.items()
    }
    measurement_context = (
        _normalize_measurement_context(
            item.measurement_context,
            evidence,
            block_map,
            item.characterization_id,
        )
        if item.measurement_context is not None
        else None
    )
    return item.model_copy(update={
        "method_raw": method_raw,
        "instrument": instrument,
        "parameters": parameters,
        "measurement_context": measurement_context,
        "evidence": evidence,
    })


def _normalize_stage5_property(
    item: Stage5PropertyCandidate,
    block_map: dict[str, Stage0Element],
    characterization: CharacterizationCandidate,
    vocabulary: Stage5PropertyVocabulary,
) -> Stage5PropertyCandidate:
    entry = vocabulary.get(item.property_name_normalized)
    if entry is None:
        raise ValueError(
            f"未知 Stage 5 property_name_normalized："
            f"{item.property_name_normalized}"
        )
    category, allowed_methods = entry
    if item.property_category != category:
        raise ValueError("Stage 5 性质名称与类别不匹配")
    if characterization.method_normalized not in allowed_methods:
        raise ValueError(
            f"{item.property_name_normalized} 不允许由 "
            f"{characterization.method_normalized} 产生"
        )

    anchors = [item.property_name_raw, item.value_raw]
    for value in (
        item.unit_raw,
        item.spectral_assignment,
        item.solvent,
    ):
        if value:
            anchors.append(value)
    evidence = [
        _normalize_evidence(candidate, block_map, anchors)
        for candidate in item.evidence
    ]
    try:
        property_name_raw = _normalize_raw_across_evidence(
            item.property_name_raw,
            evidence,
            block_map,
            f"{item.property_id}.property_name_raw",
        )
    except ValueError:
        property_name_raw = _normalize_raw_across_evidence(
            item.value_raw,
            evidence,
            block_map,
            f"{item.property_id}.property_name_raw",
        )
    updates: dict[str, Any] = {
        "property_name_raw": property_name_raw,
        "value_raw": _normalize_raw_across_evidence(
            item.value_raw,
            evidence,
            block_map,
            f"{item.property_id}.value_raw",
        ),
        "evidence": evidence,
    }
    for field in ("unit_raw", "spectral_assignment", "solvent"):
        value = getattr(item, field)
        updates[field] = (
            _normalize_raw_across_evidence(
                value,
                evidence,
                block_map,
                f"{item.property_id}.{field}",
            )
            if value
            else None
        )
    if item.measurement_context is not None:
        updates["measurement_context"] = _normalize_measurement_context(
            item.measurement_context,
            evidence,
            block_map,
            item.property_id,
        )
    return item.model_copy(update=updates)


def _stage5_property_dedupe_key(
    item: Stage5PropertyCandidate,
) -> tuple[Any, ...]:
    evidence_key = tuple(
        (
            evidence.block_id,
            evidence.source_sentence,
            evidence.table_locator.row_label
            if evidence.table_locator is not None else None,
            evidence.table_locator.column_label
            if evidence.table_locator is not None else None,
            evidence.table_locator.cell_value
            if evidence.table_locator is not None else None,
        )
        for evidence in item.evidence
    )
    return (
        item.characterization_id,
        item.sample_id,
        item.entity_id,
        tuple(item.sample_ids or []),
        tuple(item.entity_ids or []),
        item.property_name_normalized,
        item.value_raw,
        evidence_key,
    )


def _validate_response(
    response: LLMJSONResponse,
    entities: Stage2Document,
    process: Stage3Document,
    stage4: Stage4Document,
    blocks: list[Stage0Element],
    methods: MethodVocabulary,
    vocabulary: Stage5PropertyVocabulary,
    preview_scope_warnings: list[dict[str, Any]] | None = None,
) -> CharacterizationStageResponse:
    cleaned_data, _ = compact_confidence_payload(response.data)
    parsed = CharacterizationStageResponse.model_validate(cleaned_data)
    entity_ids = {item.entity_id for item in entities.polymer_entities}
    sample_ids = {item.sample_id for item in process.samples}
    sample_entities = {
        item.sample_id: item.refers_to_entity
        for item in process.samples
    }
    unresolved_by_id = {
        item.unresolved_id: item
        for item in stage4.unresolved_properties
    }
    stage4_property_ids = {
        item.property_id for item in stage4.properties
    } | set(unresolved_by_id)
    stage5_property_ids = {item.property_id for item in parsed.properties}
    series_by_id = {
        item.series_id: item for item in stage4.property_series
    }
    block_map = {block.block_id: block for block in blocks}

    characterizations = []
    seen_characterizations: set[tuple[Any, ...]] = set()
    for item in parsed.characterizations:
        series_references = set(item.series_ids or [])
        if item.series_id is not None:
            series_references.add(item.series_id)
        unknown_series = sorted(series_references - set(series_by_id))
        if unknown_series:
            raise ValueError(
                f"{item.characterization_id} 引用了未知 series：{unknown_series}"
            )
        if item.sample_id is not None and item.sample_id not in sample_ids:
            raise ValueError(
                f"{item.characterization_id} 引用了未知 sample："
                f"{item.sample_id}"
            )
        if item.entity_id is not None and item.entity_id not in entity_ids:
            raise ValueError(
                f"{item.characterization_id} 引用了未知 entity："
                f"{item.entity_id}"
            )
        unknown_samples = sorted(set(item.sample_ids or []) - sample_ids)
        if unknown_samples:
            raise ValueError(
                f"{item.characterization_id} 引用了未知 samples：{unknown_samples}"
            )
        unknown_entities = sorted(set(item.entity_ids or []) - entity_ids)
        if unknown_entities:
            raise ValueError(
                f"{item.characterization_id} 引用了未知 entities：{unknown_entities}"
            )
        if item.sample_ids is not None and item.entity_ids is not None:
            mapped_entities = {
                sample_entities.get(sample_id) for sample_id in item.sample_ids
            }
            if None in mapped_entities or mapped_entities != set(item.entity_ids):
                raise ValueError(
                    f"{item.characterization_id} 的多 Sample 与多 Entity 范围不一致"
                )
        sample_entity = (
            sample_entities.get(item.sample_id)
            if item.sample_id is not None
            else None
        )
        if (
            item.entity_id is not None
            and sample_entity is not None
            and item.entity_id != sample_entity
        ):
            raise ValueError(
                f"{item.characterization_id} 的 sample 与 entity 不一致"
            )
        mismatched_series = sorted(
            series_id
            for series_id in series_references
            if (
                item.sample_id != series_by_id[series_id].sample_id
                or (item.entity_id or sample_entity)
                != series_by_id[series_id].entity_id
            )
        )
        if mismatched_series:
            raise ValueError(
                f"{item.characterization_id} 与 Series 的归属不一致："
                f"{mismatched_series}"
            )
        unknown_properties = sorted(
            set(item.derived_property_ids)
            - stage4_property_ids
            - stage5_property_ids
        )
        if unknown_properties:
            raise ValueError(
                f"{item.characterization_id} 引用了未知 property："
                f"{unknown_properties}"
            )
        owner_entity = item.entity_id or sample_entity
        mismatched_unresolved = sorted(
            property_id
            for property_id in item.derived_property_ids
            if property_id in unresolved_by_id
            and unresolved_by_id[property_id].entity_id != owner_entity
        )
        if mismatched_unresolved:
            raise ValueError(
                f"{item.characterization_id} 跨实体引用 unresolved property："
                f"{mismatched_unresolved}"
            )
        normalized = _normalize_characterization(item, block_map, methods)
        # 去重键必须能区分「同一方法在不同位置的两次表征」。
        # 只用 (方法, sample, entity) 会把 reference_no_0071569 中
        # 薄膜 IR（P_8_66，N—H 3420/C=O 1685）与 KBr 压片 IR
        # （P_8_72，1760/1168/840）判成重复——它们是两次独立测量。
        # 因此把证据来源与派生 property 一并纳入：真正的重复输出
        # 这两项也必然相同，而不同测量至少有一项不同。
        key = (
            normalized.method_normalized,
            normalized.sample_id or tuple(normalized.sample_ids or []),
            normalized.entity_id or tuple(normalized.entity_ids or []),
            tuple(sorted(
                evidence.block_id for evidence in normalized.evidence
            )),
            tuple(sorted(normalized.derived_property_ids)),
        )
        if key in seen_characterizations:
            raise ValueError("存在重复 Characterization")
        seen_characterizations.add(key)
        characterizations.append(normalized)

    for property_item in stage4.properties:
        if not property_item.determination_method_raw:
            continue
        matching_methods = _method_names_for_raw(
            property_item.determination_method_raw,
            methods,
        )
        if not matching_methods:
            continue
        has_link = any(
            item.method_normalized in matching_methods
            and item.sample_id == property_item.sample_id
            and property_item.property_id in item.derived_property_ids
            for item in characterizations
        )
        if not has_link:
            raise ValueError(
                f"{property_item.property_id} 的测定方法 "
                f"{property_item.determination_method_raw!r} "
                "缺少同一样品 Characterization 回链"
            )
    for property_item in stage4.unresolved_properties:
        if not property_item.determination_method_raw:
            continue
        matching_methods = _method_names_for_raw(
            property_item.determination_method_raw,
            methods,
        )
        if not matching_methods:
            continue
        has_link = any(
            item.method_normalized in matching_methods
            and property_item.unresolved_id in item.derived_property_ids
            and (
                item.entity_id
                or (
                    sample_entities.get(item.sample_id)
                    if item.sample_id is not None
                    else None
                )
            )
            == property_item.entity_id
            for item in characterizations
        )
        if not has_link:
            raise ValueError(
                f"{property_item.unresolved_id} 的测定方法 "
                f"{property_item.determination_method_raw!r} "
                "缺少同一实体 Characterization 回链"
            )
    for series_item in stage4.property_series:
        if not series_item.determination_method_raw:
            continue
        matching_methods = _method_names_for_raw(
            series_item.determination_method_raw,
            methods,
        )
        if not matching_methods:
            continue
        has_link = any(
            item.method_normalized in matching_methods
            and series_item.series_id
            in ({item.series_id} if item.series_id is not None else set())
            | set(item.series_ids or [])
            for item in characterizations
        )
        if not has_link:
            raise ValueError(
                f"{series_item.series_id} 的测定方法 "
                f"{series_item.determination_method_raw!r} "
                "缺少 Characterization 回链"
            )

    characterization_map = {
        item.characterization_id: item for item in characterizations
    }
    properties = []
    seen_properties: set[tuple[Any, ...]] = set()
    for item in parsed.properties:
        if item.sample_id is not None and item.sample_id not in sample_ids:
            raise ValueError(
                f"{item.property_id} 引用了未知 sample：{item.sample_id}"
            )
        if item.entity_id is not None and item.entity_id not in entity_ids:
            raise ValueError(
                f"{item.property_id} 引用了未知 entity：{item.entity_id}"
            )
        unknown_samples = sorted(set(item.sample_ids or []) - sample_ids)
        if unknown_samples:
            raise ValueError(
                f"{item.property_id} 引用了未知 samples：{unknown_samples}"
            )
        unknown_entities = sorted(set(item.entity_ids or []) - entity_ids)
        if unknown_entities:
            raise ValueError(
                f"{item.property_id} 引用了未知 entities：{unknown_entities}"
            )
        owner = characterization_map[item.characterization_id]
        scope_mismatch = (
            item.sample_id != owner.sample_id
            or item.entity_id != owner.entity_id
            or item.sample_ids != owner.sample_ids
            or item.entity_ids != owner.entity_ids
            or item.sample_resolution_status != owner.sample_resolution_status
        )
        item_sample_ids = set(item.sample_ids or [])
        if item.sample_id is not None:
            item_sample_ids.add(item.sample_id)
        item_entity_ids = set(item.entity_ids or [])
        if item.entity_id is not None:
            item_entity_ids.add(item.entity_id)
        preview_subscope = (
            preview_scope_warnings is not None
            and owner.sample_resolution_status == "multi_resolved"
            and bool(item_sample_ids or item_entity_ids)
            and item_sample_ids.issubset(set(owner.sample_ids or []))
            and item_entity_ids.issubset(set(owner.entity_ids or []))
        )
        if scope_mismatch and not preview_subscope:
            raise ValueError(
                f"{item.property_id} 与所属 Characterization 的主体范围不一致"
            )
        if scope_mismatch:
            preview_scope_warnings.append({
                "property_id": item.property_id,
                "characterization_id": owner.characterization_id,
                "sample_id": item.sample_id,
                "entity_id": item.entity_id,
            })
        normalized = _normalize_stage5_property(
            item,
            block_map,
            owner,
            vocabulary,
        )
        key = _stage5_property_dedupe_key(normalized)
        if key in seen_properties:
            raise ValueError("存在重复 Stage 5 PropertyObservation")
        seen_properties.add(key)
        properties.append(normalized)
    return parsed.model_copy(update={
        "characterizations": characterizations,
        "properties": properties,
    })


def _repair_candidate_response_payload(
    payload: dict[str, Any],
    stage4: Stage4Document | None = None,
    process: Stage3Document | None = None,
    blocks: list[Stage0Element] | tuple[Stage0Element, ...] = (),
    methods: MethodVocabulary | None = None,
    *,
    preview_relaxed: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """将已知 Series 从性质引用字段迁移到专用关系字段。"""
    repaired, dropped_confidence_fields = compact_confidence_payload(payload)
    warnings: list[dict[str, Any]] = []
    if dropped_confidence_fields:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_fields_compacted",
            "message": "confidence 已确定性收敛为仅保留 score",
            "fields": list(dict.fromkeys(dropped_confidence_fields)),
        })
    characterizations = repaired.get("characterizations")
    if not isinstance(characterizations, list):
        return repaired, warnings
    block_map = {block.block_id: block for block in blocks}

    if preview_relaxed:
        for collection_name in ("characterizations", "properties"):
            collection = repaired.get(collection_name)
            if not isinstance(collection, list):
                continue
            for index, item in enumerate(collection):
                if not isinstance(item, dict):
                    continue
                changed = False
                for field in ("sample_ids", "entity_ids"):
                    if item.get(field) == []:
                        item[field] = None
                        changed = True
                status = item.get("sample_resolution_status")
                if status == "multi_resolved" and item.get("sample_id"):
                    item["sample_ids"] = None
                    item["entity_ids"] = None
                    item["sample_resolution_status"] = "resolved"
                    changed = True
                elif status == "multi_resolved" and item.get("entity_id"):
                    item["sample_ids"] = None
                    item["entity_ids"] = None
                    item["sample_resolution_status"] = "unresolved"
                    changed = True
                elif status == "multi_resolved":
                    sample_ids = item.get("sample_ids")
                    entity_ids = item.get("entity_ids")
                    if isinstance(sample_ids, list) and len(sample_ids) == 1:
                        item["sample_id"] = sample_ids[0]
                        item["entity_id"] = (
                            entity_ids[0]
                            if isinstance(entity_ids, list) and len(entity_ids) == 1
                            else None
                        )
                        item["sample_ids"] = None
                        item["entity_ids"] = None
                        item["sample_resolution_status"] = "resolved"
                        changed = True
                    elif isinstance(entity_ids, list) and len(entity_ids) == 1:
                        item["entity_id"] = entity_ids[0]
                        item["sample_ids"] = None
                        item["entity_ids"] = None
                        item["sample_resolution_status"] = "unresolved"
                        changed = True
                elif (
                    item.get("sample_id") is None
                    and item.get("entity_id") is None
                    and (
                        isinstance(item.get("sample_ids"), list)
                        and len(item["sample_ids"]) >= 2
                        or isinstance(item.get("entity_ids"), list)
                        and len(item["entity_ids"]) >= 2
                    )
                ):
                    item["sample_resolution_status"] = "multi_resolved"
                    changed = True
                if changed:
                    warnings.append({
                        "stage": STAGE_ID,
                        "code": "preview_subject_scope_normalized",
                        "message": "Preview 模式已将冲突或空的主体字段收敛为单一合法范围",
                        "field_path": f"{collection_name}[{index}]",
                    })

    if preview_relaxed and stage4 is not None and methods is not None:
        linked_series_ids = {
            series_id
            for item in characterizations
            if isinstance(item, dict)
            for series_id in (
                [item.get("series_id")]
                + (
                    item.get("series_ids")
                    if isinstance(item.get("series_ids"), list)
                    else []
                )
            )
            if isinstance(series_id, str)
        }
        existing_characterization_ids = {
            str(item.get("characterization_id"))
            for item in characterizations
            if isinstance(item, dict)
        }
        next_characterization_index = max(
            (
                int(match.group(1))
                for item_id in existing_characterization_ids
                if (match := re.fullmatch(r"char(\d+)", item_id))
            ),
            default=0,
        ) + 1
        for series in stage4.property_series:
            method_raw = series.determination_method_raw
            if (
                series.series_id in linked_series_ids
                or not method_raw
                or series.sample_resolution_status not in {
                    "resolved", "unresolved"
                }
            ):
                continue
            matching_methods = _method_names_for_raw(method_raw, methods)
            if len(matching_methods) != 1:
                continue
            method_normalized = next(iter(matching_methods))
            aliases = (
                method_normalized,
                *methods[method_normalized],
                *STAGE5_EXACT_METHOD_ALIASES.get(method_normalized, ()),
            )
            surfaces = {
                surface
                for alias in aliases
                if (surface := _resolve_surface_text(method_raw, alias))
                is not None
            }
            if not surfaces:
                continue
            method_surface = max(surfaces, key=len)
            evidence = [
                {
                    "block_id": item.block_id,
                    "source_sentence": item.source_sentence,
                    "table_locator": (
                        dict(item.table_locator)
                        if isinstance(item.table_locator, dict)
                        else (
                            item.table_locator.model_dump(mode="json")
                            if item.table_locator is not None
                            else None
                        )
                    ),
                }
                for item in series.evidence
            ]
            if not evidence:
                continue
            characterization_id = f"char{next_characterization_index:03d}"
            next_characterization_index += 1
            characterizations.append({
                "characterization_id": characterization_id,
                "method_raw": method_surface,
                "method_normalized": method_normalized,
                "sample_id": series.sample_id,
                "entity_id": series.entity_id,
                "sample_resolution_status": series.sample_resolution_status,
                "series_id": series.series_id,
                "derived_property_ids": [],
                "evidence": evidence,
                "confidence": {"score": 0.5},
            })
            linked_series_ids.add(series.series_id)
            warnings.append({
                "stage": STAGE_ID,
                "code": "preview_characterization_synthesized_from_series",
                "message": (
                    "Preview 模式已从方法唯一匹配且主体明确的 Stage 4 Series "
                    "补建低置信度 Characterization"
                ),
                "characterization_id": characterization_id,
                "series_id": series.series_id,
                "method_normalized": method_normalized,
            })

    if preview_relaxed:
        for collection_name in ("characterizations", "properties"):
            collection = repaired.get(collection_name)
            if not isinstance(collection, list):
                continue
            for index, item in enumerate(collection):
                if not isinstance(item, dict):
                    continue
                evidence = item.get("evidence")
                if not isinstance(evidence, list) or len(evidence) < 2:
                    continue
                incomplete_table_evidence = [
                    candidate
                    for candidate in evidence
                    if isinstance(candidate, dict)
                    and candidate.get("table_locator") is None
                    and (
                        block_map.get(str(candidate.get("block_id") or ""))
                        is not None
                    )
                    and block_map[str(candidate["block_id"])].type == "table"
                ]
                if not incomplete_table_evidence:
                    continue
                retained = [
                    candidate
                    for candidate in evidence
                    if candidate not in incomplete_table_evidence
                ]
                if not retained:
                    continue
                item["evidence"] = retained
                warnings.append({
                    "stage": STAGE_ID,
                    "code": "preview_incomplete_table_evidence_removed",
                    "message": (
                        "Preview 模式已移除缺少 table_locator 的附加表格 evidence；"
                        "同一对象仍保留其他原文 evidence"
                    ),
                    "field_path": f"{collection_name}[{index}].evidence",
                    "block_ids": [
                        candidate["block_id"]
                        for candidate in incomplete_table_evidence
                    ],
                })

    explicit_all_sample_ids: list[str] = []
    explicit_all_entity_ids: list[str] = []
    if process is not None:
        explicit_samples = [
            sample
            for sample in process.samples
            if sample.sample_kind == "synthesis_batch"
            and sample.refers_to_entity is not None
        ]
        explicit_all_sample_ids = [sample.sample_id for sample in explicit_samples]
        explicit_all_entity_ids = list(dict.fromkeys(
            sample.refers_to_entity for sample in explicit_samples
            if sample.refers_to_entity is not None
        ))
    characterization_scopes: dict[str, tuple[list[str], list[str]]] = {}
    explicit_all_block_ids = {
        block.block_id
        for block in blocks
        if re.search(
            r"\ball\s+synthesi[sz]ed\s+polymers\b",
            _element_source_text(block),
            re.I,
        )
    }
    for index, item in enumerate(characterizations):
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence")
        sentences = [
            str(candidate.get("source_sentence") or "")
            for candidate in evidence
            if isinstance(candidate, dict)
        ] if isinstance(evidence, list) else []
        explicitly_all_synthesized = any(
            re.search(r"\ball\s+synthesi[sz]ed\s+polymers\b", sentence, re.I)
            for sentence in sentences
        ) or any(
            str(candidate.get("block_id") or "") in explicit_all_block_ids
            for candidate in evidence
            if isinstance(candidate, dict)
        ) if isinstance(evidence, list) else False
        if (
            item.get("sample_id") is None
            and item.get("entity_id") is None
            and explicitly_all_synthesized
            and len(explicit_all_sample_ids) >= 2
            and len(explicit_all_entity_ids) >= 2
        ):
            item["sample_ids"] = explicit_all_sample_ids
            item["entity_ids"] = explicit_all_entity_ids
            item["sample_resolution_status"] = "multi_resolved"
            characterization_id = str(item.get("characterization_id") or "")
            characterization_scopes[characterization_id] = (
                explicit_all_sample_ids,
                explicit_all_entity_ids,
            )
            warnings.append({
                "stage": STAGE_ID,
                "code": "explicit_all_samples_scope_expanded",
                "message": (
                    "原文明示 all synthesized polymers；已展开为 Stage 3 中"
                    "全部带明确实体的 synthesis_batch"
                ),
                "field_path": f"characterizations[{index}]",
                "sample_ids": explicit_all_sample_ids,
                "entity_ids": explicit_all_entity_ids,
            })
        method_raw = item.get("method_raw")
        if not isinstance(method_raw, str) or not method_raw:
            continue
        current_evidence = item.get("evidence")
        evidence_items = (
            current_evidence if isinstance(current_evidence, list) else []
        )
        if any(
            isinstance(candidate.get("block_id"), str)
            and candidate["block_id"] in block_map
            and _resolve_surface_text(
                _element_source_text(block_map[candidate["block_id"]]),
                method_raw,
            ) is not None
            for candidate in evidence_items
            if isinstance(candidate, dict)
        ):
            continue
        unique_blocks = [
            block
            for block in blocks
            if _resolve_surface_text(
                _element_source_text(block),
                method_raw,
            ) is not None
        ]
        if len(unique_blocks) != 1:
            continue
        block = unique_blocks[0]
        item.setdefault("evidence", []).append({
            "block_id": block.block_id,
            "source_sentence": _element_source_text(block),
        })
        warnings.append({
            "stage": STAGE_ID,
            "code": "characterization_method_evidence_supplemented",
            "message": (
                "method_raw 仅在一个输入 block 中逐字出现；"
                "已补入该原文 evidence"
            ),
            "field_path": f"characterizations[{index}].method_raw",
            "block_id": block.block_id,
        })

    if preview_relaxed:
        for index, item in enumerate(characterizations):
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or len(evidence) < 2:
                continue
            incomplete_table_evidence = [
                candidate
                for candidate in evidence
                if isinstance(candidate, dict)
                and candidate.get("table_locator") is None
                and (
                    block_map.get(str(candidate.get("block_id") or ""))
                    is not None
                )
                and block_map[str(candidate["block_id"])].type == "table"
            ]
            retained = [
                candidate
                for candidate in evidence
                if candidate not in incomplete_table_evidence
            ]
            if not incomplete_table_evidence or not retained:
                continue
            item["evidence"] = retained
            warnings.append({
                "stage": STAGE_ID,
                "code": "preview_incomplete_table_evidence_removed",
                "message": (
                    "Preview 模式已移除缺少 table_locator 的附加表格 evidence；"
                    "同一对象仍保留其他原文 evidence"
                ),
                "field_path": f"characterizations[{index}].evidence",
                "block_ids": [
                    candidate["block_id"]
                    for candidate in incomplete_table_evidence
                ],
            })
    if preview_relaxed and block_map:
        for index, item in enumerate(characterizations):
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or len(evidence) < 2:
                continue
            unlocatable = [
                candidate
                for candidate in evidence
                if isinstance(candidate, dict)
                and (
                    block := block_map.get(
                        str(candidate.get("block_id") or "")
                    )
                ) is not None
                and _resolve_surface_text(
                    _element_source_text(block),
                    str(candidate.get("source_sentence") or ""),
                ) is None
            ]
            retained = [
                candidate for candidate in evidence
                if candidate not in unlocatable
            ]
            if not unlocatable or not retained:
                continue
            item["evidence"] = retained
            warnings.append({
                "stage": STAGE_ID,
                "code": "preview_unlocatable_additional_evidence_removed",
                "message": (
                    "Preview 模式已移除无法回定位到 Stage 0 block 的附加 "
                    "Characterization evidence"
                ),
                "field_path": f"characterizations[{index}].evidence",
                "block_ids": [
                    candidate.get("block_id") for candidate in unlocatable
                ],
            })
    if preview_relaxed and methods is not None:
        for index, item in enumerate(characterizations):
            if not isinstance(item, dict):
                continue
            method_raw = item.get("method_raw")
            method_normalized = item.get("method_normalized")
            if (
                not isinstance(method_raw, str)
                or not isinstance(method_normalized, str)
                or method_normalized not in methods
            ):
                continue
            evidence = item.get("evidence")
            evidence_items = evidence if isinstance(evidence, list) else []
            sources = [
                source
                for candidate in evidence_items
                if isinstance(candidate, dict)
                for source in [candidate.get("source_sentence")]
                if isinstance(source, str)
            ]
            if any(method_raw in source for source in sources):
                continue
            aliases = (
                method_normalized,
                *methods[method_normalized],
                *STAGE5_EXACT_METHOD_ALIASES.get(method_normalized, ()),
            )
            surfaces = {
                surface
                for alias in aliases
                for source in sources
                if (surface := _resolve_surface_text(source, alias)) is not None
            }
            if not surfaces:
                continue
            item["method_raw"] = max(surfaces, key=len)
            warnings.append({
                "stage": STAGE_ID,
                "code": "preview_characterization_method_surface_recovered",
                "message": "Preview 模式已将表征方法恢复为 evidence 中唯一匹配的原文词形",
                "field_path": f"characterizations[{index}].method_raw",
                "method_normalized": method_normalized,
            })
    if preview_relaxed and block_map:
        for index, item in enumerate(characterizations):
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence")
            evidence_items = evidence if isinstance(evidence, list) else []
            sources: list[str] = []
            for candidate in evidence_items:
                if not isinstance(candidate, dict):
                    continue
                block = block_map.get(str(candidate.get("block_id") or ""))
                if block is not None:
                    sources.append(_element_source_text(block))
                    continue
                source_sentence = candidate.get("source_sentence")
                if isinstance(source_sentence, str):
                    sources.append(source_sentence)

            def supported(value: Any) -> bool:
                return not isinstance(value, str) or any(
                    value in source for source in sources
                )

            def evidence_sources(value: Any) -> list[str]:
                if not isinstance(value, list):
                    return []
                result: list[str] = []
                for candidate in value:
                    if not isinstance(candidate, dict):
                        continue
                    source_sentence = candidate.get("source_sentence")
                    if isinstance(source_sentence, str):
                        result.append(source_sentence)
                    block = block_map.get(str(candidate.get("block_id") or ""))
                    if block is not None:
                        result.append(_element_source_text(block))
                return result

            cleared_fields: list[str] = []
            if item.get("instrument") and not supported(item["instrument"]):
                item["instrument"] = None
                cleared_fields.append("instrument")
            parameters = item.get("parameters")
            if isinstance(parameters, dict):
                retained_parameters = {
                    key: value
                    for key, value in parameters.items()
                    if supported(value)
                }
                if retained_parameters != parameters:
                    item["parameters"] = retained_parameters
                    cleared_fields.append("parameters")
            context = item.get("measurement_context")
            if isinstance(context, dict):
                for field in (
                    "temperature",
                    "frequency",
                    "humidity",
                    "pressure",
                    "wavelength",
                ):
                    quantity = context.get(field)
                    if isinstance(quantity, dict):
                        raw = quantity.get("raw")
                        field_sources = evidence_sources(quantity.get("evidence"))
                        if isinstance(raw, str) and not any(
                            raw in source for source in field_sources
                        ):
                            context[field] = None
                            cleared_fields.append(
                                f"measurement_context.{field}"
                            )
                other_conditions = context.get("other_conditions")
                if isinstance(other_conditions, dict):
                    condition_evidence = context.get(
                        "other_condition_evidence"
                    )
                    retained_conditions = {
                        key: value
                        for key, value in other_conditions.items()
                        if isinstance(condition_evidence, dict)
                        and any(
                            value in source
                            for source in evidence_sources(
                                condition_evidence.get(key)
                            )
                        )
                    }
                    if retained_conditions != other_conditions:
                        context["other_conditions"] = retained_conditions
                        if isinstance(condition_evidence, dict):
                            context["other_condition_evidence"] = {
                                key: value
                                for key, value in condition_evidence.items()
                                if key in retained_conditions
                            }
                        cleared_fields.append(
                            "measurement_context.other_conditions"
                        )
                has_context_value = any(
                    context.get(field) is not None
                    for field in (
                        "temperature",
                        "frequency",
                        "humidity",
                        "pressure",
                        "wavelength",
                    )
                ) or bool(context.get("other_conditions"))
                if not has_context_value:
                    item["measurement_context"] = None
                    cleared_fields.append("measurement_context")
            if cleared_fields:
                warnings.append({
                    "stage": STAGE_ID,
                    "code": "preview_unsupported_characterization_details_cleared",
                    "message": "Preview 模式已清空无对象 evidence 支撑的可选表征细节",
                    "field_path": f"characterizations[{index}]",
                    "fields": cleared_fields,
                })
    properties = repaired.get("properties")
    if preview_relaxed and isinstance(properties, list):
        for index, item in enumerate(properties):
            if not isinstance(item, dict):
                continue
            changed: dict[str, str] = {}
            for plural, singular in (
                ("sample_ids", "sample_id"),
                ("entity_ids", "entity_id"),
            ):
                values = item.get(plural)
                if (
                    item.get(singular) is None
                    and isinstance(values, list)
                    and len(values) == 1
                    and isinstance(values[0], str)
                ):
                    item[singular] = values[0]
                    item[plural] = None
                    changed[plural] = singular
            if changed:
                item["sample_resolution_status"] = "resolved"
                warnings.append({
                    "stage": STAGE_ID,
                    "code": "preview_singleton_subject_scope_normalized",
                    "message": "Preview 模式已将单元素主体列表归一为单值字段",
                    "field_path": f"properties[{index}]",
                    "fields": changed,
                })

    if preview_relaxed and isinstance(properties, list) and block_map:
        retained_properties = []
        removed_property_ids: set[str] = set()
        for index, item in enumerate(properties):
            if not isinstance(item, dict):
                retained_properties.append(item)
                continue
            evidence = item.get("evidence")
            evidence_items = evidence if isinstance(evidence, list) else []
            sources: list[str] = []
            for candidate in evidence_items:
                if not isinstance(candidate, dict):
                    continue
                block = block_map.get(str(candidate.get("block_id") or ""))
                if block is not None:
                    sources.append(_element_source_text(block))
                    continue
                source_sentence = candidate.get("source_sentence")
                if isinstance(source_sentence, str):
                    sources.append(source_sentence)

            def supported(field: str) -> bool:
                value = item.get(field)
                return (
                    not isinstance(value, str)
                    or not any(ord(character) < 32 for character in value)
                    and re.search(r"\\u[0-9a-fA-F]{4}", value) is None
                    and any(value in source for source in sources)
                )

            property_id = item.get("property_id")
            if not supported("value_raw"):
                if isinstance(property_id, str):
                    removed_property_ids.add(property_id)
                warnings.append({
                    "stage": STAGE_ID,
                    "code": "preview_unsupported_stage5_property_removed",
                    "message": "Preview 模式已移除 value_raw 无原文支撑的 Stage 5 性质",
                    "field_path": f"properties[{index}].value_raw",
                    "property_id": property_id,
                })
                continue
            unsupported_optional_fields = [
                field
                for field in ("unit_raw", "spectral_assignment", "solvent")
                if item.get(field) and not supported(field)
            ]
            if unsupported_optional_fields:
                for field in unsupported_optional_fields:
                    item[field] = None
                if "unit_raw" in unsupported_optional_fields:
                    item["unit_normalized"] = None
                warnings.append({
                    "stage": STAGE_ID,
                    "code": "preview_unsupported_stage5_optional_fields_cleared",
                    "message": "Preview 模式已清空无原文支撑的 Stage 5 可选字段",
                    "field_path": f"properties[{index}]",
                    "property_id": property_id,
                    "fields": unsupported_optional_fields,
                })
            retained_properties.append(item)
        if removed_property_ids:
            for characterization in characterizations:
                if not isinstance(characterization, dict):
                    continue
                derived_ids = characterization.get("derived_property_ids")
                if isinstance(derived_ids, list):
                    characterization["derived_property_ids"] = [
                        item for item in derived_ids
                        if item not in removed_property_ids
                    ]
        repaired["properties"] = retained_properties
        properties = retained_properties
    if isinstance(properties, list):
        for index, item in enumerate(properties):
            if not isinstance(item, dict):
                continue
            scope = characterization_scopes.get(str(
                item.get("characterization_id") or ""
            ))
            if (
                scope is None
                or item.get("sample_id") is not None
                or item.get("entity_id") is not None
            ):
                continue
            item["sample_ids"], item["entity_ids"] = scope
            item["sample_resolution_status"] = "multi_resolved"
            warnings.append({
                "stage": STAGE_ID,
                "code": "stage5_property_subject_scope_inherited",
                "message": "Stage 5 property 已继承所属 Characterization 的明确多主体范围",
                "field_path": f"properties[{index}]",
            })

    known_series = (
        {item.series_id for item in stage4.property_series}
        if stage4 is not None
        else set()
    )
    series_map = (
        {item.series_id: item for item in stage4.property_series}
        if stage4 is not None
        else {}
    )
    for index, item in enumerate(characterizations):
        if not isinstance(item, dict):
            continue
        series_id = item.get("series_id")
        series_ids = item.get("series_ids")
        references = (
            [series_id]
            if isinstance(series_id, str)
            else (
                series_ids
                if isinstance(series_ids, list)
                and all(isinstance(value, str) for value in series_ids)
                else []
            )
        )
        referenced_series = [
            series_map[value] for value in references if value in series_map
        ]
        if (
            item.get("sample_resolution_status") != "unresolved"
            or item.get("sample_id") is not None
            or item.get("entity_id") is not None
            or item.get("sample_ids") is not None
            or item.get("entity_ids") is not None
            or not references
            or len(referenced_series) != len(references)
            or any(
                series.sample_resolution_status != "unresolved"
                or not series.points
                for series in referenced_series
            )
            or any(
                point.sample_resolution_status != "resolved"
                or point.sample_id is None
                or point.entity_id is None
                for series in referenced_series
                for point in series.points
            )
        ):
            continue
        scopes: list[tuple[list[str], list[str], dict[str, str]]] = []
        conflicting_subject = False
        for series in referenced_series:
            sample_ids = list(dict.fromkeys(
                point.sample_id for point in series.points
                if point.sample_id is not None
            ))
            entity_ids = list(dict.fromkeys(
                point.entity_id for point in series.points
                if point.entity_id is not None
            ))
            sample_entities: dict[str, str] = {}
            for point in series.points:
                previous = sample_entities.setdefault(
                    str(point.sample_id),
                    str(point.entity_id),
                )
                if previous != point.entity_id:
                    conflicting_subject = True
                    break
            scopes.append((sample_ids, entity_ids, sample_entities))
        sample_ids, entity_ids, sample_entities = scopes[0]
        same_scope = all(
            set(candidate_samples) == set(sample_ids)
            and set(candidate_entities) == set(entity_ids)
            and candidate_mapping == sample_entities
            for candidate_samples, candidate_entities, candidate_mapping in scopes[1:]
        )
        compatible_scope = same_scope
        if preview_relaxed and not same_scope:
            maximal_scopes = [
                candidate
                for candidate in scopes
                if all(
                    set(other_samples).issubset(set(candidate[0]))
                    and set(other_entities).issubset(set(candidate[1]))
                    and all(
                        candidate[2].get(sample_id) == entity_id
                        for sample_id, entity_id in other_mapping.items()
                    )
                    for other_samples, other_entities, other_mapping in scopes
                )
            ]
            if len(maximal_scopes) == 1:
                sample_ids, entity_ids, sample_entities = maximal_scopes[0]
                compatible_scope = True
        if (
            conflicting_subject
            or not compatible_scope
            or len(sample_ids) < 2
            or len(entity_ids) < 2
        ):
            continue
        item["sample_ids"] = sample_ids
        item["entity_ids"] = entity_ids
        item["sample_resolution_status"] = "multi_resolved"
        characterization_id = str(item.get("characterization_id") or "")
        characterization_scopes[characterization_id] = (
            sample_ids,
            entity_ids,
        )
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_point_subject_scope_inherited",
            "message": (
                "Characterization 已从关联且主体范围一致的跨主体 Series points "
                "继承完整主体范围"
            ),
            "field_path": f"characterizations[{index}]",
            "series_ids": references,
            "sample_ids": sample_ids,
            "entity_ids": entity_ids,
        })

    if isinstance(properties, list):
        for index, item in enumerate(properties):
            if not isinstance(item, dict):
                continue
            scope = characterization_scopes.get(str(
                item.get("characterization_id") or ""
            ))
            if (
                scope is None
                or item.get("sample_resolution_status") != "unresolved"
                or item.get("sample_id") is not None
                or item.get("entity_id") is not None
                or item.get("sample_ids") is not None
                or item.get("entity_ids") is not None
            ):
                continue
            item["sample_ids"], item["entity_ids"] = scope
            item["sample_resolution_status"] = "multi_resolved"
            warnings.append({
                "stage": STAGE_ID,
                "code": "stage5_property_subject_scope_inherited",
                "message": "Stage 5 property 已继承所属 Characterization 的明确多主体范围",
                "field_path": f"properties[{index}]",
            })

    for index, item in enumerate(characterizations):
        if not isinstance(item, dict):
            continue
        series_id = item.get("series_id")
        series_ids = item.get("series_ids")
        derived_ids = item.get("derived_property_ids")
        if not isinstance(derived_ids, list):
            continue
        references = []
        if isinstance(series_id, str) and series_id:
            references.append(series_id)
        if isinstance(series_ids, list):
            references.extend(
                candidate
                for candidate in series_ids
                if isinstance(candidate, str) and candidate
            )
        migrated = [
            candidate
            for candidate in derived_ids
            if isinstance(candidate, str) and candidate in known_series
        ]
        references = list(dict.fromkeys([*references, *migrated]))
        removable = set(references) & set(derived_ids)
        if not removable:
            continue
        item["derived_property_ids"] = [
            property_id
            for property_id in derived_ids
            if property_id not in removable
        ]
        if len(references) == 1:
            item["series_id"] = references[0]
            item.pop("series_ids", None)
            code = "duplicate_series_derived_reference_removed"
            message = (
                f"{item.get('characterization_id') or f'characterizations[{index}]'} "
                f"的 series_id {references[0]} 已从 derived_property_ids 移除"
            )
        else:
            item["series_id"] = None
            item["series_ids"] = references
            code = "series_references_moved_from_derived_properties"
            message = (
                f"{item.get('characterization_id') or f'characterizations[{index}]'} "
                "的已知 Series 已从 derived_property_ids 迁移到 series_ids"
            )
        warnings.append({
            "stage": STAGE_ID,
            "code": code,
            "message": message,
            "field_path": f"characterizations[{index}].derived_property_ids",
            "series_ids": references,
        })
    if preview_relaxed:
        unscoped_characterization_ids = {
            str(item.get("characterization_id"))
            for item in characterizations
            if isinstance(item, dict)
            and item.get("sample_resolution_status") == "unresolved"
            and all(
                item.get(field) is None
                for field in (
                    "sample_id",
                    "entity_id",
                    "sample_ids",
                    "entity_ids",
                )
            )
        }
        if unscoped_characterization_ids:
            repaired["characterizations"] = [
                item
                for item in characterizations
                if not isinstance(item, dict)
                or str(item.get("characterization_id"))
                not in unscoped_characterization_ids
            ]
            characterizations = repaired["characterizations"]
            removed_property_ids = []
            if isinstance(properties, list):
                retained_properties = []
                for item in properties:
                    if (
                        isinstance(item, dict)
                        and str(item.get("characterization_id"))
                        in unscoped_characterization_ids
                    ):
                        removed_property_ids.append(item.get("property_id"))
                        continue
                    retained_properties.append(item)
                repaired["properties"] = retained_properties
                properties = retained_properties
            warnings.append({
                "stage": STAGE_ID,
                "code": "preview_unscoped_stage5_items_removed",
                "message": (
                    "Preview 模式已移除无法唯一关联任何单/多主体的 Stage 5 "
                    "表征及其派生性质；未猜测 entity_id"
                ),
                "characterization_ids": sorted(
                    unscoped_characterization_ids
                ),
                "property_ids": [
                    item for item in removed_property_ids
                    if isinstance(item, str)
                ],
            })
    characterization_map = {
        item.get("characterization_id"): item
        for item in characterizations
        if isinstance(item, dict)
        and isinstance(item.get("characterization_id"), str)
    }
    if preview_relaxed and isinstance(properties, list):
        scope_fields = (
            "sample_id",
            "entity_id",
            "sample_ids",
            "entity_ids",
            "sample_resolution_status",
        )
        for index, property_item in enumerate(properties):
            if not isinstance(property_item, dict):
                continue
            owner = characterization_map.get(
                property_item.get("characterization_id")
            )
            if owner is None or all(
                property_item.get(field) == owner.get(field)
                for field in scope_fields
            ):
                continue
            property_sample_ids = set(property_item.get("sample_ids") or [])
            if property_item.get("sample_id") is not None:
                property_sample_ids.add(property_item["sample_id"])
            property_entity_ids = set(property_item.get("entity_ids") or [])
            if property_item.get("entity_id") is not None:
                property_entity_ids.add(property_item["entity_id"])
            valid_multi_subject_subset = (
                owner.get("sample_resolution_status") == "multi_resolved"
                and bool(property_sample_ids or property_entity_ids)
                and property_sample_ids.issubset(set(owner.get("sample_ids") or []))
                and property_entity_ids.issubset(set(owner.get("entity_ids") or []))
            )
            if valid_multi_subject_subset:
                continue
            for field in scope_fields:
                property_item[field] = copy.deepcopy(owner.get(field))
            warnings.append({
                "stage": STAGE_ID,
                "code": "preview_property_subject_scope_aligned",
                "message": "Preview 模式已将冲突的派生性质主体对齐到所属 Characterization",
                "field_path": f"properties[{index}]",
                "characterization_id": property_item.get("characterization_id"),
            })
    if isinstance(properties, list):
        for index, property_item in enumerate(properties):
            if not isinstance(property_item, dict):
                continue
            property_id = property_item.get("property_id")
            characterization_id = property_item.get("characterization_id")
            owner = characterization_map.get(characterization_id)
            if (
                not isinstance(property_id, str)
                or owner is None
            ):
                continue
            current_owners = [
                candidate_id
                for candidate_id, candidate in characterization_map.items()
                if property_id in (candidate.get("derived_property_ids") or [])
            ]
            if current_owners:
                if (
                    not preview_relaxed
                    or current_owners == [characterization_id]
                ):
                    continue
                for current_owner_id in current_owners:
                    current_owner = characterization_map[current_owner_id]
                    current_owner["derived_property_ids"] = [
                        candidate
                        for candidate in (
                            current_owner.get("derived_property_ids") or []
                        )
                        if candidate != property_id
                    ]
                derived_ids = owner.get("derived_property_ids")
                if not isinstance(derived_ids, list):
                    continue
                derived_ids.append(property_id)
                warnings.append({
                    "stage": STAGE_ID,
                    "code": "preview_property_back_reference_reassigned",
                    "message": (
                        "Preview 模式已按 property.characterization_id "
                        "修正冲突的 derived_property_ids 反向引用"
                    ),
                    "field_path": f"properties[{index}].characterization_id",
                    "characterization_id": characterization_id,
                    "previous_characterization_ids": current_owners,
                    "property_id": property_id,
                })
                continue
            derived_ids = owner.get("derived_property_ids")
            if not isinstance(derived_ids, list):
                continue
            derived_ids.append(property_id)
            warnings.append({
                "stage": STAGE_ID,
                "code": "derived_property_back_reference_completed",
                "message": (
                    "Stage 5 property 已明确引用唯一 Characterization；"
                    "已补齐其 derived_property_ids 反向引用"
                ),
                "field_path": (
                    f"properties[{index}].characterization_id"
                ),
                "characterization_id": characterization_id,
                "property_id": property_id,
            })
    return repaired, warnings


def _materialize(
    parsed: CharacterizationStageResponse,
    blocks: list[Stage0Element],
) -> tuple[list[Characterization], list[Stage5PropertyObservation]]:
    block_map = {block.block_id: block for block in blocks}
    characterization_id_map = {
        item.characterization_id: f"char{index:03d}"
        for index, item in enumerate(parsed.characterizations, start=1)
    }
    property_id_map = {
        item.property_id: f"prop_s5_{index:03d}"
        for index, item in enumerate(parsed.properties, start=1)
    }
    characterizations = [
        Characterization(
            characterization_id=characterization_id_map[
                item.characterization_id
            ],
            method_raw=item.method_raw,
            method_normalized=item.method_normalized,
            sample_id=item.sample_id,
            entity_id=item.entity_id,
            sample_ids=item.sample_ids,
            entity_ids=item.entity_ids,
            sample_resolution_status=item.sample_resolution_status,
            series_id=item.series_id,
            series_ids=item.series_ids,
            instrument=item.instrument,
            measurement_context=(
                _materialize_measurement_context(
                    item.measurement_context, block_map
                )
                if item.measurement_context is not None
                else None
            ),
            parameters=item.parameters,
            result_summary=item.result_summary,
            derived_property_ids=[
                property_id_map.get(property_id, property_id)
                for property_id in item.derived_property_ids
            ],
            evidence=[
                _materialize_evidence(evidence, block_map)
                for evidence in item.evidence
            ],
            confidence=item.confidence,
        )
        for item in parsed.characterizations
    ]
    properties = [
        Stage5PropertyObservation(
            property_id=property_id_map[item.property_id],
            characterization_id=characterization_id_map[
                item.characterization_id
            ],
            sample_id=item.sample_id,
            entity_id=item.entity_id,
            sample_ids=item.sample_ids,
            entity_ids=item.entity_ids,
            sample_resolution_status=item.sample_resolution_status,
            property_name_raw=item.property_name_raw,
            property_name_normalized=item.property_name_normalized,
            property_category=item.property_category,
            value_raw=item.value_raw,
            value_min=item.value_min,
            value_max=item.value_max,
            unit_raw=item.unit_raw,
            unit_normalized=item.unit_normalized,
            measurement_context=(
                _materialize_measurement_context(
                    item.measurement_context, block_map
                )
                if item.measurement_context is not None
                else None
            ),
            spectral_assignment=item.spectral_assignment,
            solvent=item.solvent,
            source_stage=item.source_stage,
            source_type=block_map[item.evidence[0].block_id].type,
            evidence=[
                _materialize_evidence(evidence, block_map)
                for evidence in item.evidence
            ],
            confidence=item.confidence,
        )
        for item in parsed.properties
    ]
    return characterizations, properties


def _cache_components(
    document: Stage0Document,
    entities: Stage2Document,
    process: Stage3Document,
    stage4: Stage4Document,
    prompt: RenderedPrompt,
    vocabulary_sha256: str,
    client: LLMClient,
    *,
    implementation_version: str = IMPLEMENTATION_VERSION,
    preview_relaxed: bool = False,
) -> tuple[str, str, str]:
    input_hash = _sha256_json({
        "stage0": document.model_dump(mode="json"),
        "stage2": entities.model_dump(mode="json"),
        "stage3": process.model_dump(mode="json"),
        "stage4": stage4.model_dump(mode="json"),
    })
    model_config_hash = _sha256_json(
        llm_config_cache_payload(client.resolved)
    )
    cache_key = _sha256_json({
        "input_hash": input_hash,
        "prompt_id": prompt.prompt_id,
        "prompt_version": prompt.version,
        "rendered_prompt_hash": prompt.sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "model_config_hash": model_config_hash,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation_version": implementation_version,
        "preview_relaxed": preview_relaxed,
    })
    return input_hash, model_config_hash, cache_key


def extract_characterizations(
    document: Stage0Document,
    entities: Stage2Document,
    process: Stage3Document,
    stage4: Stage4Document,
    client: LLMClient,
    prompt: RenderedPrompt,
    methods: MethodVocabulary,
    vocabulary: Stage5PropertyVocabulary,
    vocabulary_sha256: str,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 90000,
    max_validation_retries: int = 1,
    max_tokens: int = 16384,
    preview_relaxed: bool = False,
) -> Stage5Document:
    history_start = len(getattr(client, "call_history", []))
    if not (
        document.document_id
        == entities.document_id
        == process.document_id
        == stage4.document_id
    ):
        raise Stage5Error(
            "Stage 0、Stage 2、Stage 3 与 Stage 4 document_id 不一致"
        )
    blocks, warnings, context_chars = select_context_blocks(
        document,
        entities,
        process,
        stage4,
        input_sections=input_sections,
        max_input_chars=max_input_chars,
    )
    actual_models: list[str] = []
    successful_repair_warnings: list[dict[str, Any]] = []
    preview_scope_warnings: list[dict[str, Any]] = []
    preview_semantic_bypass_reason: str | None = None
    preview_degraded_reason: str | None = None

    if entities.polymer_entities:
        feedback = None
        last_error: Exception | None = None
        parsed: CharacterizationStageResponse | None = None
        for attempt in range(max_validation_retries + 1):
            try:
                response = client.call_json(
                    prompt.text,
                    _user_message(
                        document.document_id,
                        entities,
                        process,
                        stage4,
                        blocks,
                        methods,
                        vocabulary,
                        feedback,
                    ),
                    max_tokens=max_tokens,
                )
                repaired_payload, repair_warnings = (
                    _repair_candidate_response_payload(
                        response.data,
                        stage4,
                        process,
                        blocks,
                        methods,
                        preview_relaxed=preview_relaxed,
                    )
                )
                try:
                    parsed = _validate_response(
                        LLMJSONResponse(
                            data=repaired_payload,
                            provider=response.provider,
                            model=response.model,
                            usage=response.usage,
                            cost=response.cost,
                        ),
                        entities,
                        process,
                        stage4,
                        blocks,
                        methods,
                        vocabulary,
                        preview_scope_warnings=(
                            preview_scope_warnings
                            if preview_relaxed
                            else None
                        ),
                    )
                except ValueError as exc:
                    if not preview_relaxed:
                        raise
                    # Preview 只要求响应结构可用；原文/evidence 语义失败时
                    # 保留已通过 Schema 的候选，Strict 继续负责完整合规。
                    parsed = CharacterizationStageResponse.model_validate(
                        repaired_payload
                    )
                    preview_semantic_bypass_reason = _validation_feedback(exc)
                successful_repair_warnings = repair_warnings
                actual_models.append(response.model)
                last_error = None
                break
            except (LLMRequestError, ValidationError, ValueError) as exc:
                last_error = exc
                feedback = _validation_feedback(exc)
                if attempt >= max_validation_retries:
                    break
        if last_error is not None or parsed is None:
            failure_reason = _validation_feedback(
                last_error or ValueError("empty")
            )
            if preview_relaxed:
                parsed = CharacterizationStageResponse()
                preview_degraded_reason = failure_reason
            else:
                raise Stage5Error(
                    f"{document.document_id} 响应校验失败：{failure_reason}"
                ) from last_error
    else:
        parsed = CharacterizationStageResponse()

    warnings.extend(successful_repair_warnings)
    if preview_scope_warnings:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_property_subject_subscope_retained",
            "message": (
                "Preview 模式已保留多主体 Characterization 下的明确单主体性质；"
                "需在正式数据中复核"
            ),
            "items": preview_scope_warnings,
        })
    if isinstance(client, _FailureReplayClient):
        warnings.append({
            "stage": STAGE_ID,
            "code": "failure_response_replayed",
            "message": "已离线回放保存的 Stage 5 响应，未请求模型",
            "source": client.failure_path.name,
        })
    try:
        characterizations, properties = _materialize(parsed, blocks)
    except (KeyError, ValidationError, ValueError) as exc:
        if not preview_relaxed:
            raise Stage5Error(
                f"{document.document_id} 响应物化失败：{_validation_feedback(exc)}"
            ) from exc
        parsed = CharacterizationStageResponse()
        characterizations, properties = _materialize(parsed, blocks)
        preview_degraded_reason = (
            "结构化候选无法安全物化：" + _validation_feedback(exc)
        )
    if preview_semantic_bypass_reason is not None:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_semantic_validation_bypassed",
            "message": (
                "Preview 已保留 Schema 合法候选，并跳过原文/evidence "
                "语义对应校验；Strict 模式仍会报错"
            ),
            "reason": preview_semantic_bypass_reason,
        })
    if preview_degraded_reason is not None:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_degraded_empty_shell",
            "message": (
                "模型响应无法安全结构化，Preview 已生成 degraded 空壳结果"
            ),
            "degraded": True,
            "reason": preview_degraded_reason,
        })
    unresolved_characterizations = [
        item.characterization_id
        for item in characterizations
        if item.sample_resolution_status == "unresolved"
    ]
    if unresolved_characterizations:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unresolved_characterizations",
            "message": (
                f"{len(unresolved_characterizations)} 条表征无法可靠关联具体 Sample"
            ),
            "characterization_ids": unresolved_characterizations,
        })
    unresolved_properties = [
        item.property_id
        for item in properties
        if item.sample_resolution_status == "unresolved"
    ]
    if unresolved_properties:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unresolved_stage5_properties",
            "message": (
                f"{len(unresolved_properties)} 条 Stage 5 性质无法可靠关联具体 Sample"
            ),
            "property_ids": unresolved_properties,
        })

    input_hash, model_config_hash, cache_key = _cache_components(
        document,
        entities,
        process,
        stage4,
        prompt,
        vocabulary_sha256,
        client,
        preview_relaxed=preview_relaxed,
    )
    unique_models = list(dict.fromkeys(actual_models))
    if not unique_models:
        unique_models = [client.resolved.model]
    usage, cost = summarize_client_calls(
        client,
        history_start,
        call_count=len(actual_models),
    )
    provenance = Stage5Provenance(
        provider=client.resolved.provider,
        model=unique_models[-1],
        models=unique_models,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        vocabulary_sha256=vocabulary_sha256,
        input_hash=input_hash,
        model_config_hash=model_config_hash,
        cache_key=cache_key,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        implementation_version=IMPLEMENTATION_VERSION,
        context_block_count=len(blocks),
        context_chars=context_chars,
        call_count=len(actual_models),
        usage=usage,
        cost=cost,
    )
    return Stage5Document(
        document_id=document.document_id,
        characterizations=characterizations,
        properties=properties,
        provenance=provenance,
        warnings=warnings,
    )


def run_stage5(
    stage0_path: Path,
    stage2_path: Path,
    stage3_path: Path,
    stage4_path: Path,
    output_path: Path,
    client: LLMClient,
    prompt: RenderedPrompt,
    methods: MethodVocabulary,
    vocabulary: Stage5PropertyVocabulary,
    vocabulary_sha256: str,
    *,
    force: bool = False,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 90000,
    max_validation_retries: int = 1,
    max_tokens: int = 16384,
    preview_relaxed: bool = False,
) -> tuple[Path, bool]:
    document = load_stage0_document(stage0_path)
    entities = load_stage2_document(stage2_path)
    process = load_stage3_document(stage3_path)
    stage4 = load_stage4_document(stage4_path)
    _, _, expected_cache_key = _cache_components(
        document,
        entities,
        process,
        stage4,
        prompt,
        vocabulary_sha256,
        client,
        preview_relaxed=preview_relaxed,
    )
    if output_path.is_file() and not force:
        try:
            cached = Stage5Document.model_validate_json(
                output_path.read_text(encoding="utf-8-sig")
            )
            if cached.provenance.cache_key == expected_cache_key:
                return output_path, True
            for compatible_version in COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS:
                _, _, compatible_cache_key = _cache_components(
                    document,
                    entities,
                    process,
                    stage4,
                    prompt,
                    vocabulary_sha256,
                    client,
                    implementation_version=compatible_version,
                    preview_relaxed=preview_relaxed,
                )
                if (
                    cached.provenance.implementation_version
                    == compatible_version
                    and cached.provenance.cache_key == compatible_cache_key
                ):
                    return output_path, True
        except (OSError, ValidationError):
            pass

    result = extract_characterizations(
        document,
        entities,
        process,
        stage4,
        client,
        prompt,
        methods,
        vocabulary,
        vocabulary_sha256,
        input_sections=input_sections,
        max_input_chars=max_input_chars,
        max_validation_retries=max_validation_retries,
        max_tokens=max_tokens,
        preview_relaxed=preview_relaxed,
    )
    write_json_atomic(
        output_path,
        result.model_dump(mode="json", exclude_none=True),
    )
    return output_path, False


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages") or {}
    stage = stages.get(STAGE_ID) or {}
    if not isinstance(stage, dict):
        raise Stage5Error(f"配置 {STAGE_ID} 必须是对象")
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 Stage 5 表征抽取")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ref-no")
    mode.add_argument("--batch", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--max-input-chars", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--preview-relaxed",
        action="store_true",
        help="演示模式：确定性归一化可保留的主体范围结果",
    )
    parser.add_argument(
        "--replay-failure",
        action="store_true",
        help="离线回放现有 stage5_failure.json，不请求模型",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    stage_config = _stage_config(config)
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
    vocabulary_path = (
        args.vocabulary.expanduser().resolve()
        if args.vocabulary
        else Path(
            stage_config.get("vocabulary_path") or DEFAULT_VOCABULARY_PATH
        ).resolve()
    )
    methods, vocabulary, vocabulary_sha256 = (
        load_characterization_vocabulary(vocabulary_path)
    )
    prompt_id = str(
        stage_config.get("prompt_id") or "polymer.stage5.characterization"
    )
    prompt = PromptLoader().render_stage_prompt(
        prompt_id,
        CharacterizationStageResponse,
        expected_stage=STAGE_ID,
        expected_output_schema=OUTPUT_SCHEMA_VERSION,
    )
    if args.replay_failure and not args.ref_no:
        raise Stage5Error("--replay-failure 必须与单篇 --ref-no 配合使用")
    client = (
        _failure_replay_client(
            input_root / args.ref_no / "stage5_failure.json",
            config,
        )
        if args.replay_failure
        else LLMClient.from_pipeline_config(
            stage=STAGE_ID,
            config_path=config_path,
        )
    )
    input_sections = tuple(
        stage_config.get("input_sections") or DEFAULT_INPUT_SECTIONS
    )
    max_input_chars = int(
        args.max_input_chars
        or stage_config.get("max_input_chars")
        or 90000
    )
    max_validation_retries = (
        0
        if args.replay_failure
        else int(stage_config.get("max_validation_retries", 1))
    )
    max_tokens = int(stage_config.get("max_tokens") or 16384)

    if args.ref_no:
        ref_nos = [args.ref_no]
    else:
        ref_nos = sorted(
            path.parent.name
            for path in input_root.glob("reference_no_*/stage4_properties.json")
        )
    if not ref_nos:
        raise Stage5Error(f"未找到 Stage 4 输出：{input_root}")

    failures: list[tuple[str, str]] = []
    for ref_no in ref_nos:
        history_start = len(client.call_history)
        try:
            output_path, cached = run_stage5(
                input_root / ref_no / "stage0_blocks.json",
                input_root / ref_no / "stage2_entities.json",
                input_root / ref_no / "stage3_process.json",
                input_root / ref_no / "stage4_properties.json",
                output_root / ref_no / "stage5_characterizations.json",
                client,
                prompt,
                methods,
                vocabulary,
                vocabulary_sha256,
                force=args.force,
                input_sections=input_sections,
                max_input_chars=max_input_chars,
                max_validation_retries=max_validation_retries,
                max_tokens=max_tokens,
                preview_relaxed=args.preview_relaxed,
            )
            state = "cached" if cached else "done"
            print(f"[{state}] {ref_no} -> {output_path}")
        except Exception as exc:
            if not args.replay_failure:
                write_json_atomic(
                    output_root / ref_no / "stage5_failure.json",
                    llm_failure_artifact(
                        client,
                        stage=STAGE_ID,
                        document_id=ref_no,
                        error=exc,
                        history_start=history_start,
                    ),
                )
            failures.append((ref_no, type(exc).__name__))
            print(f"[failed] {ref_no}: {exc}", file=sys.stderr)
    print(f"Stage 5 完成：成功 {len(ref_nos) - len(failures)}，失败 {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
