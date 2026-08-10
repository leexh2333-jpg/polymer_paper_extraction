"""Stage 2：使用 LLM 将 MaterialMention 解析为 PolymerEntity。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

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
    Evidence,
    PolymerEntity,
    PolymerEntityResponse,
    SourceImageReference,
    Stage0Document,
    Stage0Element,
    Stage1Document,
    Stage2Document,
    Stage2Provenance,
)


STAGE_ID = "stage2_polymer_entity"
OUTPUT_SCHEMA_VERSION = "polymer_entity_schema.v2"
IMPLEMENTATION_VERSION = "1.3.5"
DEFAULT_INPUT_SECTIONS = ("Methods", "Results")


class Stage2Error(RuntimeError):
    """Stage 2 输入、LLM 响应或输出验证失败。"""


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
            raise Stage2Error("failure 响应只允许离线回放一次")
        self.calls += 1
        self.call_history.append(self.record)
        return self.response


def _failure_replay_client(
    failure_path: Path,
    config: dict[str, Any],
) -> _FailureReplayClient:
    if not failure_path.is_file():
        raise Stage2Error(f"缺少 Stage 2 failure 文件：{failure_path}")
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2Error(f"Stage 2 failure 文件无效：{failure_path}") from exc
    raw = failure.get("raw_response") if isinstance(failure, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
        raise Stage2Error("Stage 2 failure 未保存可回放的 raw response")
    try:
        data = extract_json_object(raw["content"])
    except LLMRequestError as exc:
        raise Stage2Error(
            f"Stage 2 failure raw response 无法解析为 JSON 对象：{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise Stage2Error("Stage 2 failure raw response 必须是 JSON 对象")

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


def _sha256_json(data: Any) -> str:
    canonical = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _load_model(path: Path, model: type[Any], label: str) -> Any:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        cleaned, _ = compact_confidence_payload(raw)
        return model.model_validate(cleaned)
    except OSError as exc:
        raise Stage2Error(f"无法读取 {label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage2Error(f"{label} JSON 无效：{path}") from exc
    except ValidationError as exc:
        raise Stage2Error(f"{label} 未通过 Schema：{path.name}") from exc


def load_stage0_document(path: Path) -> Stage0Document:
    return _load_model(path, Stage0Document, "Stage 0")


def load_stage1_document(path: Path) -> Stage1Document:
    return _load_model(path, Stage1Document, "Stage 1")


def _element_source_text(element: Stage0Element) -> str:
    if element.type in {"text", "title", "equation", "footnote"}:
        return (element.text or "").strip()
    if element.type == "table":
        return "\n".join(
            value.strip()
            for value in (element.caption, element.table_body)
            if value and value.strip()
        )
    if element.type == "image":
        return (element.caption or "").strip()
    return ""


def select_context_blocks(
    document: Stage0Document,
    mentions: Stage1Document,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 60000,
) -> tuple[list[Stage0Element], list[dict[str, Any]], int]:
    if max_input_chars < 2000:
        raise ValueError("max_input_chars 不得小于 2000")
    element_map = {element.block_id: element for element in document.elements}
    missing = sorted({
        mention.evidence.block_id
        for mention in mentions.material_mentions
        if mention.evidence.block_id not in element_map
    })
    if missing:
        raise Stage2Error(f"Stage 1 evidence 引用了未知 block：{missing}")

    section_ids = {
        element.block_id
        for element in document.elements
        if element.section in input_sections
        and (
            (
                element.type in {"text", "title"}
                and bool((element.text or "").strip())
            )
            or (
                element.type == "image"
                and bool((element.caption or element.image_path or "").strip())
            )
        )
    }
    mention_ids = {
        mention.evidence.block_id for mention in mentions.material_mentions
    }
    selected_ids = section_ids | mention_ids
    blocks = [
        element
        for element in document.elements
        if element.block_id in selected_ids
    ]
    context_chars = sum(
        len(_element_source_text(element)) + 160 for element in blocks
    )
    if context_chars > max_input_chars:
        raise Stage2Error(
            f"{document.document_id} Stage 2 上下文 {context_chars} 字符，"
            f"超过 max_input_chars={max_input_chars}"
        )

    warnings: list[dict[str, Any]] = []
    if not section_ids and mentions.material_mentions:
        warnings.append({
            "stage": STAGE_ID,
            "code": "section_fallback",
            "message": (
                "Methods/Results 为空，仅使用 Stage 1 mention 的 evidence block；"
                "结果需人工复核"
            ),
        })
    return blocks, warnings, context_chars


def _user_message(
    document_id: str,
    mentions: Stage1Document,
    blocks: list[Stage0Element],
    validation_feedback: str | None = None,
) -> str:
    mention_data = [
        {
            "mention_id": mention.mention_id,
            "text": mention.text,
            "mention_role": mention.mention_role,
            "evidence": {
                "block_id": mention.evidence.block_id,
                "source_sentence": mention.evidence.source_sentence,
            },
        }
        for mention in mentions.material_mentions
    ]
    block_data = [
        {
            "block_id": block.block_id,
            "page": block.page,
            "type": block.type,
            "section": block.section,
            "source_text": _element_source_text(block),
            "image_path": block.image_path if block.type == "image" else None,
        }
        for block in blocks
    ]
    message = (
        f"document_id: {document_id}\n"
        "--- BEGIN UNTRUSTED MATERIAL MENTIONS ---\n"
        + json.dumps(mention_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED MATERIAL MENTIONS ---\n"
        "--- BEGIN UNTRUSTED CONTEXT BLOCKS ---\n"
        + json.dumps(block_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED CONTEXT BLOCKS ---"
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


def _nested_mention_pairs(
    mentions: Stage1Document,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    items = mentions.material_mentions
    for shorter in items:
        sentence = shorter.evidence.source_sentence
        short_spans = [
            match.span()
            for match in re.finditer(
                re.escape(shorter.text),
                sentence,
                flags=re.IGNORECASE,
            )
        ]
        if not short_spans:
            continue
        for longer in items:
            if shorter.mention_id == longer.mention_id:
                continue
            if len(shorter.text) >= len(longer.text):
                continue
            if shorter.evidence.block_id != longer.evidence.block_id:
                continue
            if sentence != longer.evidence.source_sentence:
                continue
            long_spans = [
                match.span()
                for match in re.finditer(
                    re.escape(longer.text),
                    sentence,
                    flags=re.IGNORECASE,
                )
            ]
            if not long_spans:
                continue
            if all(
                any(
                    long_start <= short_start and short_end <= long_end
                    for long_start, long_end in long_spans
                )
                for short_start, short_end in short_spans
            ):
                pairs.add((shorter.mention_id, longer.mention_id))
    return pairs


def _recover_whitespace_equivalent_surface(
    source: str,
    candidate: str,
) -> str | None:
    """仅忽略空白差异，在 source 中恢复 candidate 的逐字表面文本。"""
    compact_source: list[str] = []
    source_indices: list[int] = []
    for index, character in enumerate(source):
        if character.isspace():
            continue
        compact_source.append(character)
        source_indices.append(index)
    compact_candidate = "".join(
        character for character in candidate if not character.isspace()
    )
    if not compact_candidate:
        return None
    position = "".join(compact_source).find(compact_candidate)
    if position < 0:
        return None
    start = source_indices[position]
    end = source_indices[position + len(compact_candidate) - 1] + 1
    return source[start:end]


def _validate_response(
    response: LLMJSONResponse,
    mentions: Stage1Document,
    blocks: list[Stage0Element],
    dropped_confidence_fields: list[str] | None = None,
    surface_repairs: list[dict[str, str]] | None = None,
    allowed_nested_splits: list[dict[str, str]] | None = None,
    preview_duplicate_mention_repairs: list[dict[str, Any]] | None = None,
    preview_nested_splits: list[dict[str, str]] | None = None,
    preview_evidence_fallbacks: list[dict[str, str]] | None = None,
    preview_invalid_entities_removed: list[dict[str, Any]] | None = None,
) -> PolymerEntityResponse:
    cleaned_data, dropped = compact_confidence_payload(response.data)
    parsed = PolymerEntityResponse.model_validate(cleaned_data)
    mention_map = {
        mention.mention_id: mention for mention in mentions.material_mentions
    }
    block_map = {block.block_id: block for block in blocks}
    normalized_entities = []
    removed_mention_ids: set[str] = set()
    for entity in parsed.entities:
        unknown_mentions = sorted(
            set(entity.resolved_from_mentions) - set(mention_map)
        )
        if unknown_mentions:
            raise ValueError(
                f"{entity.entity_id} 引用了未知 mention：{unknown_mentions}"
            )
        source_names = {
            mention_map[mention_id].text
            for mention_id in entity.resolved_from_mentions
        }
        if entity.polymer_name not in source_names:
            if preview_invalid_entities_removed is not None:
                removed_mention_ids.update(entity.resolved_from_mentions)
                preview_invalid_entities_removed.append({
                    "entity_id": entity.entity_id,
                    "polymer_name": entity.polymer_name,
                    "mention_ids": list(entity.resolved_from_mentions),
                })
                continue
            raise ValueError(
                f"{entity.entity_id}.polymer_name 必须来自 resolved mention"
            )

        evidence_block = block_map.get(entity.evidence.block_id)
        if evidence_block is None:
            raise ValueError(
                f"{entity.entity_id} 引用了未知 evidence block："
                f"{entity.evidence.block_id}"
            )
        source_text = _element_source_text(evidence_block)
        if entity.evidence.source_sentence not in source_text:
            recovered = _recover_whitespace_equivalent_surface(
                source_text,
                entity.evidence.source_sentence,
            )
            if recovered is None and preview_evidence_fallbacks is not None:
                matching_mentions = [
                    mention_map[mention_id]
                    for mention_id in entity.resolved_from_mentions
                    if mention_id in mention_map
                    and mention_map[mention_id].text == entity.polymer_name
                    and mention_map[mention_id].evidence.block_id
                    == evidence_block.block_id
                    and mention_map[mention_id].evidence.source_sentence
                    in source_text
                ]
                fallback_sentences = {
                    mention.evidence.source_sentence
                    for mention in matching_mentions
                }
                if len(fallback_sentences) == 1:
                    recovered = next(iter(fallback_sentences))
                    preview_evidence_fallbacks.append({
                        "entity_id": entity.entity_id,
                        "mention_id": matching_mentions[0].mention_id,
                        "block_id": evidence_block.block_id,
                    })
            if recovered is None:
                raise ValueError(
                    f"{entity.entity_id}.evidence.source_sentence 不是原文子串"
                )
            if surface_repairs is not None:
                surface_repairs.append({
                    "entity_id": entity.entity_id,
                    "block_id": evidence_block.block_id,
                    "field": "evidence.source_sentence",
                })
            entity = entity.model_copy(update={
                "evidence": entity.evidence.model_copy(
                    update={"source_sentence": recovered}
                )
            })

        for image_block_id in entity.source_image_block_ids:
            image_block = block_map.get(image_block_id)
            if image_block is None or image_block.type != "image":
                raise ValueError(
                    f"{entity.entity_id} 引用了无效 image block："
                    f"{image_block_id}"
                )
        normalized_entities.append(entity)

    parsed = parsed.model_copy(update={
        "entities": normalized_entities,
        "unresolved_mention_ids": list(dict.fromkeys([
            *parsed.unresolved_mention_ids,
            *sorted(removed_mention_ids),
        ])),
    })

    assignments: dict[str, list[int]] = {}
    for entity_index, entity in enumerate(parsed.entities):
        for mention_id in entity.resolved_from_mentions:
            assignments.setdefault(mention_id, []).append(entity_index)
    duplicate_assignments = {
        mention_id: entity_indices
        for mention_id, entity_indices in assignments.items()
        if len(entity_indices) > 1
    }
    if duplicate_assignments and preview_duplicate_mention_repairs is None:
        raise ValueError("同一 mention 不得解析到多个实体")

    repaired_entities = list(parsed.entities)
    removed_entity_indices: set[int] = set()
    duplicate_unresolved_ids: set[str] = set()
    for mention_id, original_entity_indices in duplicate_assignments.items():
        entity_indices = [
            entity_index
            for entity_index in original_entity_indices
            if entity_index not in removed_entity_indices
            and mention_id
            in repaired_entities[entity_index].resolved_from_mentions
        ]
        if len(entity_indices) < 2:
            continue
        mention = mention_map[mention_id]
        owners = [
            entity_index
            for entity_index in entity_indices
            if (
                repaired_entities[entity_index].polymer_name == mention.text
                and repaired_entities[entity_index].evidence.block_id
                == mention.evidence.block_id
            )
        ]
        owner_index = owners[0] if len(owners) == 1 else None
        removed_from_entity_ids: list[str] = []
        removed_entity_ids: list[str] = []
        for entity_index in entity_indices:
            if entity_index == owner_index:
                continue
            entity = repaired_entities[entity_index]
            remaining = [
                item
                for item in entity.resolved_from_mentions
                if item != mention_id
            ]
            remaining_names = {
                mention_map[item].text
                for item in remaining
            }
            removed_from_entity_ids.append(entity.entity_id)
            if remaining and entity.polymer_name in remaining_names:
                repaired_entities[entity_index] = entity.model_copy(update={
                    "resolved_from_mentions": remaining,
                })
                continue
            removed_entity_indices.add(entity_index)
            removed_entity_ids.append(entity.entity_id)
            duplicate_unresolved_ids.update(remaining)
        if owner_index is None:
            duplicate_unresolved_ids.add(mention_id)
        preview_duplicate_mention_repairs.append({
            "mention_id": mention_id,
            "action": (
                "kept_unique_text_and_evidence_owner"
                if owner_index is not None
                else "marked_unresolved"
            ),
            "kept_entity_id": (
                repaired_entities[owner_index].entity_id
                if owner_index is not None
                else None
            ),
            "removed_from_entity_ids": removed_from_entity_ids,
            "removed_entity_ids": removed_entity_ids,
            "evidence_block_id": mention.evidence.block_id,
        })

    repaired_entities = [
        entity
        for entity_index, entity in enumerate(repaired_entities)
        if entity_index not in removed_entity_indices
    ]
    repaired_resolved_ids = {
        mention_id
        for entity in repaired_entities
        for mention_id in entity.resolved_from_mentions
    }
    repaired_unresolved_ids = list(dict.fromkeys([
        *parsed.unresolved_mention_ids,
        *sorted(duplicate_unresolved_ids),
    ]))
    parsed = parsed.model_copy(update={
        "entities": repaired_entities,
        "unresolved_mention_ids": [
            mention_id
            for mention_id in repaired_unresolved_ids
            if mention_id not in repaired_resolved_ids
        ],
    })

    resolved_ids = [
        mention_id
        for entity in parsed.entities
        for mention_id in entity.resolved_from_mentions
    ]
    if len(resolved_ids) != len(set(resolved_ids)):
        raise ValueError("同一 mention 不得解析到多个实体")
    unresolved_ids = parsed.unresolved_mention_ids
    unknown_unresolved = sorted(set(unresolved_ids) - set(mention_map))
    if unknown_unresolved:
        raise ValueError(f"存在未知 unresolved mention：{unknown_unresolved}")
    if set(resolved_ids) & set(unresolved_ids):
        raise ValueError("resolved 与 unresolved mention 不得重叠")
    covered_ids = set(resolved_ids) | set(unresolved_ids)
    if covered_ids != set(mention_map):
        missing = sorted(set(mention_map) - covered_ids)
        parsed = parsed.model_copy(update={
            "unresolved_mention_ids": [
                *parsed.unresolved_mention_ids,
                *missing,
            ]
        })
        unresolved_ids = parsed.unresolved_mention_ids

    assignments = {
        mention_id: entity.entity_id
        for entity in parsed.entities
        for mention_id in entity.resolved_from_mentions
    }
    assignments.update({
        mention_id: "__unresolved__"
        for mention_id in parsed.unresolved_mention_ids
    })
    entities_by_id = {entity.entity_id: entity for entity in parsed.entities}
    for shorter_id, longer_id in _nested_mention_pairs(mentions):
        shorter_entity_id = assignments[shorter_id]
        longer_entity_id = assignments[longer_id]
        if shorter_entity_id == longer_entity_id:
            continue
        shorter_entity = entities_by_id.get(shorter_entity_id)
        longer_entity = entities_by_id.get(longer_entity_id)
        component_to_blend = (
            shorter_entity is not None
            and longer_entity is not None
            and shorter_entity.polymer_type != "blend"
            and longer_entity.polymer_type == "blend"
        )
        if component_to_blend:
            if allowed_nested_splits is not None:
                allowed_nested_splits.append({
                    "shorter_mention_id": shorter_id,
                    "longer_mention_id": longer_id,
                    "component_entity_id": shorter_entity_id,
                    "blend_entity_id": longer_entity_id,
                })
            continue
        if preview_nested_splits is not None:
            preview_nested_splits.append({
                "shorter_mention_id": shorter_id,
                "longer_mention_id": longer_id,
                "shorter_assignment": shorter_entity_id,
                "longer_assignment": longer_entity_id,
            })
            continue
        if shorter_entity_id != longer_entity_id:
            raise ValueError(
                f"嵌套 mention {shorter_id}/{longer_id} 不得拆分到不同实体"
            )
    if dropped_confidence_fields is not None:
        dropped_confidence_fields.extend(dropped)
    return parsed


_CODE_LIKE_POLYMER_NAME_RE = re.compile(
    r"^[A-Za-z]{1,10}(?:[-_ ]?[A-Za-z]{0,4})?[-_ ]?\d+"
    r"(?:[-_][A-Za-z0-9]+)*$"
)
_SIMPLE_ABBREVIATION_RE = re.compile(r"^(?=.{3,6}$)(?=.*[A-Z])[A-Za-z]+$")
_SIMPLE_NUMBERED_LABEL_RE = re.compile(r"^\d+[A-Za-z]$")
_BLEND_CODE_RE = re.compile(r"^[A-Z][A-Z0-9-]*(?:/[A-Z][A-Z0-9-]*)+$")
_MATERIAL_CLASS_TERM_RE = re.compile(
    r"\b(?:blend|blends|composite|composites)\b|(?:ゴム|樹脂|高分子)",
    re.IGNORECASE,
)
_SPECIFIC_POLYMER_NAME_CUE_RE = re.compile(
    r"\b(?:based\s+on|derived\s+from|prepared\s+from|synthesized\s+from)\b"
    r"|\b[A-Za-z0-9'-]+-based\b",
    re.IGNORECASE,
)
_POLYMER_NAME_TERM_RE = re.compile(
    r"\b(?:homo|co)?poly[a-z0-9'-]*\b|\b(?:polymer|resin|rubber)\b",
    re.IGNORECASE,
)


def _is_code_like_polymer_name(name: str) -> bool:
    value = name.strip()
    return any(pattern.fullmatch(value) for pattern in (
        _CODE_LIKE_POLYMER_NAME_RE,
        _SIMPLE_ABBREVIATION_RE,
        _SIMPLE_NUMBERED_LABEL_RE,
        _BLEND_CODE_RE,
    ))


def _candidate_merely_wraps_code(candidate_name: str, code_name: str) -> bool:
    match = re.search(
        rf"(?<![A-Za-z0-9]){re.escape(code_name)}(?![A-Za-z0-9])",
        candidate_name,
        re.IGNORECASE,
    )
    if match is None:
        return False
    return not (
        _BLEND_CODE_RE.fullmatch(code_name)
        and _MATERIAL_CLASS_TERM_RE.search(candidate_name)
    )


def _preferred_polymer_name_mention(
    candidate: Any,
    mention_map: dict[str, Any],
) -> Any | None:
    """Prefer an evidence-backed polymer name over a sample/code label.

    The LLM may legally choose ``PC-1`` as ``polymer_name`` because it is a
    resolved mention.  When the same entity also owns a real polymer-name
    mention, select that mention deterministically.  Non-code model choices
    are preserved, and no new name is synthesized.
    """
    current_name = candidate.polymer_name.strip()
    if not _is_code_like_polymer_name(current_name):
        return None

    choices: list[tuple[tuple[int, int, int], int, Any]] = []
    for order, mention_id in enumerate(candidate.resolved_from_mentions):
        mention = mention_map.get(mention_id)
        if mention is None or mention.mention_role != "polymer_name":
            continue
        name = mention.text.strip()
        if not name or _is_code_like_polymer_name(name):
            continue
        if (
            _POLYMER_NAME_TERM_RE.search(name) is None
            and _MATERIAL_CLASS_TERM_RE.search(name) is None
        ):
            continue
        if _candidate_merely_wraps_code(name, current_name):
            continue
        score = (
            1 if _SPECIFIC_POLYMER_NAME_CUE_RE.search(name) else 0,
            len(name.split()),
            len(name),
        )
        choices.append((score, -order, mention))
    if not choices:
        return None
    return max(choices, key=lambda item: (item[0], item[1]))[2]


def _materialize_entities(
    parsed: PolymerEntityResponse,
    blocks: list[Stage0Element],
    mentions: Stage1Document,
    preferred_name_repairs: list[dict[str, str]] | None = None,
) -> list[PolymerEntity]:
    block_map = {block.block_id: block for block in blocks}
    mention_map = {
        mention.mention_id: mention for mention in mentions.material_mentions
    }
    id_map = {
        entity.entity_id: f"pe{index:03d}"
        for index, entity in enumerate(parsed.entities, start=1)
    }
    entities: list[PolymerEntity] = []
    for candidate in parsed.entities:
        preferred_name_mention = _preferred_polymer_name_mention(
            candidate,
            mention_map,
        )
        polymer_name = (
            preferred_name_mention.text
            if preferred_name_mention is not None
            else candidate.polymer_name
        )
        evidence_candidate = (
            preferred_name_mention.evidence
            if preferred_name_mention is not None
            else candidate.evidence
        )
        evidence_block = block_map[evidence_candidate.block_id]
        if preferred_name_mention is not None and preferred_name_repairs is not None:
            preferred_name_repairs.append({
                "entity_id": candidate.entity_id,
                "previous_name": candidate.polymer_name,
                "polymer_name": polymer_name,
                "mention_id": preferred_name_mention.mention_id,
            })
        image_refs = []
        for image_block_id in candidate.source_image_block_ids:
            image_block = block_map[image_block_id]
            image_refs.append(SourceImageReference(
                block_id=image_block.block_id,
                page=image_block.page,
                bbox=image_block.bbox,
                image_path=image_block.image_path,
                caption=image_block.caption,
            ))
        entities.append(PolymerEntity(
            entity_id=id_map[candidate.entity_id],
            polymer_name=polymer_name,
            polymer_type=candidate.polymer_type,
            variant_of=(
                id_map[candidate.variant_of]
                if candidate.variant_of is not None
                else None
            ),
            structural_features=candidate.structural_features,
            source_names=list(dict.fromkeys(
                mention_map[mention_id].text
                for mention_id in candidate.resolved_from_mentions
            )),
            resolved_from_mentions=candidate.resolved_from_mentions,
            evidence=Evidence(
                block_id=evidence_block.block_id,
                page=evidence_block.page,
                bbox=evidence_block.bbox,
                source_type=evidence_block.type,
                source_sentence=evidence_candidate.source_sentence,
            ),
            source_image_refs=image_refs,
            confidence=candidate.confidence,
        ))
    return entities


def _cache_components(
    document: Stage0Document,
    mentions: Stage1Document,
    prompt: RenderedPrompt,
    client: LLMClient,
    *,
    preview_relaxed: bool = False,
) -> tuple[str, str, str]:
    input_hash = _sha256_json({
        "stage0": document.model_dump(mode="json"),
        "stage1": mentions.model_dump(mode="json"),
    })
    model_config_hash = _sha256_json(
        llm_config_cache_payload(client.resolved)
    )
    cache_key = _sha256_json({
        "input_hash": input_hash,
        "prompt_id": prompt.prompt_id,
        "prompt_version": prompt.version,
        "rendered_prompt_hash": prompt.sha256,
        "model_config_hash": model_config_hash,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "preview_relaxed": preview_relaxed,
    })
    return input_hash, model_config_hash, cache_key


def extract_polymer_entities(
    document: Stage0Document,
    mentions: Stage1Document,
    client: LLMClient,
    prompt: RenderedPrompt,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 60000,
    max_validation_retries: int = 1,
    max_tokens: int = 8192,
    preview_relaxed: bool = False,
) -> Stage2Document:
    history_start = len(getattr(client, "call_history", []))
    if document.document_id != mentions.document_id:
        raise Stage2Error("Stage 0 与 Stage 1 document_id 不一致")
    blocks, warnings, context_chars = select_context_blocks(
        document,
        mentions,
        input_sections=input_sections,
        max_input_chars=max_input_chars,
    )
    actual_models: list[str] = []
    dropped_confidence_fields: list[str] = []
    surface_repairs: list[dict[str, str]] = []
    allowed_nested_splits: list[dict[str, str]] = []
    preview_nested_splits: list[dict[str, str]] = []
    preview_evidence_fallbacks: list[dict[str, str]] = []
    preview_invalid_entities_removed: list[dict[str, Any]] = []
    preview_duplicate_mention_repairs: list[dict[str, Any]] = []
    preferred_name_repairs: list[dict[str, str]] = []

    if mentions.material_mentions:
        feedback = None
        last_error: Exception | None = None
        parsed: PolymerEntityResponse | None = None
        for attempt in range(max_validation_retries + 1):
            try:
                response = client.call_json(
                    prompt.text,
                    _user_message(
                        document.document_id,
                        mentions,
                        blocks,
                        feedback,
                    ),
                    max_tokens=max_tokens,
                )
                parsed = _validate_response(
                    response,
                    mentions,
                    blocks,
                    dropped_confidence_fields,
                    surface_repairs,
                    allowed_nested_splits,
                    preview_duplicate_mention_repairs=(
                        preview_duplicate_mention_repairs
                        if preview_relaxed else None
                    ),
                    preview_nested_splits=(
                        preview_nested_splits if preview_relaxed else None
                    ),
                    preview_evidence_fallbacks=(
                        preview_evidence_fallbacks
                        if preview_relaxed else None
                    ),
                    preview_invalid_entities_removed=(
                        preview_invalid_entities_removed
                        if preview_relaxed else None
                    ),
                )
                actual_models.append(response.model)
                last_error = None
                break
            except (LLMRequestError, ValidationError, ValueError) as exc:
                last_error = exc
                feedback = _validation_feedback(exc)
                if attempt >= max_validation_retries:
                    break
        if last_error is not None or parsed is None:
            raise Stage2Error(
                f"{document.document_id} 响应校验失败："
                f"{_validation_feedback(last_error or ValueError('empty'))}"
            ) from last_error
    else:
        parsed = PolymerEntityResponse()

    entities = _materialize_entities(
        parsed,
        blocks,
        mentions,
        preferred_name_repairs,
    )
    if preferred_name_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "polymer_name_preferred_over_sample_label",
            "message": (
                "实体同时包含具体聚合物名称和样品代号时，已优先采用"
                "有原文 mention 支持的具体名称"
            ),
            "items": preferred_name_repairs,
        })
    if isinstance(client, _FailureReplayClient):
        warnings.append({
            "stage": STAGE_ID,
            "code": "failure_response_replayed",
            "message": "已离线回放保存的 Stage 2 响应，未请求模型",
            "source": client.failure_path.name,
        })
    if dropped_confidence_fields:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_fields_compacted",
            "message": "confidence 已确定性收敛为仅保留 score",
            "fields": list(dict.fromkeys(dropped_confidence_fields)),
        })
    if surface_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "evidence_surface_whitespace_recovered",
            "message": "仅按空白等价在同一 block 内恢复了原文 evidence 表面文本",
            "items": surface_repairs,
        })
    if allowed_nested_splits:
        warnings.append({
            "stage": STAGE_ID,
            "code": "component_blend_nested_mentions_split",
            "message": "组分名称与包含该名称的 blend 实体已保留为不同实体",
            "items": allowed_nested_splits,
        })
    if preview_nested_splits:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_nested_mentions_split_retained",
            "message": (
                "Preview 模式已保留无法确定合并关系的嵌套 mention 拆分；"
                "需在正式数据中复核"
            ),
            "items": preview_nested_splits,
        })
    if preview_evidence_fallbacks:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_entity_evidence_inherited_from_mention",
            "message": (
                "Preview 模式已从同名、同 block 的 Stage 1 mention "
                "继承逐字 evidence sentence"
            ),
            "items": preview_evidence_fallbacks,
        })
    if preview_invalid_entities_removed:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_invalid_entities_removed",
            "message": (
                "Preview 模式已移除 polymer_name 并非来自 resolved mention "
                "的实体，并将相关 mention 保守标记为 unresolved"
            ),
            "items": preview_invalid_entities_removed,
        })
    reported_unresolved = set(
        response.data.get("unresolved_mention_ids", [])
        if mentions.material_mentions
        and isinstance(response.data.get("unresolved_mention_ids"), list)
        else []
    )
    auto_unresolved = sorted(
        set(parsed.unresolved_mention_ids) - reported_unresolved
    )
    if auto_unresolved:
        warnings.append({
            "stage": STAGE_ID,
            "code": "missing_mentions_marked_unresolved",
            "message": "模型漏覆盖的 mention 已保守标记为 unresolved",
            "mention_ids": auto_unresolved,
        })
    if preview_duplicate_mention_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_duplicate_mention_recovered",
            "message": (
                "Preview 模式仅在名称与 evidence block 唯一匹配时保留归属；"
                "无法唯一归属的重复 mention 已标记为 unresolved"
            ),
            "items": preview_duplicate_mention_repairs,
        })
    if parsed.unresolved_mention_ids:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unresolved_mentions",
            "message": (
                f"{len(parsed.unresolved_mention_ids)} 个 mention 未解析为实体，"
                "已原样保留引用"
            ),
            "mention_ids": parsed.unresolved_mention_ids,
        })

    input_hash, model_config_hash, cache_key = _cache_components(
        document,
        mentions,
        prompt,
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
    provenance = Stage2Provenance(
        provider=client.resolved.provider,
        model=unique_models[-1],
        models=unique_models,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
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
    return Stage2Document(
        document_id=document.document_id,
        polymer_entities=entities,
        unresolved_mention_ids=parsed.unresolved_mention_ids,
        provenance=provenance,
        warnings=warnings,
    )


def run_stage2(
    stage0_path: Path,
    stage1_path: Path,
    output_path: Path,
    client: LLMClient,
    prompt: RenderedPrompt,
    *,
    force: bool = False,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 60000,
    max_validation_retries: int = 1,
    max_tokens: int = 8192,
    preview_relaxed: bool = False,
) -> tuple[Path, bool]:
    document = load_stage0_document(stage0_path)
    mentions = load_stage1_document(stage1_path)
    _, _, expected_cache_key = _cache_components(
        document,
        mentions,
        prompt,
        client,
        preview_relaxed=preview_relaxed,
    )
    if output_path.is_file() and not force:
        try:
            cached = Stage2Document.model_validate_json(
                output_path.read_text(encoding="utf-8-sig")
            )
            if cached.provenance.cache_key == expected_cache_key:
                return output_path, True
        except (OSError, ValidationError):
            pass

    result = extract_polymer_entities(
        document,
        mentions,
        client,
        prompt,
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
        raise Stage2Error(f"配置 {STAGE_ID} 必须是对象")
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 Stage 2 PolymerEntity 构建")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ref-no")
    mode.add_argument("--batch", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-input-chars", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--preview-relaxed",
        action="store_true",
        help="演示模式：保留有 warning 的可解析实体结果",
    )
    parser.add_argument(
        "--replay-failure",
        action="store_true",
        help="离线回放现有 stage2_failure.json，不请求模型",
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
    prompt_id = str(
        stage_config.get("prompt_id") or "polymer.stage2.polymer_entity"
    )
    prompt = PromptLoader().render_stage_prompt(
        prompt_id,
        PolymerEntityResponse,
        expected_stage=STAGE_ID,
        expected_output_schema=OUTPUT_SCHEMA_VERSION,
    )
    if args.replay_failure and not args.ref_no:
        raise Stage2Error("--replay-failure 必须与单篇 --ref-no 配合使用")
    client = (
        _failure_replay_client(
            input_root / args.ref_no / "stage2_failure.json",
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
        or 60000
    )
    max_validation_retries = (
        0
        if args.replay_failure
        else int(stage_config.get("max_validation_retries", 1))
    )
    max_tokens = int(stage_config.get("max_tokens") or 8192)

    if args.ref_no:
        ref_nos = [args.ref_no]
    else:
        ref_nos = sorted(
            path.parent.name
            for path in input_root.glob("reference_no_*/stage1_mentions.json")
        )
    if not ref_nos:
        raise Stage2Error(f"未找到 Stage 1 输出：{input_root}")

    failures: list[tuple[str, str]] = []
    for ref_no in ref_nos:
        history_start = len(client.call_history)
        try:
            output_path, cached = run_stage2(
                input_root / ref_no / "stage0_blocks.json",
                input_root / ref_no / "stage1_mentions.json",
                output_root / ref_no / "stage2_entities.json",
                client,
                prompt,
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
                    output_root / ref_no / "stage2_failure.json",
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
    print(f"Stage 2 完成：成功 {len(ref_nos) - len(failures)}，失败 {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
