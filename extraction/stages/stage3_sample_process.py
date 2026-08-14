"""Stage 3：使用 LLM 抽取 Sample 与 ProcessStep DAG。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
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
    ProcessStep,
    Sample,
    SampleProcessResponse,
    Stage0Document,
    Stage0Element,
    Stage2Document,
    Stage3Document,
    Stage3Provenance,
)


STAGE_ID = "stage3_sample_process"
OUTPUT_SCHEMA_VERSION = "sample_process_schema.v4"
IMPLEMENTATION_VERSION = "1.7.0"
# 类型推断与材料加工链继承已变化，旧缓存不能复用。
COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS: tuple[str, ...] = ()
DEFAULT_INPUT_SECTIONS = ("Methods",)
INITIAL_SAMPLE_KINDS = {"synthesis_batch", "commercial_batch"}
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？]\s+|\n+")
HTML_CHARACTER_REFERENCE_RE = re.compile(
    r"&(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);",
    flags=re.IGNORECASE,
)
COMPOSITE_EVIDENCE_RE = re.compile(
    r"\b(?:reinforced|reinforcement|filler|filled)\b|"
    r"\bcarbon\s+(?:fibers?|fibres?|black)\b|\bcCF\b|"
    r"\b\d+(?:\.\d+)?\s*(?:wt|vol)\s*%\s*CB\b|\b[A-Z]+/CB\d*\b",
    re.IGNORECASE,
)
COMPOUND_EVIDENCE_RE = re.compile(
    r"\b(?:dopant|doped|doping|plasticizer|plasticised|plasticized|additive|"
    r"electrolyte(?:\s+salt)?|masterbatch)\b|"
    r"\bLiClO\s*4\b|\blithium\s+perchlorate\b",
    re.IGNORECASE,
)
AMBIGUOUS_COMPOSITION_RE = re.compile(
    r"\b(?:blend(?:ed|s)?|mixture|compounded|formulation)\b",
    re.IGNORECASE,
)
COMPOSITION_PRESERVING_PROCESS_TYPES = {
    "casting",
    "film_formation",
    "extrusion",
    "molding",
    "pressing",
    "annealing",
    "hydration",
    "drying",
    "hot_pressing",
    "electrospinning",
    "specimen_preparation",
    "cutting",
    "punching",
}
COMPOSITION_CHANGING_PROCESS_TYPES = {"blending", "compounding", "mixing"}


class Stage3Error(RuntimeError):
    """Stage 3 输入、LLM 响应或输出验证失败。"""


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
            raise Stage3Error("failure 响应只允许离线回放一次")
        self.calls += 1
        self.call_history.append(self.record)
        return self.response


def _failure_replay_client(
    failure_path: Path,
    config: dict[str, Any],
) -> _FailureReplayClient:
    if not failure_path.is_file():
        raise Stage3Error(f"缺少 Stage 3 failure 文件：{failure_path}")
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3Error(f"Stage 3 failure 文件无效：{failure_path}") from exc
    raw = failure.get("raw_response") if isinstance(failure, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
        raise Stage3Error("Stage 3 failure 未保存可回放的 raw response")
    try:
        data = extract_json_object(raw["content"])
    except LLMRequestError as exc:
        raise Stage3Error(
            f"Stage 3 failure raw response 无法解析为 JSON 对象：{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise Stage3Error("Stage 3 failure raw response 必须是 JSON 对象")

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


def _stage3_output_payload(result: Stage3Document) -> dict[str, Any]:
    payload = result.model_dump(mode="json", exclude_none=True)
    for sample in payload.get("samples", []):
        sample.setdefault("polymer_type", None)
        sample.setdefault("copolymer_type", None)
        sample.setdefault("material_type", None)
    return payload


def _load_model(path: Path, model: type[Any], label: str) -> Any:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        cleaned, _ = compact_confidence_payload(raw)
        return model.model_validate(cleaned)
    except OSError as exc:
        raise Stage3Error(f"无法读取 {label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage3Error(f"{label} JSON 无效：{path}") from exc
    except ValidationError as exc:
        raise Stage3Error(f"{label} 未通过 Schema：{path.name}") from exc


def load_stage0_document(path: Path) -> Stage0Document:
    return _load_model(path, Stage0Document, "Stage 0")


def load_stage2_document(path: Path) -> Stage2Document:
    return _load_model(path, Stage2Document, "Stage 2")


def _element_source_text(element: Stage0Element) -> str:
    if element.type in {"text", "title", "equation", "footnote"}:
        return (element.text or "").strip()
    if element.type == "table":
        return (element.table_body or element.caption or "").strip()
    if element.type == "image":
        return (element.caption or "").strip()
    return ""


def select_context_blocks(
    document: Stage0Document,
    entities: Stage2Document,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 50000,
) -> tuple[list[Stage0Element], list[dict[str, Any]], int]:
    if max_input_chars < 2000:
        raise ValueError("max_input_chars 不得小于 2000")
    element_map = {element.block_id: element for element in document.elements}
    referenced_ids = {
        entity.evidence.block_id
        for entity in entities.polymer_entities
    } | {
        image.block_id
        for entity in entities.polymer_entities
        for image in entity.source_image_refs
    }
    missing = sorted(referenced_ids - set(element_map))
    if missing:
        raise Stage3Error(f"Stage 2 引用了未知 block：{missing}")

    section_ids = {
        element.block_id
        for element in document.elements
        if element.section in input_sections
        and bool(_element_source_text(element) or element.image_path)
        and element.type in {
            "text",
            "title",
            "table",
            "image",
            "equation",
            "footnote",
        }
    }
    selected_ids = section_ids | referenced_ids
    blocks = [
        element
        for element in document.elements
        if element.block_id in selected_ids
    ]
    context_chars = sum(
        len(_element_source_text(element)) + 200 for element in blocks
    )
    if context_chars > max_input_chars:
        raise Stage3Error(
            f"{document.document_id} Stage 3 上下文 {context_chars} 字符，"
            f"超过 max_input_chars={max_input_chars}"
        )

    warnings: list[dict[str, Any]] = []
    if not section_ids and entities.polymer_entities:
        warnings.append({
            "stage": STAGE_ID,
            "code": "section_fallback",
            "message": (
                "Methods 为空，仅使用 Stage 2 entity 的 evidence block；"
                "结果需人工复核"
            ),
        })
    return blocks, warnings, context_chars


def _user_message(
    document_id: str,
    entities: Stage2Document,
    blocks: list[Stage0Element],
    validation_feedback: str | None = None,
) -> str:
    entity_data = [
        {
            "entity_id": entity.entity_id,
            "polymer_name": entity.polymer_name,
            "polymer_type": entity.polymer_type,
            "copolymer_type": entity.copolymer_type,
            "variant_of": entity.variant_of,
            "structural_features": entity.structural_features,
            "source_names": entity.source_names,
            "resolved_from_mentions": entity.resolved_from_mentions,
            "evidence": {
                "block_id": entity.evidence.block_id,
                "source_sentence": entity.evidence.source_sentence,
            },
        }
        for entity in entities.polymer_entities
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
        "--- BEGIN UNTRUSTED POLYMER ENTITIES ---\n"
        + json.dumps(entity_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED POLYMER ENTITIES ---\n"
        "--- BEGIN UNTRUSTED METHODS BLOCKS ---\n"
        + json.dumps(block_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED METHODS BLOCKS ---"
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


def _resolve_html_entity_surface(source: str, candidate: str) -> str | None:
    """仅将 source 中的 HTML character reference 解码后匹配。

    返回值始终是 source 中的原始编码片段，不写入解码后文本。
    """
    decoded: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for match in HTML_CHARACTER_REFERENCE_RE.finditer(source):
        for index in range(position, match.start()):
            decoded.append(source[index])
            spans.append((index, index + 1))
        replacement = html.unescape(match.group(0))
        if replacement == match.group(0):
            replacement = ""
        for char in replacement:
            decoded.append(char)
            spans.append((match.start(), match.end()))
        position = match.end()
    for index in range(position, len(source)):
        decoded.append(source[index])
        spans.append((index, index + 1))
    decoded_source = "".join(decoded)
    match = re.search(re.escape(candidate), decoded_source, flags=re.IGNORECASE)
    if match is None or match.start() == match.end():
        return None
    return source[spans[match.start()][0]:spans[match.end() - 1][1]]


def _resolve_surface_text(
    source: str,
    candidate: str,
    *,
    allow_html_entities: bool = False,
) -> str | None:
    if candidate in source:
        return candidate
    direct = re.search(re.escape(candidate), source, flags=re.IGNORECASE)
    if direct:
        return direct.group(0)
    if allow_html_entities:
        entity_surface = _resolve_html_entity_surface(source, candidate)
        if entity_surface is not None:
            return entity_surface
    latex_surface = _resolve_latex_group_surface(source, candidate)
    if latex_surface is not None:
        return latex_surface
    tokens = candidate.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    pattern = pattern.replace(r"\-", "[-‐‑‒–—]")
    tolerant = re.search(pattern, source, flags=re.IGNORECASE)
    return tolerant.group(0) if tolerant else None


def _formula_key(value: str) -> str:
    without_commands = re.sub(r"\\[A-Za-z]+", "", html.unescape(value))
    return "".join(
        character.casefold()
        for character in without_commands
        if character.isalnum()
    )


def _resolve_latex_group_surface(source: str, candidate: str) -> str | None:
    candidate_key = _formula_key(candidate)
    if len(candidate_key) < 2:
        return None
    matches = [
        match.group(0)
        for match in re.finditer(
            r"\\(?:mathrm|mathbf|mathsf)\s*\{[^{}]*\}",
            source,
        )
        if _formula_key(match.group(0)) == candidate_key
    ]
    return matches[0] if len(matches) == 1 else None


def _source_sentence(text: str, anchor: str, max_chars: int = 800) -> str:
    position = text.find(anchor)
    if position < 0:
        raise ValueError("anchor 不在 evidence block 中")
    start = 0
    end = len(text)
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        if match.end() <= position:
            start = match.end()
        elif match.start() >= position + len(anchor):
            end = match.start() + 1
            break
    sentence = text[start:end].strip()
    if len(sentence) <= max_chars:
        return sentence
    anchor_position = sentence.find(anchor)
    left = max(0, anchor_position - max_chars // 2)
    right = min(len(sentence), left + max_chars)
    left = max(0, right - max_chars)
    return sentence[left:right].strip()


def _drop_unknown_confidence_fields(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    cleaned, dropped = compact_confidence_payload(data)
    return cleaned, dropped, []


def _repair_preview_evidence_key_typos(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cleaned = copy.deepcopy(data)
    repairs: list[dict[str, Any]] = []
    for collection_name in ("samples", "process_steps"):
        items = cleaned.get(collection_name)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence")
            if not isinstance(evidence, dict) or "source_sentence:" not in evidence:
                continue
            typo_value = evidence.pop("source_sentence:")
            if "source_sentence" not in evidence and isinstance(typo_value, str):
                evidence["source_sentence"] = typo_value
            repairs.append({
                "collection": collection_name,
                "index": index,
                "field": "evidence.source_sentence",
                "pattern": "trailing_colon_key_removed",
            })
    return cleaned, repairs


def _normalize_preview_process_types(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preview 下规范化已核实的工艺修饰词，其他未知类型仍交给 Schema 拒绝。"""
    cleaned = copy.deepcopy(data)
    repairs: list[dict[str, Any]] = []
    steps = cleaned.get("process_steps")
    if not isinstance(steps, list):
        return cleaned, repairs
    for step in steps:
        if not isinstance(step, dict):
            continue
        process_type = step.get("process_type")
        if (
            not isinstance(process_type, str)
            or process_type.strip().casefold() != "oxidative polymerization"
        ):
            continue
        step["process_type"] = "polymerization"
        repairs.append({
            "pattern": "process_type_normalized",
            "step_id": step.get("step_id"),
            "original_value": process_type,
            "resolved_value": "polymerization",
        })
    return cleaned, repairs


def _remove_preview_ambiguous_producer_outputs(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preview 下隔离无法唯一归属的多 producer Sample 关联。"""
    cleaned = copy.deepcopy(data)
    steps = cleaned.get("process_steps")
    if not isinstance(steps, list):
        return cleaned, []

    producers: dict[str, list[str]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id")
        outputs = step.get("output_sample_ids")
        if not isinstance(step_id, str) or not isinstance(outputs, list):
            continue
        for sample_id in dict.fromkeys(outputs):
            if isinstance(sample_id, str):
                producers.setdefault(sample_id, []).append(step_id)

    conflicts = {
        sample_id: step_ids
        for sample_id, step_ids in producers.items()
        if len(step_ids) > 1
    }
    if not conflicts:
        return cleaned, []

    retained_steps: list[Any] = []
    modified_step_ids: list[str] = []
    dropped_step_ids: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            retained_steps.append(step)
            continue
        outputs = step.get("output_sample_ids")
        if not isinstance(outputs, list):
            retained_steps.append(step)
            continue
        remaining = [
            sample_id for sample_id in outputs
            if sample_id not in conflicts
        ]
        if remaining == outputs:
            retained_steps.append(step)
            continue
        step_id = str(step.get("step_id") or "")
        if remaining:
            step["output_sample_ids"] = remaining
            modified_step_ids.append(step_id)
            retained_steps.append(step)
        else:
            dropped_step_ids.append(step_id)
    cleaned["process_steps"] = retained_steps
    return cleaned, [{
        "pattern": "ambiguous_multiple_producers_removed",
        "sample_ids": sorted(conflicts),
        "producer_step_ids": conflicts,
        "modified_step_ids": modified_step_ids,
        "dropped_step_ids": dropped_step_ids,
    }]


def _remove_process_input_output_overlap(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """仅从 ProcessStep 输入中删除与输出重叠的 Sample。"""
    cleaned = copy.deepcopy(data)
    repairs: list[dict[str, Any]] = []
    steps = cleaned.get("process_steps")
    if not isinstance(steps, list):
        return cleaned, repairs
    for step in steps:
        if not isinstance(step, dict):
            continue
        inputs = step.get("input_sample_ids")
        outputs = step.get("output_sample_ids")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            continue
        duplicate_inputs = [
            sample_id
            for index, sample_id in enumerate(inputs)
            if sample_id in inputs[:index]
        ]
        duplicate_outputs = [
            sample_id
            for index, sample_id in enumerate(outputs)
            if sample_id in outputs[:index]
        ]
        if duplicate_inputs:
            inputs = list(dict.fromkeys(inputs))
            step["input_sample_ids"] = inputs
        if duplicate_outputs:
            outputs = list(dict.fromkeys(outputs))
            step["output_sample_ids"] = outputs
        output_ids = {
            sample_id for sample_id in outputs if isinstance(sample_id, str)
        }
        removed = [
            sample_id
            for sample_id in inputs
            if isinstance(sample_id, str) and sample_id in output_ids
        ]
        if not removed and not duplicate_inputs and not duplicate_outputs:
            continue
        step["input_sample_ids"] = [
            sample_id for sample_id in inputs if sample_id not in output_ids
        ]
        repair: dict[str, Any] = {
            "step_id": str(step.get("step_id") or ""),
            "removed_input_sample_ids": list(dict.fromkeys(removed)),
        }
        if duplicate_inputs:
            repair["duplicate_input_sample_ids"] = list(dict.fromkeys(
                duplicate_inputs
            ))
        if duplicate_outputs:
            repair["duplicate_output_sample_ids"] = list(dict.fromkeys(
                duplicate_outputs
            ))
        repairs.append(repair)
    return cleaned, repairs


def _split_consecutive_extraction_drying_outputs(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """为明确的连续工艺插入一一对应的中间 Sample。"""
    cleaned = copy.deepcopy(data)
    samples = cleaned.get("samples")
    steps = cleaned.get("process_steps")
    if not isinstance(samples, list) or not isinstance(steps, list):
        return cleaned, []
    sample_map = {
        sample.get("sample_id"): sample
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("sample_id"), str)
    }
    numeric_ids = [
        int(match.group(1))
        for sample_id in sample_map
        if (match := re.fullmatch(r"s(\d+)", sample_id)) is not None
    ]
    next_id = max(numeric_ids, default=0) + 1
    repairs: list[dict[str, Any]] = []
    for previous, current in zip(steps, steps[1:]):
        if not isinstance(previous, dict) or not isinstance(current, dict):
            continue
        if current.get("process_type") not in {"drying", "pressing"}:
            continue
        previous_inputs = previous.get("input_sample_ids")
        previous_outputs = previous.get("output_sample_ids")
        current_inputs = current.get("input_sample_ids")
        current_outputs = current.get("output_sample_ids")
        if not all(isinstance(value, list) for value in (
            previous_inputs,
            previous_outputs,
            current_inputs,
            current_outputs,
        )):
            continue
        extraction_then_drying = (
            previous.get("process_type") == "solvent_extraction"
            and bool(previous_outputs)
            and previous_outputs == current_inputs == current_outputs
            and len(previous_inputs) == len(previous_outputs)
        )
        previous_evidence = previous.get("evidence")
        current_evidence = current.get("evidence")
        casting_then_drying = (
            previous.get("process_type") == "casting"
            and bool(previous_outputs)
            and previous_inputs == current_inputs
            and previous_outputs == current_outputs
            and len(previous_outputs) == len(current_inputs)
            and isinstance(previous_evidence, dict)
            and isinstance(current_evidence, dict)
            and previous_evidence.get("block_id")
            == current_evidence.get("block_id")
            and previous_evidence.get("source_sentence")
            == current_evidence.get("source_sentence")
        )
        casting_then_pressing = (
            previous.get("process_type") == "casting"
            and current.get("process_type") == "pressing"
            and bool(previous_outputs)
            and previous_inputs == current_inputs
            and previous_outputs == current_outputs
        )
        if not any((
            extraction_then_drying,
            casting_then_drying,
            casting_then_pressing,
        )):
            continue
        paired_inputs = (
            previous_inputs if extraction_then_drying else previous_outputs
        )
        if any(
            input_id not in sample_map or output_id not in sample_map
            for input_id, output_id in zip(
                paired_inputs,
                previous_outputs,
                strict=True,
            )
        ):
            continue
        entity_pairs = [
            (
                sample_map[input_id].get("refers_to_entity"),
                sample_map[output_id].get("refers_to_entity"),
            )
            for input_id, output_id in zip(
                paired_inputs,
                previous_outputs,
                strict=True,
            )
        ]
        if any(
            not isinstance(input_entity, str)
            or not isinstance(output_entity, str)
            or input_entity != output_entity
            for input_entity, output_entity in entity_pairs
        ):
            continue
        evidence = previous_evidence
        sentence = (
            evidence.get("source_sentence")
            if isinstance(evidence, dict)
            else None
        )
        confidence = previous.get("confidence")
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        if not isinstance(confidence, dict):
            continue
        intermediate_ids: list[str] = []
        for output_id in previous_outputs:
            final_sample = sample_map[output_id]
            intermediate_id = f"s{next_id:03d}"
            next_id += 1
            intermediate = {
                "sample_id": intermediate_id,
                "sample_kind": "intermediate",
                "refers_to_entity": final_sample.get("refers_to_entity"),
                "sample_label_raw": None,
                "state_description": sentence,
                "intended_use": [],
                "evidence": copy.deepcopy(evidence),
                "confidence": copy.deepcopy(confidence),
            }
            samples.append(intermediate)
            sample_map[intermediate_id] = intermediate
            intermediate_ids.append(intermediate_id)
        previous["output_sample_ids"] = intermediate_ids
        current["input_sample_ids"] = intermediate_ids
        repairs.append({
            "pattern": (
                "extraction_then_drying"
                if extraction_then_drying
                else (
                    "casting_then_drying"
                    if casting_then_drying
                    else "casting_then_pressing"
                )
            ),
            "previous_step_id": str(previous.get("step_id") or ""),
            "current_step_id": str(current.get("step_id") or ""),
            "intermediate_sample_ids": intermediate_ids,
        })
    return cleaned, repairs


def _split_preview_in_place_postprocess_outputs(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preview 下为原位后处理创建一一对应的最终 Sample。"""
    cleaned = copy.deepcopy(data)
    samples = cleaned.get("samples")
    steps = cleaned.get("process_steps")
    if not isinstance(samples, list) or not isinstance(steps, list):
        return cleaned, []
    sample_map = {
        sample.get("sample_id"): sample
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("sample_id"), str)
    }
    next_id = max((
        int(match.group(1))
        for sample_id in sample_map
        if (match := re.fullmatch(r"s(\d+)", sample_id)) is not None
    ), default=0) + 1
    produced: set[str] = set()
    repairs: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        inputs = step.get("input_sample_ids")
        outputs = step.get("output_sample_ids")
        evidence = step.get("evidence")
        confidence = step.get("confidence")
        can_split = (
            step.get("process_type") == "drying"
            and isinstance(inputs, list)
            and isinstance(outputs, list)
            and bool(outputs)
            and inputs == outputs
            and len(outputs) == len(set(outputs))
            and set(outputs).issubset(produced)
            and all(sample_id in sample_map for sample_id in outputs)
            and isinstance(evidence, dict)
            and isinstance(evidence.get("source_sentence"), str)
            and bool(evidence.get("source_sentence", "").strip())
            and isinstance(confidence, dict)
        )
        if can_split:
            final_ids: list[str] = []
            for input_id in inputs:
                source_sample = sample_map[input_id]
                final_id = f"s{next_id:03d}"
                next_id += 1
                final_sample = {
                    "sample_id": final_id,
                    "sample_kind": "processed_material",
                    "refers_to_entity": source_sample.get("refers_to_entity"),
                    "sample_label_raw": None,
                    "state_description": evidence["source_sentence"],
                    "intended_use": [],
                    "evidence": copy.deepcopy(evidence),
                    "confidence": copy.deepcopy(confidence),
                }
                samples.append(final_sample)
                sample_map[final_id] = final_sample
                final_ids.append(final_id)
            step["output_sample_ids"] = final_ids
            repairs.append({
                "step_id": str(step.get("step_id") or ""),
                "input_sample_ids": list(inputs),
                "final_sample_ids": final_ids,
            })
            outputs = final_ids
        if isinstance(outputs, list):
            produced.update(
                sample_id for sample_id in outputs
                if isinstance(sample_id, str)
            )
    return cleaned, repairs


def _split_misbound_hot_pressing_drying_outputs(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """拆分误绑到另一 evidence 路线的 hot_pressing→drying 输出。"""

    cleaned = copy.deepcopy(data)
    samples = cleaned.get("samples")
    steps = cleaned.get("process_steps")
    if not isinstance(samples, list) or not isinstance(steps, list):
        return cleaned, []
    sample_map = {
        sample.get("sample_id"): sample
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("sample_id"), str)
    }
    next_id = max((
        int(match.group(1))
        for sample_id in sample_map
        if (match := re.fullmatch(r"s(\d+)", sample_id)) is not None
    ), default=0) + 1
    repairs = []
    for previous, current in zip(steps, steps[1:]):
        if (
            not isinstance(previous, dict)
            or not isinstance(current, dict)
            or previous.get("process_type") != "hot_pressing"
            or current.get("process_type") != "drying"
        ):
            continue
        previous_inputs = previous.get("input_sample_ids")
        shared = previous.get("output_sample_ids")
        if (
            not isinstance(previous_inputs, list)
            or len(previous_inputs) != 1
            or not isinstance(shared, list)
            or len(shared) != 1
            or current.get("input_sample_ids") != shared
            or current.get("output_sample_ids") != shared
        ):
            continue
        input_sample = sample_map.get(previous_inputs[0])
        wrong_sample = sample_map.get(shared[0])
        previous_evidence = previous.get("evidence")
        current_evidence = current.get("evidence")
        wrong_evidence = (
            wrong_sample.get("evidence")
            if isinstance(wrong_sample, dict) else None
        )
        if not all(isinstance(value, dict) for value in (
            input_sample,
            wrong_sample,
            previous_evidence,
            current_evidence,
            wrong_evidence,
        )):
            continue
        route_block = previous_evidence.get("block_id")
        input_entity = input_sample.get("refers_to_entity")
        wrong_entity = wrong_sample.get("refers_to_entity")
        if (
            route_block != current_evidence.get("block_id")
            or route_block == wrong_evidence.get("block_id")
            or not isinstance(input_entity, str)
            or not isinstance(wrong_entity, str)
            or input_entity == wrong_entity
        ):
            continue
        new_ids = []
        for kind, evidence, confidence in (
            ("intermediate", previous_evidence, previous.get("confidence")),
            ("processed_material", current_evidence, current.get("confidence")),
        ):
            if not isinstance(confidence, dict):
                break
            sample_id = f"s{next_id:03d}"
            next_id += 1
            samples.append({
                "sample_id": sample_id,
                "sample_kind": kind,
                "sample_label_raw": None,
                "state_description": evidence.get("source_sentence"),
                "refers_to_entity": input_entity,
                "intended_use": [],
                "evidence": copy.deepcopy(evidence),
                "confidence": copy.deepcopy(confidence),
            })
            new_ids.append(sample_id)
        if len(new_ids) != 2:
            continue
        previous["output_sample_ids"] = [new_ids[0]]
        current["input_sample_ids"] = [new_ids[0]]
        current["output_sample_ids"] = [new_ids[1]]
        repairs.append({
            "pattern": "misbound_hot_pressing_then_drying",
            "previous_step_id": previous.get("step_id"),
            "current_step_id": current.get("step_id"),
            "intermediate_sample_ids": [new_ids[0]],
            "final_sample_ids": [new_ids[1]],
        })
    return cleaned, repairs



def _remove_preview_duplicate_upstream_fraction_outputs(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preview 下移除明确 fraction 的上游重复 producer。"""
    cleaned = copy.deepcopy(data)
    samples = cleaned.get("samples")
    steps = cleaned.get("process_steps")
    if not isinstance(samples, list) or not isinstance(steps, list):
        return cleaned, []

    sample_map = {
        sample.get("sample_id"): sample
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("sample_id"), str)
    }
    producers: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        outputs = step.get("output_sample_ids")
        if not isinstance(outputs, list):
            continue
        for sample_id in outputs:
            if isinstance(sample_id, str):
                producers.setdefault(sample_id, []).append(step)

    removals_by_step: dict[int, list[str]] = {}
    repairs_by_pair: dict[tuple[str, str], list[str]] = {}
    for sample_id, producer_steps in producers.items():
        if len(producer_steps) != 2:
            continue
        polymerization_steps = [
            step for step in producer_steps
            if step.get("process_type") == "polymerization"
        ]
        fractionation_steps = [
            step for step in producer_steps
            if step.get("process_type") == "fractionation"
        ]
        if len(polymerization_steps) != 1 or len(fractionation_steps) != 1:
            continue
        polymerization_step = polymerization_steps[0]
        fractionation_step = fractionation_steps[0]
        fractionation_inputs = fractionation_step.get("input_sample_ids")
        if (
            not isinstance(fractionation_inputs, list)
            or not any(
                isinstance(input_id, str) and input_id != sample_id
                for input_id in fractionation_inputs
            )
        ):
            continue
        sample = sample_map.get(sample_id)
        if not isinstance(sample, dict):
            continue
        sample_evidence = sample.get("evidence")
        fractionation_evidence = fractionation_step.get("evidence")
        anchor_text = " ".join(
            value
            for value in (
                sample.get("sample_label_raw"),
                sample.get("state_description"),
                (
                    sample_evidence.get("source_sentence")
                    if isinstance(sample_evidence, dict) else None
                ),
                (
                    fractionation_evidence.get("source_sentence")
                    if isinstance(fractionation_evidence, dict) else None
                ),
            )
            if isinstance(value, str)
        )
        if re.search(r"\b(?:soluble|fractions?)\b", anchor_text, re.IGNORECASE) is None:
            continue
        polymerization_outputs = polymerization_step.get("output_sample_ids")
        if not isinstance(polymerization_outputs, list):
            continue
        removals_by_step.setdefault(id(polymerization_step), []).append(sample_id)
        pair = (
            str(polymerization_step.get("step_id") or ""),
            str(fractionation_step.get("step_id") or ""),
        )
        repairs_by_pair.setdefault(pair, []).append(sample_id)

    skipped_sample_ids: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        sample_ids = removals_by_step.get(id(step), [])
        if not sample_ids:
            continue
        outputs = step.get("output_sample_ids")
        remaining = [
            sample_id for sample_id in outputs
            if sample_id not in set(sample_ids)
        ]
        if not remaining:
            skipped_sample_ids.update(sample_ids)
            continue
        step["output_sample_ids"] = remaining

    repairs = []
    for (polymerization_step_id, fractionation_step_id), sample_ids in repairs_by_pair.items():
        repaired_ids = [
            sample_id for sample_id in sample_ids
            if sample_id not in skipped_sample_ids
        ]
        if repaired_ids:
            repairs.append({
                "polymerization_step_id": polymerization_step_id,
                "fractionation_step_id": fractionation_step_id,
                "sample_ids": repaired_ids,
            })
    return cleaned, repairs


def _prepare_response_structure(
    response: LLMJSONResponse,
    *,
    preview_relaxed: bool = False,
) -> tuple[
    SampleProcessResponse,
    list[str],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    (
        cleaned_data,
        dropped_confidence_fields,
        moved_confidence_values,
    ) = _drop_unknown_confidence_fields(response.data)
    cleaned_data, consecutive_process_repairs = (
        _split_consecutive_extraction_drying_outputs(cleaned_data)
    )
    cleaned_data, misbound_process_repairs = (
        _split_misbound_hot_pressing_drying_outputs(cleaned_data)
    )
    consecutive_process_repairs.extend(misbound_process_repairs)
    preview_in_place_repairs: list[dict[str, Any]] = []
    preview_fraction_repairs: list[dict[str, Any]] = []
    if preview_relaxed:
        cleaned_data, evidence_key_repairs = (
            _repair_preview_evidence_key_typos(cleaned_data)
        )
        cleaned_data, process_type_repairs = (
            _normalize_preview_process_types(cleaned_data)
        )
        cleaned_data, preview_in_place_repairs = (
            _split_preview_in_place_postprocess_outputs(cleaned_data)
        )
        preview_in_place_repairs = [
            *evidence_key_repairs,
            *process_type_repairs,
            *preview_in_place_repairs,
        ]
        cleaned_data, preview_fraction_repairs = (
            _remove_preview_duplicate_upstream_fraction_outputs(cleaned_data)
        )
        cleaned_data, producer_conflict_repairs = (
            _remove_preview_ambiguous_producer_outputs(cleaned_data)
        )
        preview_in_place_repairs.extend(producer_conflict_repairs)
    cleaned_data, process_input_repairs = (
        _remove_process_input_output_overlap(cleaned_data)
    )
    return (
        SampleProcessResponse.model_validate(cleaned_data),
        dropped_confidence_fields,
        moved_confidence_values,
        consecutive_process_repairs,
        process_input_repairs,
        preview_in_place_repairs,
        preview_fraction_repairs,
    )


def _preview_semantic_validation_can_bypass(exc: ValueError) -> bool:
    message = str(exc)
    return any(
        marker in message
        for marker in (
            ".sample_label_raw 不是 evidence 原文子串",
            ".intended_use 不是 evidence 原文子串",
            ".state_description 不是 evidence 原文子串",
        )
    )


def _validate_response(
    response: LLMJSONResponse,
    entities: Stage2Document,
    blocks: list[Stage0Element],
    *,
    preview_relaxed: bool = False,
) -> tuple[
    SampleProcessResponse,
    list[tuple[str, str]],
    list[str],
    list[str],
    list[dict[str, str]],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[dict[str, str]],
    list[str],
]:
    (
        parsed,
        dropped_confidence_fields,
        moved_confidence_values,
        consecutive_process_repairs,
        process_input_repairs,
        preview_in_place_repairs,
        preview_fraction_repairs,
    ) = _prepare_response_structure(
        response,
        preview_relaxed=preview_relaxed,
    )
    entity_ids = {
        entity.entity_id for entity in entities.polymer_entities
    }
    entity_map = {
        entity.entity_id: entity for entity in entities.polymer_entities
    }
    block_map = {block.block_id: block for block in blocks}

    resolved_entity_ids: set[str] = set()
    normalized_samples = []
    evidence_relinks: list[dict[str, str]] = []
    dropped_sample_labels: list[str] = []
    dropped_intended_uses: list[dict[str, Any]] = []
    state_descriptions_replaced: list[str] = []
    state_descriptions_dropped: list[str] = []
    html_entity_label_repairs: list[dict[str, str]] = []
    for sample in parsed.samples:
        if (
            sample.refers_to_entity is not None
            and sample.refers_to_entity not in entity_ids
        ):
            raise ValueError(
                f"{sample.sample_id} 引用了未知 entity："
                f"{sample.refers_to_entity}"
            )
        if sample.refers_to_entity is not None:
            resolved_entity_ids.add(sample.refers_to_entity)
        evidence_block = block_map.get(sample.evidence.block_id)
        if evidence_block is None:
            raise ValueError(
                f"{sample.sample_id} 引用了未知 evidence block："
                f"{sample.evidence.block_id}"
            )
        source_text = _element_source_text(evidence_block)
        resolved_state = (
            _resolve_surface_text(source_text, sample.state_description)
            if sample.state_description is not None
            else None
        )
        resolved_label = (
            _resolve_surface_text(
                source_text,
                sample.sample_label_raw,
                allow_html_entities=True,
            )
            if sample.sample_label_raw is not None
            else None
        )
        linked_entity_for_label = (
            entity_map.get(sample.refers_to_entity)
            if sample.refers_to_entity is not None
            else None
        )
        linked_names_for_label = (
            [linked_entity_for_label.polymer_name]
            + linked_entity_for_label.source_names
            if linked_entity_for_label is not None
            else []
        )
        has_linked_entity_anchor = any(
            _resolve_surface_text(source_text, source_name) is not None
            for source_name in linked_names_for_label
        )
        if (
            sample.sample_label_raw is not None
            and resolved_label is not None
            and resolved_label != sample.sample_label_raw
            and html.unescape(resolved_label).casefold()
            == sample.sample_label_raw.casefold()
        ):
            html_entity_label_repairs.append({
                "sample_id": sample.sample_id,
                "field": "sample_label_raw",
                "from": sample.sample_label_raw,
                "to": resolved_label,
            })
        if sample.sample_label_raw is not None and resolved_label is None:
            matching_blocks = [
                (block, resolved)
                for block in blocks
                if (resolved := _resolve_surface_text(
                    _element_source_text(block),
                    sample.sample_label_raw,
                    allow_html_entities=True,
                )) is not None
            ]
            target_block = matching_blocks[0][0] if len(matching_blocks) == 1 else None
            target_text = (
                _element_source_text(target_block)
                if target_block is not None
                else ""
            )
            can_relink = (
                target_block is not None
                and (
                    sample.state_description is None
                    or _resolve_surface_text(
                        target_text,
                        sample.state_description,
                    ) is not None
                )
                and all(
                    _resolve_surface_text(target_text, item) is not None
                    for item in sample.intended_use
                )
            )
            if can_relink:
                previous_block_id = evidence_block.block_id
                evidence_block, resolved_label = matching_blocks[0]
                source_text = _element_source_text(evidence_block)
                resolved_state = (
                    _resolve_surface_text(
                        source_text,
                        sample.state_description,
                    )
                    if sample.state_description is not None
                    else None
                )
                evidence_relinks.append({
                    "sample_id": sample.sample_id,
                    "from_block_id": previous_block_id,
                    "to_block_id": evidence_block.block_id,
                })
            elif (
                resolved_state is not None
                or has_linked_entity_anchor
            ):
                dropped_sample_labels.append(sample.sample_id)
            elif sample.sample_kind == "intermediate":
                fallback_state = _resolve_surface_text(
                    source_text,
                    sample.evidence.source_sentence,
                )
                if fallback_state is not None:
                    dropped_sample_labels.append(sample.sample_id)
                    resolved_state = fallback_state
                    state_descriptions_replaced.append(sample.sample_id)
        if (
            sample.sample_label_raw is not None
            and resolved_label is None
            and sample.sample_id not in dropped_sample_labels
        ):
            raise ValueError(
                f"{sample.sample_id}.sample_label_raw 不是 evidence 原文子串"
            )
        resolved_uses: list[str] = []
        for intended_use in sample.intended_use:
            resolved_use = _resolve_surface_text(source_text, intended_use)
            if resolved_use is None:
                if resolved_label is not None or resolved_state is not None:
                    dropped_intended_uses.append({
                        "sample_id": sample.sample_id,
                        "values": [intended_use],
                    })
                    continue
                raise ValueError(
                    f"{sample.sample_id}.intended_use 不是 evidence 原文子串："
                    f"{intended_use!r}"
                )
            resolved_uses.append(resolved_use)
        linked_entity = (
            entity_map.get(sample.refers_to_entity)
            if sample.refers_to_entity is not None
            else None
        )
        linked_names = (
            [linked_entity.polymer_name] + linked_entity.source_names
            if linked_entity is not None
            else []
        )
        matched_linked_names = [
            resolved
            for source_name in linked_names
            if (resolved := _resolve_surface_text(
                source_text,
                source_name,
            )) is not None
        ]
        resolved_evidence = _resolve_surface_text(
            source_text,
            sample.evidence.source_sentence,
        )
        if resolved_evidence is None:
            evidence_anchor = next(
                (
                    value
                    for value in (
                        resolved_label,
                        resolved_state,
                        *resolved_uses,
                        *matched_linked_names,
                    )
                    if value is not None
                ),
                None,
            )
            if evidence_anchor is None:
                if source_text:
                    resolved_evidence = source_text
                else:
                    raise ValueError(
                        f"{sample.sample_id}.evidence block 没有可保存的原文"
                    )
            else:
                resolved_evidence = _source_sentence(
                    source_text,
                    evidence_anchor,
                )
        if sample.state_description is not None and resolved_state is None:
            if resolved_label is not None:
                state_descriptions_dropped.append(sample.sample_id)
            elif (
                sample.sample_label_raw is None
                or sample.sample_id in dropped_sample_labels
            ):
                resolved_state = resolved_evidence
                state_descriptions_replaced.append(sample.sample_id)
            else:
                raise ValueError(
                    f"{sample.sample_id}.state_description 不是 evidence 原文子串"
                )
        normalized_samples.append(sample.model_copy(update={
            "sample_label_raw": resolved_label,
            "state_description": resolved_state,
            "intended_use": resolved_uses,
            "evidence": sample.evidence.model_copy(
                update={
                    "block_id": evidence_block.block_id,
                    "source_sentence": resolved_evidence,
                }
            ),
        }))

    unresolved_ids = set(parsed.unresolved_entity_ids)
    unknown_unresolved = sorted(unresolved_ids - entity_ids)
    if unknown_unresolved:
        raise ValueError(f"存在未知 unresolved entity：{unknown_unresolved}")
    if resolved_entity_ids & unresolved_ids:
        raise ValueError("resolved 与 unresolved entity 不得重叠")
    completed_unresolved_ids = sorted(
        entity_ids - resolved_entity_ids - unresolved_ids
    )
    if completed_unresolved_ids:
        parsed = parsed.model_copy(update={
            "unresolved_entity_ids": [
                *parsed.unresolved_entity_ids,
                *completed_unresolved_ids,
            ]
        })

    normalized_steps = []
    dropped_parameters: list[tuple[str, str]] = []
    for step in parsed.process_steps:
        evidence_block = block_map.get(step.evidence.block_id)
        if evidence_block is None:
            raise ValueError(
                f"{step.step_id} 引用了未知 evidence block："
                f"{step.evidence.block_id}"
            )
        source_text = _element_source_text(evidence_block)
        normalized_parameters: dict[str, str] = {}
        for key, value in step.parameters.items():
            resolved_value = _resolve_surface_text(source_text, value)
            if resolved_value is None:
                dropped_parameters.append((step.step_id, key))
                continue
            normalized_parameters[key] = resolved_value
        resolved_evidence = _resolve_surface_text(
            source_text,
            step.evidence.source_sentence,
        )
        if resolved_evidence is None:
            if normalized_parameters:
                resolved_evidence = _source_sentence(
                    source_text,
                    next(iter(normalized_parameters.values())),
                )
            elif source_text:
                resolved_evidence = source_text
            else:
                raise ValueError(
                    f"{step.step_id}.evidence block 没有可保存的原文"
                )
        normalized_steps.append(step.model_copy(update={
            "parameters": normalized_parameters,
            "evidence": step.evidence.model_copy(
                update={"source_sentence": resolved_evidence}
            ),
        }))
    return (
        parsed.model_copy(update={
            "samples": normalized_samples,
            "process_steps": normalized_steps,
        }),
        dropped_parameters,
        dropped_confidence_fields,
        moved_confidence_values,
        evidence_relinks,
        dropped_sample_labels,
        dropped_intended_uses,
        consecutive_process_repairs,
        process_input_repairs,
        preview_in_place_repairs,
        preview_fraction_repairs,
        state_descriptions_replaced,
        state_descriptions_dropped,
        html_entity_label_repairs,
        completed_unresolved_ids,
    )


def _materialize(
    parsed: SampleProcessResponse,
    blocks: list[Stage0Element],
    entities: Stage2Document,
) -> tuple[list[Sample], list[ProcessStep], list[dict[str, Any]]]:
    block_map = {block.block_id: block for block in blocks}
    entity_map = {
        entity.entity_id: entity
        for entity in entities.polymer_entities
    }
    sample_id_map = {
        sample.sample_id: f"s{index:03d}"
        for index, sample in enumerate(parsed.samples, start=1)
    }
    step_id_map = {
        step.step_id: f"ps{index:03d}"
        for index, step in enumerate(parsed.process_steps, start=1)
    }
    samples: list[Sample] = []
    polymer_type_overrides: list[dict[str, Any]] = []
    for candidate in parsed.samples:
        block = block_map[candidate.evidence.block_id]
        linked_entity = (
            entity_map.get(candidate.refers_to_entity)
            if candidate.refers_to_entity is not None
            else None
        )
        polymer_name = (
            linked_entity.polymer_name
            if linked_entity is not None
            else candidate.sample_label_raw or candidate.state_description
        )
        if polymer_name is None:
            raise Stage3Error(
                f"{candidate.sample_id} 无法生成兼容 polymer_name"
            )
        polymer_type = candidate.polymer_type
        copolymer_type = candidate.copolymer_type
        if linked_entity is not None and linked_entity.polymer_type is not None:
            if (
                polymer_type is not None
                and polymer_type != linked_entity.polymer_type
            ):
                polymer_type_overrides.append({
                    "field": "polymer_type",
                    "sample_id": sample_id_map[candidate.sample_id],
                    "entity_id": linked_entity.entity_id,
                    "model_value": polymer_type,
                    "resolved_value": linked_entity.polymer_type,
                })
            polymer_type = linked_entity.polymer_type
            if linked_entity.polymer_type != "copolymer":
                if copolymer_type is not None:
                    polymer_type_overrides.append({
                        "field": "copolymer_type",
                        "sample_id": sample_id_map[candidate.sample_id],
                        "entity_id": linked_entity.entity_id,
                        "model_value": copolymer_type,
                        "resolved_value": None,
                    })
                copolymer_type = None
            elif linked_entity.copolymer_type is not None:
                if (
                    copolymer_type is not None
                    and copolymer_type != linked_entity.copolymer_type
                ):
                    polymer_type_overrides.append({
                        "field": "copolymer_type",
                        "sample_id": sample_id_map[candidate.sample_id],
                        "entity_id": linked_entity.entity_id,
                        "model_value": copolymer_type,
                        "resolved_value": linked_entity.copolymer_type,
                    })
                copolymer_type = linked_entity.copolymer_type
        samples.append(Sample(
            sample_id=sample_id_map[candidate.sample_id],
            sample_kind=candidate.sample_kind,
            refers_to_entity=candidate.refers_to_entity,
            polymer_name=polymer_name,
            polymer_type=polymer_type,
            copolymer_type=copolymer_type,
            material_type=candidate.material_type,
            sample_label_raw=candidate.sample_label_raw,
            state_description=candidate.state_description,
            intended_use=candidate.intended_use,
            evidence=Evidence(
                block_id=block.block_id,
                page=block.page,
                bbox=block.bbox,
                source_type=block.type,
                source_sentence=candidate.evidence.source_sentence,
            ),
            confidence=candidate.confidence,
        ))
    steps: list[ProcessStep] = []
    for candidate in parsed.process_steps:
        block = block_map[candidate.evidence.block_id]
        steps.append(ProcessStep(
            step_id=step_id_map[candidate.step_id],
            process_type=candidate.process_type,
            input_sample_ids=[
                sample_id_map[sample_id]
                for sample_id in candidate.input_sample_ids
            ],
            output_sample_ids=[
                sample_id_map[sample_id]
                for sample_id in candidate.output_sample_ids
            ],
            parameters=candidate.parameters,
            evidence=Evidence(
                block_id=block.block_id,
                page=block.page,
                bbox=block.bbox,
                source_type=block.type,
                source_sentence=candidate.evidence.source_sentence,
            ),
            confidence=candidate.confidence,
        ))
    return samples, steps, polymer_type_overrides


def _sample_material_text(sample: Sample) -> str:
    return "\n".join(
        value
        for value in (
            sample.polymer_name,
            sample.sample_label_raw,
            sample.state_description,
            sample.evidence.source_sentence,
        )
        if value
    )


def _compact_formula_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", html.unescape(value))
    without_latex_commands = re.sub(r"\\[A-Za-z]+", "", without_tags)
    return re.sub(r"[^A-Za-z0-9]+", "", without_latex_commands).lower()


def _apply_process_polymer_type_policy(
    samples: list[Sample],
    process_steps: list[ProcessStep],
) -> tuple[list[Sample], list[dict[str, Any]]]:
    """Infer blend outputs only from multiple distinct polymer inputs."""

    by_id = {sample.sample_id: sample for sample in samples}
    inferences: list[dict[str, Any]] = []
    for step in process_steps:
        if step.process_type not in COMPOSITION_CHANGING_PROCESS_TYPES:
            continue
        entity_ids = {
            sample.refers_to_entity
            for sample_id in step.input_sample_ids
            if (sample := by_id.get(sample_id)) is not None
            and sample.refers_to_entity is not None
        }
        if len(entity_ids) < 2:
            continue
        for sample_id in step.output_sample_ids:
            output = by_id.get(sample_id)
            if output is None or output.polymer_type is not None:
                continue
            by_id[sample_id] = output.model_copy(update={
                "polymer_type": "polymer_blend",
                "copolymer_type": None,
            })
            inferences.append({
                "sample_id": sample_id,
                "field": "polymer_type",
                "value": "polymer_blend",
                "step_id": step.step_id,
                "process_type": step.process_type,
                "input_entity_ids": sorted(entity_ids),
                "reason": "配方步骤包含至少两个不同的聚合物实体输入",
            })

    changed = True
    while changed:
        changed = False
        for step in process_steps:
            if step.process_type not in COMPOSITION_PRESERVING_PROCESS_TYPES:
                continue
            input_types = [
                by_id[sample_id].polymer_type
                for sample_id in step.input_sample_ids
                if sample_id in by_id
            ]
            if (
                not input_types
                or len(input_types) != len(step.input_sample_ids)
                or any(value is None for value in input_types)
                or len(set(input_types)) != 1
            ):
                continue
            inherited = input_types[0]
            for sample_id in step.output_sample_ids:
                output = by_id.get(sample_id)
                if output is None or output.polymer_type is not None:
                    continue
                by_id[sample_id] = output.model_copy(update={
                    "polymer_type": inherited,
                    "copolymer_type": None,
                })
                inferences.append({
                    "sample_id": sample_id,
                    "field": "polymer_type",
                    "value": inherited,
                    "step_id": step.step_id,
                    "process_type": step.process_type,
                    "input_sample_ids": step.input_sample_ids,
                    "reason": "成分保持工艺的全部输入具有一致 polymer_type",
                })
                changed = True
    return [by_id[sample.sample_id] for sample in samples], inferences


def _apply_material_type_policy(
    samples: list[Sample],
    process_steps: list[ProcessStep],
) -> tuple[
    list[Sample],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Infer evidence-backed types, inherit safe processes, then default neat resin."""

    by_id = {sample.sample_id: sample for sample in samples}
    evidence_inferences: list[dict[str, Any]] = []
    inheritance_items: list[dict[str, Any]] = []
    default_inferences: list[dict[str, Any]] = []

    for sample_id, sample in list(by_id.items()):
        if sample.material_type is not None:
            continue
        text = _sample_material_text(sample)
        compact_text = _compact_formula_text(text)
        inferred: str | None = None
        reason: str | None = None
        if COMPOSITE_EVIDENCE_RE.search(text):
            inferred = "composite"
            reason = "样品证据明确包含复合材料、增强体或填料"
        elif COMPOUND_EVIDENCE_RE.search(text) or "liclo4" in compact_text:
            inferred = "compound"
            reason = "样品证据明确包含掺杂剂、电解质盐、增塑剂或其他配方组分"
        if inferred is not None:
            by_id[sample_id] = sample.model_copy(update={"material_type": inferred})
            evidence_inferences.append({
                "sample_id": sample_id,
                "field": "material_type",
                "value": inferred,
                "reason": reason,
            })

    for step in process_steps:
        if step.process_type not in COMPOSITION_CHANGING_PROCESS_TYPES:
            continue
        related = [
            by_id[sample_id]
            for sample_id in (*step.input_sample_ids, *step.output_sample_ids)
            if sample_id in by_id
        ]
        process_text = "\n".join([
            step.evidence.source_sentence,
            *(_sample_material_text(sample) for sample in related),
        ])
        inferred = (
            "composite"
            if COMPOSITE_EVIDENCE_RE.search(process_text)
            else "compound"
        )
        for sample_id in step.output_sample_ids:
            output = by_id.get(sample_id)
            if output is None:
                continue
            previous = output.material_type
            should_update = previous is None or (
                previous == "composite" and inferred == "compound"
            )
            if not should_update:
                continue
            by_id[sample_id] = output.model_copy(update={"material_type": inferred})
            evidence_inferences.append({
                "sample_id": sample_id,
                "field": "material_type",
                "previous_value": previous,
                "value": inferred,
                "step_id": step.step_id,
                "process_type": step.process_type,
                "reason": (
                    "配方步骤包含明确填料或增强体"
                    if inferred == "composite"
                    else "多组分配方步骤无填料或增强体证据"
                ),
            })

    changed = True
    while changed:
        changed = False
        for step in process_steps:
            if step.process_type not in COMPOSITION_PRESERVING_PROCESS_TYPES:
                continue
            input_types = [
                by_id[sample_id].material_type
                for sample_id in step.input_sample_ids
                if sample_id in by_id
            ]
            if (
                not input_types
                or len(input_types) != len(step.input_sample_ids)
                or any(value is None for value in input_types)
                or len(set(input_types)) != 1
            ):
                continue
            inherited = input_types[0]
            for sample_id in step.output_sample_ids:
                output = by_id.get(sample_id)
                if output is None or output.material_type is not None:
                    continue
                by_id[sample_id] = output.model_copy(
                    update={"material_type": inherited}
                )
                inheritance_items.append({
                    "sample_id": sample_id,
                    "field": "material_type",
                    "value": inherited,
                    "step_id": step.step_id,
                    "process_type": step.process_type,
                    "input_sample_ids": step.input_sample_ids,
                })
                changed = True

    for sample_id, sample in list(by_id.items()):
        if sample.material_type is not None:
            continue
        text = _sample_material_text(sample)
        compact_text = _compact_formula_text(text)
        has_contrary_evidence = bool(
            sample.polymer_type not in {"homopolymer", "copolymer"}
            or COMPOSITE_EVIDENCE_RE.search(text)
            or COMPOUND_EVIDENCE_RE.search(text)
            or "liclo4" in compact_text
            or AMBIGUOUS_COMPOSITION_RE.search(text)
        )
        if has_contrary_evidence:
            continue
        by_id[sample_id] = sample.model_copy(update={"material_type": "neat_resin"})
        default_inferences.append({
            "sample_id": sample_id,
            "field": "material_type",
            "value": "neat_resin",
            "reason": "已建立单一聚合物样品，且没有第二配方组分反证",
        })

    return (
        [by_id[sample.sample_id] for sample in samples],
        evidence_inferences,
        inheritance_items,
        default_inferences,
    )


def _cache_components(
    document: Stage0Document,
    entities: Stage2Document,
    prompt: RenderedPrompt,
    client: LLMClient,
    *,
    implementation_version: str = IMPLEMENTATION_VERSION,
    preview_relaxed: bool = False,
) -> tuple[str, str, str]:
    input_hash = _sha256_json({
        "stage0": document.model_dump(mode="json"),
        "stage2": entities.model_dump(mode="json"),
    })
    model_config_hash = _sha256_json(
        llm_config_cache_payload(client.resolved)
    )
    cache_payload = {
        "input_hash": input_hash,
        "prompt_id": prompt.prompt_id,
        "prompt_version": prompt.version,
        "rendered_prompt_hash": prompt.sha256,
        "model_config_hash": model_config_hash,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation_version": implementation_version,
    }
    if preview_relaxed:
        cache_payload["preview_relaxed"] = True
    cache_key = _sha256_json(cache_payload)
    return input_hash, model_config_hash, cache_key


def extract_samples_processes(
    document: Stage0Document,
    entities: Stage2Document,
    client: LLMClient,
    prompt: RenderedPrompt,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 50000,
    max_validation_retries: int = 1,
    max_tokens: int = 8192,
    preview_relaxed: bool = False,
) -> Stage3Document:
    history_start = len(getattr(client, "call_history", []))
    if document.document_id != entities.document_id:
        raise Stage3Error("Stage 0 与 Stage 2 document_id 不一致")
    blocks, warnings, context_chars = select_context_blocks(
        document,
        entities,
        input_sections=input_sections,
        max_input_chars=max_input_chars,
    )
    actual_models: list[str] = []
    dropped_parameters: list[tuple[str, str]] = []
    dropped_confidence_fields: list[str] = []
    moved_confidence_values: list[str] = []
    evidence_relinks: list[dict[str, str]] = []
    dropped_sample_labels: list[str] = []
    dropped_intended_uses: list[dict[str, Any]] = []
    consecutive_process_repairs: list[dict[str, Any]] = []
    process_input_repairs: list[dict[str, Any]] = []
    preview_in_place_repairs: list[dict[str, Any]] = []
    preview_fraction_repairs: list[dict[str, Any]] = []
    state_descriptions_replaced: list[str] = []
    state_descriptions_dropped: list[str] = []
    html_entity_label_repairs: list[dict[str, str]] = []
    completed_unresolved_ids: list[str] = []
    preview_semantic_bypass_reason: str | None = None

    if entities.polymer_entities:
        feedback = None
        last_error: Exception | None = None
        parsed: SampleProcessResponse | None = None
        for attempt in range(max_validation_retries + 1):
            try:
                response = client.call_json(
                    prompt.text,
                    _user_message(
                        document.document_id,
                        entities,
                        blocks,
                        feedback,
                    ),
                    max_tokens=max_tokens,
                )
                try:
                    (
                        parsed,
                        dropped_parameters,
                        dropped_confidence_fields,
                        moved_confidence_values,
                        evidence_relinks,
                        dropped_sample_labels,
                        dropped_intended_uses,
                        consecutive_process_repairs,
                        process_input_repairs,
                        preview_in_place_repairs,
                        preview_fraction_repairs,
                        state_descriptions_replaced,
                        state_descriptions_dropped,
                        html_entity_label_repairs,
                        completed_unresolved_ids,
                    ) = _validate_response(
                        response,
                        entities,
                        blocks,
                        preview_relaxed=preview_relaxed,
                    )
                except ValueError as exc:
                    if (
                        not preview_relaxed
                        or not _preview_semantic_validation_can_bypass(exc)
                    ):
                        raise
                    (
                        parsed,
                        dropped_confidence_fields,
                        moved_confidence_values,
                        consecutive_process_repairs,
                        process_input_repairs,
                        preview_in_place_repairs,
                        preview_fraction_repairs,
                    ) = _prepare_response_structure(
                        response,
                        preview_relaxed=True,
                    )
                    preview_semantic_bypass_reason = _validation_feedback(exc)
                actual_models.append(response.model)
                last_error = None
                break
            except (LLMRequestError, ValidationError, ValueError) as exc:
                last_error = exc
                feedback = _validation_feedback(exc)
                if attempt >= max_validation_retries:
                    break
        if last_error is not None or parsed is None:
            raise Stage3Error(
                f"{document.document_id} 响应校验失败："
                f"{_validation_feedback(last_error or ValueError('empty'))}"
            ) from last_error
    else:
        parsed = SampleProcessResponse()

    samples, process_steps, polymer_type_overrides = _materialize(
        parsed,
        blocks,
        entities,
    )
    samples, process_polymer_type_inferences = (
        _apply_process_polymer_type_policy(samples, process_steps)
    )
    (
        samples,
        material_type_evidence_inferences,
        material_type_inheritance_items,
        material_type_defaults,
    ) = _apply_material_type_policy(samples, process_steps)
    sample_id_map = {
        candidate.sample_id: samples[index].sample_id
        for index, candidate in enumerate(parsed.samples)
    }
    step_id_map = {
        candidate.step_id: process_steps[index].step_id
        for index, candidate in enumerate(parsed.process_steps)
    }
    if isinstance(client, _FailureReplayClient):
        warnings.append({
            "stage": STAGE_ID,
            "code": "failure_response_replayed",
            "message": "已离线回放保存的 Stage 3 响应，未请求模型",
            "source": client.failure_path.name,
        })
    if preview_semantic_bypass_reason is not None:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_semantic_validation_bypassed",
            "message": (
                "Preview 已保留 Schema/Process 图合法候选，并跳过原文/evidence "
                "语义对应校验；Strict 模式仍会报错"
            ),
            "reason": preview_semantic_bypass_reason,
        })
    if polymer_type_overrides:
        warnings.append({
            "stage": STAGE_ID,
            "code": "sample_polymer_type_overridden",
            "message": (
                "Sample 的聚合物分类与已解析 PolymerEntity 冲突；"
                "已确定性采用 Stage 2 entity 的明确分类"
            ),
            "repairs": polymer_type_overrides,
        })
    if process_polymer_type_inferences:
        warnings.append({
            "stage": STAGE_ID,
            "code": "sample_polymer_type_process_inferred",
            "message": (
                "已根据多聚合物配方步骤或成分保持工艺补全 polymer_type"
            ),
            "items": process_polymer_type_inferences,
        })
    if material_type_evidence_inferences:
        warnings.append({
            "stage": STAGE_ID,
            "code": "material_type_evidence_inferred",
            "message": "已根据样品中的明确配方或增强组分证据补全 material_type",
            "items": material_type_evidence_inferences,
        })
    if material_type_inheritance_items:
        warnings.append({
            "stage": STAGE_ID,
            "code": "material_type_process_inherited",
            "message": "成分保持工艺的输出样品已继承一致的输入 material_type",
            "items": material_type_inheritance_items,
        })
    if material_type_defaults:
        warnings.append({
            "stage": STAGE_ID,
            "code": "material_type_default_inferred",
            "message": "无第二配方组分反证的单一聚合物样品已推断为 neat_resin",
            "items": material_type_defaults,
        })
    if consecutive_process_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "consecutive_process_samples_inserted",
            "message": (
                "连续工艺共用了最终 Sample；已根据明确的工艺顺序和"
                "逐字证据插入一一对应中间 Sample"
            ),
            "repairs": [
                {
                    "pattern": item["pattern"],
                    "previous_step_id": step_id_map[item["previous_step_id"]],
                    "current_step_id": step_id_map[item["current_step_id"]],
                    "intermediate_sample_ids": [
                        sample_id_map[sample_id]
                        for sample_id in item["intermediate_sample_ids"]
                    ],
                }
                for item in consecutive_process_repairs
            ],
        })
    evidence_key_repairs = [
        item for item in preview_in_place_repairs
        if item.get("pattern") == "trailing_colon_key_removed"
    ]
    if evidence_key_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_evidence_key_repaired",
            "message": (
                "Preview 已确定性修复 evidence.source_sentence 的键名拼写；"
                "Strict 模式仍会报错"
            ),
            "repairs": evidence_key_repairs,
        })
    process_type_repairs = [
        item for item in preview_in_place_repairs
        if item.get("pattern") == "process_type_normalized"
    ]
    if process_type_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_process_type_normalized",
            "message": (
                "Preview 已将明确的 oxidative polymerization 修饰词"
                "规范为 polymerization；Strict 模式仍会报错"
            ),
            "repairs": process_type_repairs,
        })
    in_place_output_repairs = [
        item for item in preview_in_place_repairs
        if "step_id" in item and "final_sample_ids" in item
    ]
    if in_place_output_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_in_place_postprocess_outputs_split",
            "message": (
                "Preview 下检测到原位后处理复用了既有 Sample；"
                "已一一创建新的最终 Sample，保留唯一 producer 约束"
            ),
            "repairs": [
                {
                    "step_id": step_id_map[item["step_id"]],
                    "input_sample_ids": [
                        sample_id_map[sample_id]
                        for sample_id in item["input_sample_ids"]
                    ],
                    "final_sample_ids": [
                        sample_id_map[sample_id]
                        for sample_id in item["final_sample_ids"]
                    ],
                }
                for item in in_place_output_repairs
            ],
        })
    producer_conflict_repairs = [
        item for item in preview_in_place_repairs
        if item.get("pattern") == "ambiguous_multiple_producers_removed"
    ]
    if producer_conflict_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_ambiguous_producer_links_removed",
            "message": (
                "Preview 下同一 Sample 存在多个 producer 且无法唯一归属；"
                "已删除全部冲突生成关联，未猜测保留任一步骤"
            ),
            "repairs": [
                {
                    **item,
                    "sample_ids": [
                        sample_id_map[sample_id]
                        for sample_id in item["sample_ids"]
                    ],
                }
                for item in producer_conflict_repairs
            ],
        })
    if preview_fraction_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_duplicate_upstream_fraction_outputs_removed",
            "message": (
                "Preview 下检测到明确的 fraction 被 polymerization 与 "
                "fractionation 重复生成；已从上游 polymerization 输出移除，"
                "保留 fractionation 为唯一 producer"
            ),
            "repairs": [
                {
                    "polymerization_step_id": step_id_map[
                        item["polymerization_step_id"]
                    ],
                    "fractionation_step_id": step_id_map[
                        item["fractionation_step_id"]
                    ],
                    "sample_ids": [
                        sample_id_map[sample_id]
                        for sample_id in item["sample_ids"]
                    ],
                }
                for item in preview_fraction_repairs
            ],
        })
    if process_input_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "process_input_unresolved",
            "message": (
                "ProcessStep Sample 引用存在重复或输入输出重叠；"
                "已去重，并仅删除重叠输入，未推测替代 Sample"
            ),
            "steps": [
                {
                    "step_id": step_id_map[item["step_id"]],
                    "removed_input_sample_ids": [
                        sample_id_map[sample_id]
                        for sample_id in item["removed_input_sample_ids"]
                    ],
                    **({
                        "duplicate_input_sample_ids": [
                            sample_id_map[sample_id]
                            for sample_id in item[
                                "duplicate_input_sample_ids"
                            ]
                        ],
                    } if item.get("duplicate_input_sample_ids") else {}),
                    **({
                        "duplicate_output_sample_ids": [
                            sample_id_map[sample_id]
                            for sample_id in item[
                                "duplicate_output_sample_ids"
                            ]
                        ],
                    } if item.get("duplicate_output_sample_ids") else {}),
                }
                for item in process_input_repairs
            ],
        })
    if state_descriptions_replaced:
        warnings.append({
            "stage": STAGE_ID,
            "code": "state_description_replaced_with_evidence",
            "message": (
                "无标签 Sample 的 state_description 无法逐字定位；"
                "已使用验证后的完整 evidence.source_sentence"
            ),
            "sample_ids": [
                sample_id_map[sample_id]
                for sample_id in state_descriptions_replaced
            ],
        })
    if state_descriptions_dropped:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unsupported_state_descriptions_dropped",
            "message": (
                "已有有效 sample_label_raw；删除无法逐字定位的 "
                "state_description"
            ),
            "sample_ids": [
                sample_id_map[sample_id]
                for sample_id in state_descriptions_dropped
            ],
        })
    if html_entity_label_repairs:
        warnings.append({
            "stage": STAGE_ID,
            "code": "sample_label_html_entity_surface_recovered",
            "message": (
                "sample_label_raw 与 evidence 仅存在 HTML entity "
                "编码差异；已写回 Stage 0 原始文本"
            ),
            "items": [
                {
                    **item,
                    "sample_id": sample_id_map[item["sample_id"]],
                }
                for item in html_entity_label_repairs
            ],
        })
    if dropped_intended_uses:
        dropped_by_sample: dict[str, list[str]] = {}
        for item in dropped_intended_uses:
            final_sample_id = sample_id_map[item["sample_id"]]
            dropped_by_sample.setdefault(final_sample_id, []).extend(
                item["values"]
            )
        warnings.append({
            "stage": STAGE_ID,
            "code": "unsupported_intended_uses_dropped",
            "message": (
                "Sample 已由逐字标签或状态锚定；"
                "删除无法在 evidence 中逐字定位的 intended_use"
            ),
            "samples": [
                {
                    "sample_id": sample_id,
                    "values": list(dict.fromkeys(values)),
                }
                for sample_id, values in dropped_by_sample.items()
            ],
        })
    if evidence_relinks:
        warnings.append({
            "stage": STAGE_ID,
            "code": "sample_evidence_relinked",
            "message": "样品标签仅在一个输入 block 中出现，已重定位 evidence",
            "samples": evidence_relinks,
        })
    if dropped_sample_labels:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unsupported_sample_labels_dropped",
            "message": (
                "sample_label_raw 无法在 evidence 中定位；"
                "已有逐字 state_description，故删除无依据标签"
            ),
            "sample_ids": dropped_sample_labels,
        })
    if dropped_parameters:
        step_id_map = {
            candidate.step_id: process_steps[index].step_id
            for index, candidate in enumerate(parsed.process_steps)
        }
        dropped_by_step: dict[str, list[str]] = {}
        for candidate_step_id, key in dropped_parameters:
            final_step_id = step_id_map[candidate_step_id]
            dropped_by_step.setdefault(final_step_id, []).append(key)
        warnings.append({
            "stage": STAGE_ID,
            "code": "parameters_not_in_source",
            "message": "无法定位到 evidence 原文的参数已舍弃",
            "steps": [
                {
                    "step_id": step_id,
                    "parameter_keys": keys,
                }
                for step_id, keys in dropped_by_step.items()
            ],
        })
    if parsed.unresolved_entity_ids:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unresolved_entities",
            "message": (
                f"{len(parsed.unresolved_entity_ids)} 个 entity 未解析到实际样品"
            ),
            "entity_ids": parsed.unresolved_entity_ids,
        })
    if completed_unresolved_ids:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unresolved_entities_completed",
            "message": (
                "模型遗漏了 entity 覆盖清单；已将没有任何 Sample 引用的"
                "已知 entity 确定性补入 unresolved_entity_ids"
            ),
            "entity_ids": completed_unresolved_ids,
        })
    samples_without_entity = [
        sample.sample_id
        for sample in samples
        if sample.refers_to_entity is None
    ]
    if samples_without_entity:
        warnings.append({
            "stage": STAGE_ID,
            "code": "samples_without_entity",
            "message": "部分样品无法可靠关联 PolymerEntity",
            "sample_ids": samples_without_entity,
        })
    produced_ids = {
        sample_id
        for step in process_steps
        for sample_id in step.output_sample_ids
    }
    orphan_ids = [
        sample.sample_id
        for sample in samples
        if sample.sample_kind not in INITIAL_SAMPLE_KINDS
        and sample.sample_id not in produced_ids
    ]
    if orphan_ids:
        warnings.append({
            "stage": STAGE_ID,
            "code": "orphan_noninitial_samples",
            "message": "非初始样品缺少原文支持的生成 ProcessStep",
            "sample_ids": orphan_ids,
        })
    if dropped_confidence_fields:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_fields_compacted",
            "message": "confidence 已确定性收敛为仅保留 score",
            "fields": list(dict.fromkeys(dropped_confidence_fields)),
        })

    input_hash, model_config_hash, cache_key = _cache_components(
        document,
        entities,
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
    provenance = Stage3Provenance(
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
    return Stage3Document(
        document_id=document.document_id,
        samples=samples,
        process_steps=process_steps,
        unresolved_entity_ids=parsed.unresolved_entity_ids,
        provenance=provenance,
        warnings=warnings,
    )


def run_stage3(
    stage0_path: Path,
    stage2_path: Path,
    output_path: Path,
    client: LLMClient,
    prompt: RenderedPrompt,
    *,
    force: bool = False,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 50000,
    max_validation_retries: int = 1,
    max_tokens: int = 8192,
    preview_relaxed: bool = False,
) -> tuple[Path, bool]:
    document = load_stage0_document(stage0_path)
    entities = load_stage2_document(stage2_path)
    _, _, expected_cache_key = _cache_components(
        document,
        entities,
        prompt,
        client,
        preview_relaxed=preview_relaxed,
    )
    if output_path.is_file() and not force:
        try:
            cached = Stage3Document.model_validate_json(
                output_path.read_text(encoding="utf-8-sig")
            )
            if cached.provenance.cache_key == expected_cache_key:
                return output_path, True
            for compatible_version in COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS:
                _, _, compatible_cache_key = _cache_components(
                    document,
                    entities,
                    prompt,
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

    result = extract_samples_processes(
        document,
        entities,
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
        _stage3_output_payload(result),
    )
    return output_path, False


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages") or {}
    stage = stages.get(STAGE_ID) or {}
    if not isinstance(stage, dict):
        raise Stage3Error(f"配置 {STAGE_ID} 必须是对象")
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 Stage 3 Sample/Process 抽取")
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
        help="Preview 下允许确定性修复原位后处理的重复 Sample 输出",
    )
    parser.add_argument(
        "--replay-failure",
        action="store_true",
        help="离线回放现有 stage3_failure.json，不请求模型",
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
        stage_config.get("prompt_id") or "polymer.stage3.sample_process"
    )
    prompt = PromptLoader().render_stage_prompt(
        prompt_id,
        SampleProcessResponse,
        expected_stage=STAGE_ID,
        expected_output_schema=OUTPUT_SCHEMA_VERSION,
    )
    if args.replay_failure and not args.ref_no:
        raise Stage3Error("--replay-failure 必须与单篇 --ref-no 配合使用")
    client = (
        _failure_replay_client(
            input_root / args.ref_no / "stage3_failure.json",
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
        or 50000
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
            for path in input_root.glob("reference_no_*/stage2_entities.json")
        )
    if not ref_nos:
        raise Stage3Error(f"未找到 Stage 2 输出：{input_root}")

    failures: list[tuple[str, str]] = []
    for ref_no in ref_nos:
        history_start = len(client.call_history)
        try:
            output_path, cached = run_stage3(
                input_root / ref_no / "stage0_blocks.json",
                input_root / ref_no / "stage2_entities.json",
                output_root / ref_no / "stage3_process.json",
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
                    output_root / ref_no / "stage3_failure.json",
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
    print(f"Stage 3 完成：成功 {len(ref_nos) - len(failures)}，失败 {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
