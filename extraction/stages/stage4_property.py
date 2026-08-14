"""Stage 4：抽取 PropertyObservation 与 MeasurementCondition。"""

from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import asdict
import html
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
    ConditionQuantity,
    ConditionQuantityCandidate,
    Evidence,
    MeasurementCondition,
    MeasurementConditionCandidate,
    MeasurementContext,
    MeasurementContextCandidate,
    PropertyEvidenceCandidate,
    PropertyObservation,
    PropertyObservationCandidate,
    PropertySeries,
    PropertySeriesCandidate,
    PropertySeriesCoordinate,
    PropertySeriesCoordinateCandidate,
    PropertySeriesPoint,
    PropertySeriesPointCandidate,
    PropertyStageResponse,
    SeriesCoverage,
    Stage0Document,
    Stage0Element,
    Stage2Document,
    Stage3Document,
    Stage4Document,
    Stage4Provenance,
    TableLocatorCandidate,
    UnresolvedPropertyCandidate,
    UnresolvedPropertyObservation,
)
from stages.table_grid import resolve_table_locator, table_cells_for


STAGE_ID = "stage4_property"
OUTPUT_SCHEMA_VERSION = "property_observation_schema.v7"
IMPLEMENTATION_VERSION = "1.7.10"
CACHE_REVISION = "stage4-preview-repair-20260809"
# CACHE_REVISION 用于使本轮 Preview 修复后的缓存与旧空壳缓存隔离；
# provenance 版本保持现有 Schema 支持的 1.7.10，Strict 语义不变。
# 1.7.10 Preview 在响应结构合法但 evidence 语义校验失败时保留原候选；
# 响应结构仍无法安全解析时生成 degraded 空壳结果，Strict 行为不变。
# 1.7.9 将仅作为其他 Series 的 coordinate 留存的多值性质列降级为告警，
# 不再硬失败。该告警是本版本核心产物，旧缓存无法补出，故不声明兼容版本。
# 1.7.8 收紧了条件 locator 的反向匹配：占位单元格（纯标点/破折号）
# 不再可作锚点。凡带有该反向匹配的旧版本都可能已把条件绑定到占位格，
# 其缓存产物中的 table_locator 与新版本不一致；而本仓库无版本历史可查
# 该机制的引入版本，故不声明任何兼容版本，一律重算。
COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS: tuple[str, ...] = ()
DEFAULT_INPUT_SECTIONS = ("Methods", "Results")
DEFAULT_VOCABULARY_PATH = EXTRACTION_ROOT / "config" / "polymer_schema.yaml"
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？]\s+|\n+")
HTML_CHARACTER_REFERENCE_RE = re.compile(
    r"&(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);",
    flags=re.IGNORECASE,
)


class Stage4Error(RuntimeError):
    """Stage 4 输入、词表、LLM 响应或输出验证失败。"""


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
            raise Stage4Error("failure 响应只允许离线回放一次")
        self.calls += 1
        self.call_history.append(self.record)
        return self.response


def _failure_replay_client(
    failure_path: Path,
    config: dict[str, Any],
) -> _FailureReplayClient:
    if not failure_path.is_file():
        raise Stage4Error(f"缺少 Stage 4 failure 文件：{failure_path}")
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage4Error(f"Stage 4 failure 文件无效：{failure_path}") from exc
    raw = failure.get("raw_response") if isinstance(failure, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
        raise Stage4Error("Stage 4 failure 未保存可回放的 raw response")
    try:
        data = extract_json_object(raw["content"])
    except LLMRequestError as exc:
        raise Stage4Error(
            f"Stage 4 failure raw response 无法解析为 JSON 对象：{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise Stage4Error("Stage 4 failure raw response 必须是 JSON 对象")

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
        raise Stage4Error(f"无法读取 {label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage4Error(f"{label} JSON 无效：{path}") from exc
    except ValidationError as exc:
        raise Stage4Error(f"{label} 未通过 Schema：{path.name}") from exc


def load_stage0_document(path: Path) -> Stage0Document:
    return _load_model(path, Stage0Document, "Stage 0")


def load_stage2_document(path: Path) -> Stage2Document:
    return _load_model(path, Stage2Document, "Stage 2")


def load_stage3_document(path: Path) -> Stage3Document:
    return _load_model(path, Stage3Document, "Stage 3")


def _resolve_vocabulary_path(value: str | Path, *, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        EXTRACTION_ROOT.parent / path,
        EXTRACTION_ROOT / path,
        config_path.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_property_vocabulary(
    path: Path,
) -> tuple[dict[str, tuple[str, str]], str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Stage4Error(f"性质词表无效：{path}") from exc
    if raw.get("schema_version") != "1.0":
        raise Stage4Error("性质词表 schema_version 必须为 1.0")
    entries = raw.get("property_vocabulary")
    if not isinstance(entries, dict) or not entries:
        raise Stage4Error("性质词表 property_vocabulary 必须为非空对象")

    vocabulary: dict[str, tuple[str, str]] = {}
    for name, value in entries.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            or not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise Stage4Error(f"性质词表条目无效：{name!r}")
        vocabulary[name] = (value[0].strip(), value[1].strip())
    return vocabulary, _sha256_json(raw)


def _element_source_text(element: Stage0Element) -> str:
    if element.type in {"text", "title", "equation", "footnote"}:
        return (element.text or "").strip()
    if element.type == "table":
        parts = [
            part.strip()
            for part in (element.caption, element.table_body)
            if part and part.strip()
        ]
        return "\n".join(parts)
    if element.type == "image":
        return (element.caption or "").strip()
    return ""


def select_context_blocks(
    document: Stage0Document,
    entities: Stage2Document,
    process: Stage3Document,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 90000,
) -> tuple[list[Stage0Element], list[dict[str, Any]], int]:
    if max_input_chars < 2000:
        raise ValueError("max_input_chars 不得小于 2000")
    element_map = {element.block_id: element for element in document.elements}
    referenced_ids = {
        entity.evidence.block_id
        for entity in entities.polymer_entities
    } | {
        sample.evidence.block_id for sample in process.samples
    } | {
        step.evidence.block_id for step in process.process_steps
    }
    missing = sorted(referenced_ids - set(element_map))
    if missing:
        raise Stage4Error(f"上游输出引用了未知 block：{missing}")

    eligible_types = {
        "text",
        "title",
        "table",
        "image",
        "equation",
        "footnote",
    }
    section_ids = {
        element.block_id
        for element in document.elements
        if element.section in input_sections
        and element.type in eligible_types
        and bool(_element_source_text(element))
    }
    all_table_ids = {
        element.block_id
        for element in document.elements
        if element.type == "table" and bool(_element_source_text(element))
    }
    fallback_ids: set[str] = set()
    if not section_ids:
        fallback_ids = {
            element.block_id
            for element in document.elements
            if element.type in eligible_types
            and (element.section or "").strip().casefold() != "references"
            and bool(_element_source_text(element))
        }
    selected_ids = section_ids | all_table_ids | referenced_ids | fallback_ids
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
        raise Stage4Error(
            f"{document.document_id} Stage 4 上下文 {context_chars} 字符，"
            f"超过 max_input_chars={max_input_chars}"
        )
    warnings: list[dict[str, Any]] = []
    if not section_ids and (entities.polymer_entities or process.samples):
        warnings.append({
            "stage": STAGE_ID,
            "code": "section_fallback",
            "message": (
                "Methods/Results 为空，已回落使用全部非 References 正文与全部表格；"
                "结果需人工复核"
            ),
            "fallback_block_count": len(fallback_ids),
        })
    tables_outside_sections = sorted(all_table_ids - section_ids)
    if tables_outside_sections:
        warnings.append({
            "stage": STAGE_ID,
            "code": "all_tables_included",
            "message": "已将目标 section 之外的非空表格一并加入 Stage 4 输入",
            "table_block_ids": tables_outside_sections,
        })
    return blocks, warnings, context_chars


def _user_message(
    document_id: str,
    entities: Stage2Document,
    process: Stage3Document,
    blocks: list[Stage0Element],
    vocabulary: dict[str, tuple[str, str]],
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
    process_data = {
        "samples": [
            {
                "sample_id": item.sample_id,
                "sample_kind": item.sample_kind,
                "refers_to_entity": item.refers_to_entity,
                "polymer_name": item.polymer_name,
            }
            for item in process.samples
        ],
        "process_steps": [
            {
                "step_id": item.step_id,
                "process_type": item.process_type,
                "input_sample_ids": item.input_sample_ids,
                "output_sample_ids": item.output_sample_ids,
            }
            for item in process.process_steps
        ],
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
    vocabulary_data = {
        name: {"property_code": code, "property_category": category}
        for name, (code, category) in vocabulary.items()
    }
    message = (
        f"document_id: {document_id}\n"
        "--- BEGIN UNTRUSTED POLYMER ENTITIES ---\n"
        + json.dumps(entity_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED POLYMER ENTITIES ---\n"
        "--- BEGIN UNTRUSTED SAMPLES AND PROCESS ---\n"
        + json.dumps(process_data, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED SAMPLES AND PROCESS ---\n"
        "--- BEGIN CONTROLLED PROPERTY VOCABULARY ---\n"
        + json.dumps(vocabulary_data, ensure_ascii=False, indent=2)
        + "\n--- END CONTROLLED PROPERTY VOCABULARY ---\n"
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


LATEX_SURFACE_REPLACEMENTS = (
    (r"^{\circ}", "°"),
    (r"^\circ", "°"),
    (r"\delta", "δ"),
    (r"\chi", "χ"),
    (r"\eta", "η"),
    (r"\beta", "β"),
    (r"\pm", "±"),
    (r"\mp", "∓"),
    (r"\times", "×"),
    (r"\Omega", "Ω"),
    (r"\cdot", "·"),
    (r"\%", "%"),
)
LATEX_SURFACE_WRAPPER_RE = re.compile(
    r"\\(?:mathrm|text|overline|operatorname\*?)\s*\{([^{}]*)\}"
)
SPACED_LATEX_DEGREE_RE = re.compile(
    r"\^\s*(?:\{\s*)?\\circ\s*(?:\}\s*)?"
)
SURFACE_HYPHENS = frozenset("-‐‑‒–—−")
SURFACE_IGNORED_COMBINING_MARKS = frozenset("\u0304\u0305")
SURFACE_CHARACTER_REPLACEMENTS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
}


def _surface_projection(
    value: str,
    *,
    compact_math: bool = False,
) -> tuple[str, list[tuple[int, int]]]:
    projected: list[str] = []
    spans: list[tuple[int, int]] = []

    def append(text: str, start: int, end: int) -> None:
        for char in text:
            if char in SURFACE_IGNORED_COMBINING_MARKS:
                continue
            if char.isspace():
                if compact_math:
                    continue
                if projected and projected[-1] != " ":
                    projected.append(" ")
                    spans.append((start, end))
                elif projected:
                    spans[-1] = (spans[-1][0], end)
                continue
            if compact_math and char in "^_{}":
                continue
            normalized = SURFACE_CHARACTER_REPLACEMENTS.get(
                char,
                "-" if char in SURFACE_HYPHENS else char,
            )
            for folded in normalized.casefold():
                projected.append(folded)
                spans.append((start, end))

    def is_ocr_degree(index: int) -> bool:
        if value[index] != "～":
            return False
        previous = index - 1
        while previous >= 0 and value[previous].isspace():
            previous -= 1
        following = index + 1
        while following < len(value) and value[following].isspace():
            following += 1
        return (
            previous >= 0
            and value[previous].isdigit()
            and following < len(value)
            and value[following].casefold() in {"c", "f", "k"}
        )

    index = 0
    while index < len(value):
        spaced_degree = SPACED_LATEX_DEGREE_RE.match(value, index)
        if spaced_degree is not None:
            append("°", index, spaced_degree.end())
            index = spaced_degree.end()
            continue
        wrapper = LATEX_SURFACE_WRAPPER_RE.match(value, index)
        if wrapper is not None:
            inner, _ = _surface_projection(
                wrapper.group(1),
                compact_math=compact_math,
            )
            append(inner, index, wrapper.end())
            index = wrapper.end()
            continue
        replacement = next(
            (
                (source, target)
                for source, target in LATEX_SURFACE_REPLACEMENTS
                if value.startswith(source, index)
            ),
            None,
        )
        if replacement is not None:
            source, target = replacement
            append(target, index, index + len(source))
            index += len(source)
            continue
        if is_ocr_degree(index):
            append("°", index, index + 1)
        elif value[index] not in "$~":
            append(value[index], index, index + 1)
        index += 1

    while projected and projected[-1] == " ":
        projected.pop()
        spans.pop()
    return "".join(projected), spans


def _looks_mathematical(value: str) -> bool:
    return bool(
        re.search(r"[/\\$*^_{}±∓×°βδχρηΦφ]", value)
        or re.fullmatch(r"[A-Z][A-Za-z0-9]{0,3}", value.strip())
    )


def _resolve_html_entity_surface(source: str, candidate: str) -> str | None:
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


def _is_anchorable_cell_text(text: str) -> bool:
    """单元格文本是否可作为「反向包含」定位的锚点。

    反向匹配（判断单元格文本是否出现在句子里）对空白占位单元格没有区分力：
    表格里表示「无数据」的 "-" 会命中 caption "Measured in m-cresol at 30.0°C."
    中 m-cresol 的连字符，把测量条件锚定到一个不含任何条件信息的占位格上。
    只有含字母或数字的单元格才携带可核对的信息。
    """
    return any(character.isalnum() for character in text)


def _resolve_surface_text(source: str, candidate: str) -> str | None:
    if candidate in source:
        return candidate
    direct = re.search(re.escape(candidate), source, flags=re.IGNORECASE)
    if direct:
        return direct.group(0)
    entity_surface = _resolve_html_entity_surface(source, candidate)
    if entity_surface is not None:
        return entity_surface
    tokens = candidate.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    pattern = pattern.replace(r"\-", "[-‐‑‒–—]")
    tolerant = re.search(pattern, source, flags=re.IGNORECASE)
    if tolerant:
        return tolerant.group(0)
    normalized_source, source_spans = _surface_projection(source)
    normalized_candidate, _ = _surface_projection(candidate)
    if not normalized_candidate:
        return None
    position = normalized_source.find(normalized_candidate)
    if position >= 0:
        start = source_spans[position][0]
        end = source_spans[position + len(normalized_candidate) - 1][1]
        return source[start:end]
    if not _looks_mathematical(candidate):
        return None
    normalized_source, source_spans = _surface_projection(
        source,
        compact_math=True,
    )
    normalized_candidate, _ = _surface_projection(
        candidate,
        compact_math=True,
    )
    if not normalized_candidate:
        return None
    position = normalized_source.find(normalized_candidate)
    if position < 0:
        return None
    start = source_spans[position][0]
    end = source_spans[position + len(normalized_candidate) - 1][1]
    closing_end = end
    while closing_end < len(source):
        whitespace_end = closing_end
        while whitespace_end < len(source) and source[whitespace_end].isspace():
            whitespace_end += 1
        if whitespace_end >= len(source) or source[whitespace_end] != "}":
            break
        closing_end = whitespace_end + 1
    return source[start:closing_end]


def _source_excerpt(text: str, anchor: str, max_chars: int = 800) -> str:
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
    excerpt = text[start:end].strip()
    if len(excerpt) <= max_chars:
        return excerpt
    anchor_position = excerpt.find(anchor)
    left = max(0, anchor_position - max_chars // 2)
    right = min(len(excerpt), left + max_chars)
    left = max(0, right - max_chars)
    return excerpt[left:right].strip()


def _normalize_evidence(
    candidate: PropertyEvidenceCandidate,
    block_map: dict[str, Stage0Element],
    anchors: list[str],
    *,
    allow_table_reference: bool = False,
) -> PropertyEvidenceCandidate:
    block = block_map.get(candidate.block_id)
    if block is None:
        raise ValueError(f"未知 evidence block：{candidate.block_id}")
    source = _element_source_text(block)
    normalized_locator = None
    locator_anchor = None
    if block.type == "table":
        if candidate.table_locator is None:
            if not allow_table_reference:
                raise ValueError(f"table block {block.block_id} 缺少 table_locator")
        elif candidate.table_locator.table_id != block.block_id:
            raise ValueError("table_locator.table_id 必须等于 evidence block_id")
        if candidate.table_locator is not None:
            updates: dict[str, str | None] = {"table_id": block.block_id}
            for field in ("row_label", "column_label", "cell_value"):
                value = getattr(candidate.table_locator, field)
                if value is None and field == "cell_value":
                    updates[field] = None
                    continue
                resolved = _resolve_surface_text(source, value)
                if resolved is None:
                    raise ValueError(
                        f"{block.block_id}.table_locator.{field} 不是表格原文"
                    )
                updates[field] = resolved
            locator_payload = candidate.table_locator.model_dump(mode="python")
            locator_payload.update(updates)
            stable = resolve_table_locator(block, locator_payload)
            if stable is None:
                for field in ("cell_id", "row_index", "column_index"):
                    locator_payload[field] = None
                stable = resolve_table_locator(block, locator_payload)
            if stable is not None:
                locator_payload.update(stable)
            normalized_locator = TableLocatorCandidate.model_validate(
                locator_payload
            )
            locator_anchor = (
                normalized_locator.cell_value
                or normalized_locator.row_label
            )
    elif candidate.table_locator is not None:
        raise ValueError("非 table evidence 不得包含 table_locator")

    resolved_sentence = _resolve_surface_text(
        source,
        candidate.source_sentence,
    )
    if resolved_sentence is None:
        resolved_anchor = locator_anchor
        if resolved_anchor is None:
            for anchor in anchors:
                resolved_anchor = _resolve_surface_text(source, anchor)
                if resolved_anchor is not None:
                    break
        if resolved_anchor is None:
            raise ValueError(f"{block.block_id} evidence 无法定位到原文")
        resolved_sentence = _source_excerpt(source, resolved_anchor)
    return candidate.model_copy(update={
        "source_sentence": resolved_sentence,
        "table_locator": normalized_locator,
    })


def _normalize_raw_across_evidence(
    value: str,
    evidence: list[PropertyEvidenceCandidate],
    block_map: dict[str, Stage0Element],
    field_name: str,
) -> str:
    for item in evidence:
        resolved = _resolve_surface_text(
            _element_source_text(block_map[item.block_id]),
            value,
        )
        if resolved is not None:
            return resolved
    raise ValueError(
        f"{field_name}={value!r} 不是任一 evidence block 的原文子串"
    )


def _normalize_property_name(
    value: str,
    evidence: list[PropertyEvidenceCandidate],
    block_map: dict[str, Stage0Element],
    field_name: str,
) -> str:
    try:
        return _normalize_raw_across_evidence(
            value,
            evidence,
            block_map,
            field_name,
        )
    except ValueError as original_error:
        def comparison_key(text: str) -> str:
            without_footnotes = re.sub(
                r"\$\^\{?[A-Za-z]\}?\$",
                "",
                text,
            )
            return _normalized_table_label(without_footnotes).replace("-", "")

        normalized_value = comparison_key(value)
        candidates: dict[str, str] = {}
        for item in evidence:
            locator = item.table_locator
            if locator is None:
                continue
            source = _element_source_text(block_map[item.block_id])
            resolved = _resolve_surface_text(source, locator.column_label)
            if resolved is None:
                continue
            normalized_label = comparison_key(resolved)
            if not normalized_label or not (
                normalized_label in normalized_value
                or normalized_value in normalized_label
            ):
                continue
            candidates.setdefault(normalized_label, resolved)
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        symbols = re.findall(
            r"[A-Za-z]+_(?:\{[^{}]+\}|[A-Za-z0-9]+)",
            value,
        )
        resolved_symbols: dict[str, str] = {}
        for item in evidence:
            source = _element_source_text(block_map[item.block_id])
            for symbol in symbols:
                resolved = _resolve_surface_text(source, symbol)
                if resolved is not None:
                    resolved_symbols.setdefault(
                        _normalized_table_label(resolved),
                        resolved,
                    )
        if len(resolved_symbols) == 1:
            return next(iter(resolved_symbols.values()))
        raise original_error


def _normalize_determination_method(
    value: str | None,
    evidence: list[PropertyEvidenceCandidate],
    block_map: dict[str, Stage0Element],
    field_name: str,
    *,
    excluded_values: tuple[str | None, ...] = (),
) -> tuple[str | None, bool]:
    """优先保留原文方法名；模型改写时仅回退到唯一非通用表头。"""

    if value is None:
        return None, False
    try:
        return (
            _normalize_raw_across_evidence(
                value,
                evidence,
                block_map,
                field_name,
            ),
            False,
        )
    except ValueError as original_error:
        fragments = [
            fragment.strip()
            for fragment in re.split(
                r"[,;]|\band\b|\bas well as\b",
                value,
                flags=re.IGNORECASE,
            )
            if len(fragment.strip()) >= 4
        ]
        if len(fragments) >= 2:
            resolved_fragments: dict[str, str] = {}
            for item in evidence:
                source = _element_source_text(block_map[item.block_id])
                for fragment in fragments:
                    resolved = _resolve_surface_text(source, fragment)
                    if resolved is not None:
                        resolved_fragments.setdefault(
                            _normalized_table_label(resolved),
                            resolved,
                        )
            if len(resolved_fragments) >= 2:
                ranked = sorted(
                    resolved_fragments.values(),
                    key=lambda item: sum(char.isalpha() for char in item),
                    reverse=True,
                )
                if (
                    len(ranked) == 1
                    or sum(char.isalpha() for char in ranked[0])
                    > sum(char.isalpha() for char in ranked[1])
                ):
                    return ranked[0], True
        generic_columns = {
            "value",
            "values",
            "result",
            "results",
            "property",
            "properties",
        }
        excluded = {
            _normalized_table_label(item)
            for item in excluded_values
            if item
        }
        candidates: dict[str, str] = {}
        for item in evidence:
            locator = item.table_locator
            if locator is None:
                continue
            label = locator.column_label
            normalized = _normalized_table_label(label)
            generic = re.sub(r"[^a-z0-9]+", "", label.casefold())
            if not normalized or normalized in excluded or generic in generic_columns:
                continue
            resolved = _resolve_surface_text(
                _element_source_text(block_map[item.block_id]),
                label,
            )
            if resolved is not None:
                candidates.setdefault(normalized, resolved)
        if len(candidates) == 1:
            return next(iter(candidates.values())), True
        raise original_error


def _mark_method_recovery_confidence(item: Any, recovered: bool) -> Any:
    if not recovered:
        return item.confidence
    confidence = item.confidence
    return confidence.model_copy(update={"score": min(confidence.score, 0.5)})


def _normalize_condition_field_evidence(
    candidates: list[PropertyEvidenceCandidate],
    fallback: list[PropertyEvidenceCandidate],
    raw: str,
    block_map: dict[str, Stage0Element],
    path: str,
    degraded_blocks: list[str] | None = None,
) -> list[PropertyEvidenceCandidate]:
    source_candidates = candidates or fallback
    matching = []
    for candidate in source_candidates:
        block = block_map.get(candidate.block_id)
        if block is None:
            if candidates:
                raise ValueError(
                    f"{path}.evidence 引用了未知 block：{candidate.block_id}"
                )
            continue
        if _resolve_surface_text(_element_source_text(block), raw) is None:
            continue
        try:
            matching.append(_normalize_evidence(candidate, block_map, [raw]))
        except ValueError as exc:
            # 表格脚注/caption 中的条件（"Measured in m-cresol at 30.0°C."）
            # 不属于任何单元格，本就不存在 locator。对象级 evidence 已有
            # 同样的降级路径（见 _normalize_condition），字段级若在此硬失败
            # 会造成同一条证据两套判定标准。
            if block.type != "table" or "table_locator" not in str(exc):
                raise
            if degraded_blocks is not None:
                degraded_blocks.append(candidate.block_id)
            matching.append(
                candidate.model_copy(update={"table_locator": None})
            )
    if not matching:
        source = "字段专属 evidence" if candidates else "对象 evidence"
        raise ValueError(f"{path}.raw 无法在{source}中定位")
    return _deduplicate_candidate_evidence(matching)


def _normalize_condition(
    condition: MeasurementConditionCandidate,
    block_map: dict[str, Stage0Element],
    linked_properties: list[PropertyObservationCandidate],
) -> tuple[MeasurementConditionCandidate, bool, bool]:
    quantity_fields = (
        "temperature",
        "frequency",
        "humidity",
        "pressure",
        "wavelength",
    )
    anchors = [
        quantity.raw
        for field in quantity_fields
        if (quantity := getattr(condition, field)) is not None
    ] + list(condition.other_conditions.values())
    degraded_table_evidence = False
    try:
        evidence = _normalize_evidence(
            condition.evidence,
            block_map,
            anchors,
        )
    except ValueError as exc:
        block = block_map.get(condition.evidence.block_id)
        if (
            block is None
            or block.type != "table"
            or "table_locator" not in str(exc)
        ):
            raise
        source = _element_source_text(block)
        resolved_sentence = _resolve_surface_text(
            source,
            condition.evidence.source_sentence,
        )
        if resolved_sentence is None:
            raise
        evidence = condition.evidence.model_copy(update={
            "source_sentence": resolved_sentence,
            "table_locator": None,
        })
        degraded_table_evidence = True
    source = _element_source_text(block_map[evidence.block_id])
    if anchors and not any(
        _resolve_surface_text(source, raw_value) is not None
        for raw_value in anchors
    ):
        linkage_anchors = [
            raw_value
            for item in linked_properties
            for raw_value in (
                item.property_name_raw,
                item.value_raw,
                item.determination_method_raw,
            )
            if raw_value is not None
        ]
        scoring_anchors = linkage_anchors + [
            item.unit_raw
            for item in linked_properties
            if item.unit_raw is not None
        ]
        candidates = []
        for block in block_map.values():
            candidate_source = _element_source_text(block)
            resolved_condition_values = [
                _resolve_surface_text(candidate_source, raw_value)
                for raw_value in anchors
            ]
            if (
                any(value is None for value in resolved_condition_values)
                or not any(
                    _resolve_surface_text(candidate_source, raw_value)
                    is not None
                    for raw_value in linkage_anchors
                )
            ):
                continue
            score = sum(
                _resolve_surface_text(candidate_source, raw_value)
                is not None
                for raw_value in scoring_anchors
            )
            candidates.append((
                score,
                block,
                resolved_condition_values[0],
            ))
        if not candidates:
            linked_property_context = [
                {
                    "property_id": item.property_id,
                    "property_name_raw": item.property_name_raw,
                    "value_raw": item.value_raw,
                    "determination_method_raw": (
                        item.determination_method_raw
                    ),
                    "evidence_blocks": [
                        candidate.block_id
                        for candidate in item.evidence
                    ],
                }
                for item in linked_properties
            ]
            raise ValueError(
                f"{condition.condition_id} 的 condition raw "
                "无法与关联 property 建立可定位关系："
                f"condition_raw={anchors!r}, "
                f"evidence_block={condition.evidence.block_id!r}, "
                f"linked_properties={linked_property_context!r}"
            )
        _, replacement_block, replacement_anchor = max(
            candidates,
            key=lambda candidate: candidate[0],
        )
        evidence = condition.evidence.model_copy(update={
            "block_id": replacement_block.block_id,
            "source_sentence": _source_excerpt(
                _element_source_text(replacement_block),
                replacement_anchor,
            ),
            "table_locator": None,
        })
        source = _element_source_text(replacement_block)
        supplemented_evidence = True
        if replacement_block.type == "table":
            degraded_table_evidence = True
    else:
        supplemented_evidence = False
    updates: dict[str, Any] = {"evidence": evidence}
    degraded_field_blocks: list[str] = []
    for field in quantity_fields:
        quantity = getattr(condition, field)
        if quantity is None:
            continue
        field_evidence = _normalize_condition_field_evidence(
            quantity.evidence,
            [evidence],
            quantity.raw,
            block_map,
            f"{condition.condition_id}.{field}",
            degraded_field_blocks,
        )
        resolved_values = [
            _resolve_surface_text(
                _element_source_text(block_map[item.block_id]),
                quantity.raw,
            )
            for item in field_evidence
        ]
        raw_values = [value for value in resolved_values if value is not None]
        if not raw_values:
            raise ValueError(f"{condition.condition_id}.{field}.raw 不是原文")
        raw = raw_values[0]
        updates[field] = quantity.model_copy(update={
            "raw": raw,
            "evidence": field_evidence,
        })
    normalized_other: dict[str, str] = {}
    normalized_other_evidence: dict[str, list[PropertyEvidenceCandidate]] = {}
    for key, value in condition.other_conditions.items():
        field_evidence = _normalize_condition_field_evidence(
            condition.other_condition_evidence.get(key, []),
            [evidence],
            value,
            block_map,
            f"{condition.condition_id}.other_conditions.{key}",
            degraded_field_blocks,
        )
        resolved_values = [
            _resolve_surface_text(
                _element_source_text(block_map[item.block_id]),
                value,
            )
            for item in field_evidence
        ]
        resolved_values = [item for item in resolved_values if item is not None]
        if not resolved_values:
            raise ValueError(
                f"{condition.condition_id}.other_conditions.{key} 不是原文"
            )
        normalized_other[key] = resolved_values[0]
        normalized_other_evidence[key] = field_evidence
    updates["other_conditions"] = normalized_other
    updates["other_condition_evidence"] = normalized_other_evidence
    if degraded_field_blocks:
        degraded_table_evidence = True
    return (
        condition.model_copy(update=updates),
        degraded_table_evidence,
        supplemented_evidence,
    )


def _normalize_property(
    item: PropertyObservationCandidate,
    block_map: dict[str, Stage0Element],
    vocabulary: dict[str, tuple[str, str]],
) -> tuple[
    PropertyObservationCandidate,
    list[str],
    list[str],
    list[str],
    list[str],
]:
    if item.property_name_normalized is not None:
        entry = vocabulary.get(item.property_name_normalized)
        if entry is None:
            raise ValueError(
                f"未知 property_name_normalized："
                f"{item.property_name_normalized}"
            )
        if (item.property_code, item.property_category) != entry:
            raise ValueError("受控性质名称、代码和类别不匹配")

    anchors = [item.property_name_raw, item.value_raw]
    if item.unit_raw:
        anchors.append(item.unit_raw)
    if item.determination_method_raw:
        anchors.append(item.determination_method_raw)
    if item.determination_method_raw:
        anchors.append(item.determination_method_raw)
    usable_evidence = []
    dropped_table_blocks = []
    missing_locator_candidates = []
    for candidate in item.evidence:
        block = block_map.get(candidate.block_id)
        if block is None:
            raise ValueError(f"未知 evidence block：{candidate.block_id}")
        if block.type == "table" and candidate.table_locator is None:
            dropped_table_blocks.append(candidate.block_id)
            missing_locator_candidates.append(candidate)
            continue
        usable_evidence.append(candidate)
    evidence = []
    dropped_unanchored_blocks = []
    for candidate in usable_evidence:
        try:
            normalized_evidence = _normalize_evidence(
                candidate,
                block_map,
                anchors,
            )
            source = _element_source_text(
                block_map[normalized_evidence.block_id]
            )
            if not any(
                _resolve_surface_text(source, anchor) is not None
                for anchor in anchors
            ):
                dropped_unanchored_blocks.append(candidate.block_id)
                continue
            evidence.append(normalized_evidence)
        except ValueError as exc:
            block = block_map[candidate.block_id]
            if block.type == "table" and "table_locator" in str(exc):
                missing_locator_candidates.append(
                    candidate.model_copy(update={"table_locator": None})
                )
                dropped_table_blocks.append(candidate.block_id)
                continue
            if str(exc).endswith("evidence 无法定位到原文"):
                dropped_unanchored_blocks.append(candidate.block_id)
                continue
            raise
    degraded_table_blocks = []
    if not evidence:
        for candidate in missing_locator_candidates:
            source = _element_source_text(block_map[candidate.block_id])
            resolved_anchors = [
                _resolve_surface_text(source, anchor)
                for anchor in anchors
            ]
            if any(anchor is None for anchor in resolved_anchors):
                continue
            resolved_sentence = _resolve_surface_text(
                source,
                candidate.source_sentence,
            )
            if resolved_sentence is None:
                resolved_sentence = _source_excerpt(
                    source,
                    resolved_anchors[1],
                )
            evidence.append(candidate.model_copy(update={
                "source_sentence": resolved_sentence,
                "table_locator": None,
            }))
            degraded_table_blocks.append(candidate.block_id)
        dropped_table_blocks = [
            block_id
            for block_id in dropped_table_blocks
            if block_id not in degraded_table_blocks
        ]
    supplemented_blocks = []
    raw_fields = [
        ("property_name_raw", item.property_name_raw),
        ("value_raw", item.value_raw),
        ("unit_raw", item.unit_raw),
        ("determination_method_raw", item.determination_method_raw),
    ]
    for field_name, raw_value in raw_fields:
        if raw_value is None:
            continue
        if any(
            _resolve_surface_text(
                _element_source_text(block_map[candidate.block_id]),
                raw_value,
            )
            is not None
            for candidate in evidence
        ):
            continue
        tie_value = (
            item.property_name_raw
            if field_name == "value_raw"
            else item.value_raw
        )
        requires_tie = field_name != "determination_method_raw"
        candidates = []
        for block in block_map.values():
            if block.type not in {"text", "title", "equation", "footnote"}:
                continue
            source = _element_source_text(block)
            resolved_raw = _resolve_surface_text(source, raw_value)
            if resolved_raw is None and field_name == "determination_method_raw":
                fragments = [
                    fragment.strip()
                    for fragment in re.split(r"[,;]", raw_value)
                    if len(fragment.strip()) >= 4
                ]
                resolved_fragments = [
                    _resolve_surface_text(source, fragment)
                    for fragment in fragments
                ]
                if (
                    len(resolved_fragments) >= 2
                    and all(item is not None for item in resolved_fragments)
                ):
                    resolved_raw = resolved_fragments[0]
            resolved_tie = _resolve_surface_text(source, tie_value)
            if resolved_raw is None or (requires_tie and resolved_tie is None):
                continue
            score = sum(
                _resolve_surface_text(source, anchor) is not None
                for anchor in anchors
            )
            candidates.append((score, block, resolved_raw))
        if not candidates:
            continue
        _, block, resolved_raw = max(
            candidates,
            key=lambda candidate: candidate[0],
        )
        evidence.append(PropertyEvidenceCandidate(
            block_id=block.block_id,
            source_sentence=_source_excerpt(
                _element_source_text(block),
                resolved_raw,
            ),
            table_locator=None,
        ))
        supplemented_blocks.append(block.block_id)
    if not evidence:
        raise ValueError(
            f"{item.property_id} 没有可保留的可定位 evidence："
            f"property_name_raw={item.property_name_raw!r}, "
            f"value_raw={item.value_raw!r}, "
            f"unit_raw={item.unit_raw!r}, "
            f"determination_method_raw={item.determination_method_raw!r}, "
            f"evidence_blocks="
            f"{[candidate.block_id for candidate in item.evidence]!r}"
        )
    name_raw = _normalize_property_name(
        item.property_name_raw,
        evidence,
        block_map,
        f"{item.property_id}.property_name_raw",
    )
    value_raw = _normalize_raw_across_evidence(
        item.value_raw,
        evidence,
        block_map,
        f"{item.property_id}.value_raw",
    )
    unit_raw = (
        _normalize_raw_across_evidence(
            item.unit_raw,
            evidence,
            block_map,
            f"{item.property_id}.unit_raw",
        )
        if item.unit_raw
        else None
    )
    determination_method_raw, recovered_method = _normalize_determination_method(
        item.determination_method_raw,
        evidence,
        block_map,
        f"{item.property_id}.determination_method_raw",
        excluded_values=(
            item.property_name_raw,
            item.value_raw,
            item.unit_raw,
        ),
    )
    return (
        item.model_copy(update={
            "property_name_raw": name_raw,
            "value_raw": value_raw,
            "unit_raw": unit_raw,
            "determination_method_raw": determination_method_raw,
            "evidence": evidence,
            "confidence": _mark_method_recovery_confidence(
                item,
                recovered_method,
            ),
        }),
        dropped_table_blocks,
        dropped_unanchored_blocks,
        degraded_table_blocks,
        list(dict.fromkeys(supplemented_blocks)),
    )


def _normalize_unresolved(
    item: UnresolvedPropertyCandidate,
    block_map: dict[str, Stage0Element],
) -> tuple[
    UnresolvedPropertyCandidate,
    list[str],
    list[str],
    list[str],
    list[str],
]:
    anchors = [item.property_name_raw, item.value_raw]
    if item.unit_raw:
        anchors.append(item.unit_raw)
    evidence = []
    degraded_candidates = []
    dropped_table_blocks = []
    dropped_unanchored_blocks = []
    for candidate in item.evidence:
        block = block_map.get(candidate.block_id)
        if block is None:
            raise ValueError(f"未知 evidence block：{candidate.block_id}")
        if block.type == "table" and candidate.table_locator is None:
            degraded_candidates.append(candidate)
            dropped_table_blocks.append(candidate.block_id)
            continue
        try:
            normalized = _normalize_evidence(
                candidate,
                block_map,
                anchors,
            )
        except ValueError as exc:
            if block.type == "table" and "table_locator" in str(exc):
                degraded_candidates.append(
                    candidate.model_copy(update={"table_locator": None})
                )
                dropped_table_blocks.append(candidate.block_id)
                continue
            if str(exc).endswith("evidence 无法定位到原文"):
                dropped_unanchored_blocks.append(candidate.block_id)
                continue
            raise
        source = _element_source_text(block)
        if not any(
            _resolve_surface_text(source, anchor) is not None
            for anchor in anchors
        ):
            dropped_unanchored_blocks.append(candidate.block_id)
            continue
        evidence.append(normalized)

    degraded_table_blocks = []
    if not evidence:
        for candidate in degraded_candidates:
            source = _element_source_text(block_map[candidate.block_id])
            resolved_anchors = [
                _resolve_surface_text(source, anchor)
                for anchor in anchors
            ]
            if (
                resolved_anchors[1] is None
                or sum(anchor is not None for anchor in resolved_anchors) < 2
            ):
                continue
            resolved_sentence = _resolve_surface_text(
                source,
                candidate.source_sentence,
            )
            if resolved_sentence is None:
                resolved_sentence = _source_excerpt(
                    source,
                    resolved_anchors[1],
                )
            evidence.append(candidate.model_copy(update={
                "source_sentence": resolved_sentence,
                "table_locator": None,
            }))
            degraded_table_blocks.append(candidate.block_id)
        dropped_table_blocks = [
            block_id
            for block_id in dropped_table_blocks
            if block_id not in degraded_table_blocks
        ]

    supplemented_blocks = []
    for field_name, raw_value in (
        ("property_name_raw", item.property_name_raw),
        ("value_raw", item.value_raw),
        ("unit_raw", item.unit_raw),
        ("determination_method_raw", item.determination_method_raw),
    ):
        if raw_value is None:
            continue
        if any(
            _resolve_surface_text(
                _element_source_text(block_map[candidate.block_id]),
                raw_value,
            )
            is not None
            for candidate in evidence
        ):
            continue
        tie_value = (
            item.property_name_raw
            if field_name == "value_raw"
            else item.value_raw
        )
        requires_tie = field_name != "determination_method_raw"
        candidates = []
        for block in block_map.values():
            if block.type not in {"text", "title", "equation", "footnote"}:
                continue
            source = _element_source_text(block)
            resolved_raw = _resolve_surface_text(source, raw_value)
            if resolved_raw is None and field_name == "determination_method_raw":
                fragments = [
                    fragment.strip()
                    for fragment in re.split(r"[,;]", raw_value)
                    if len(fragment.strip()) >= 4
                ]
                resolved_fragments = [
                    _resolve_surface_text(source, fragment)
                    for fragment in fragments
                ]
                if (
                    len(resolved_fragments) >= 2
                    and all(item is not None for item in resolved_fragments)
                ):
                    resolved_raw = resolved_fragments[0]
            if (
                resolved_raw is None
                or (
                    requires_tie
                    and _resolve_surface_text(source, tie_value) is None
                )
            ):
                continue
            score = sum(
                _resolve_surface_text(source, anchor) is not None
                for anchor in anchors
            )
            candidates.append((score, block, resolved_raw))
        if not candidates:
            continue
        _, block, resolved_raw = max(
            candidates,
            key=lambda candidate: candidate[0],
        )
        evidence.append(PropertyEvidenceCandidate(
            block_id=block.block_id,
            source_sentence=_source_excerpt(
                _element_source_text(block),
                resolved_raw,
            ),
            table_locator=None,
        ))
        supplemented_blocks.append(block.block_id)
    if not evidence:
        raise ValueError(
            f"{item.unresolved_id} 没有可保留的 evidence"
        )
    determination_method_raw, recovered_method = _normalize_determination_method(
        item.determination_method_raw,
        evidence,
        block_map,
        f"{item.unresolved_id}.determination_method_raw",
        excluded_values=(
            item.property_name_raw,
            item.value_raw,
            item.unit_raw,
        ),
    )
    normalized = item.model_copy(update={
        "property_name_raw": _normalize_property_name(
            item.property_name_raw,
            evidence,
            block_map,
            f"{item.unresolved_id}.property_name_raw",
        ),
        "value_raw": _normalize_raw_across_evidence(
            item.value_raw,
            evidence,
            block_map,
            f"{item.unresolved_id}.value_raw",
        ),
        "unit_raw": (
            _normalize_raw_across_evidence(
                item.unit_raw,
                evidence,
                block_map,
                f"{item.unresolved_id}.unit_raw",
            )
            if item.unit_raw
            else None
        ),
        "determination_method_raw": determination_method_raw,
        "evidence": evidence,
        "confidence": _mark_method_recovery_confidence(
            item,
            recovered_method,
        ),
    })
    return (
        normalized,
        dropped_table_blocks,
        dropped_unanchored_blocks,
        degraded_table_blocks,
        list(dict.fromkeys(supplemented_blocks)),
    )


def _deduplicate_candidate_evidence(
    evidence: list[PropertyEvidenceCandidate],
) -> list[PropertyEvidenceCandidate]:
    unique: list[PropertyEvidenceCandidate] = []
    seen: set[str] = set()
    for item in evidence:
        key = json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _require_stable_table_locator(
    evidence: PropertyEvidenceCandidate,
    block_map: dict[str, Stage0Element],
    object_id: str,
) -> None:
    block = block_map[evidence.block_id]
    if block.type != "table":
        return
    locator = evidence.table_locator
    if locator is None or locator.cell_id is None:
        raise ValueError(
            f"{object_id} 的 table evidence 无法唯一解析到稳定 cell_id"
        )


def _normalize_measurement_context(
    context: MeasurementContextCandidate,
    evidence: list[PropertyEvidenceCandidate],
    block_map: dict[str, Stage0Element],
    object_id: str,
) -> MeasurementContextCandidate:
    if context.condition_status == "not_reported":
        return context
    quantity_fields = (
        "temperature",
        "frequency",
        "humidity",
        "pressure",
        "wavelength",
    )
    updates: dict[str, Any] = {}
    for field in quantity_fields:
        quantity = getattr(context, field)
        if quantity is None:
            continue
        matching_evidence = _normalize_condition_field_evidence(
            quantity.evidence,
            evidence,
            quantity.raw,
            block_map,
            f"{object_id}.measurement_context.{field}",
        )
        resolved = _normalize_raw_across_evidence(
            quantity.raw,
            matching_evidence,
            block_map,
            f"{object_id}.measurement_context.{field}.raw",
        )
        updates[field] = quantity.model_copy(update={
            "raw": resolved,
            "evidence": matching_evidence,
        })
    normalized_other = {}
    for key, value in context.other_conditions.items():
        matching_evidence = _normalize_condition_field_evidence(
            context.other_condition_evidence.get(key, []),
            evidence,
            value,
            block_map,
            f"{object_id}.measurement_context.other_conditions.{key}",
        )
        normalized_other[key] = _normalize_raw_across_evidence(
            value,
            matching_evidence,
            block_map,
            f"{object_id}.measurement_context.other_conditions.{key}",
        )
    updates["other_conditions"] = normalized_other
    updates["other_condition_evidence"] = {
        key: _normalize_condition_field_evidence(
            context.other_condition_evidence.get(key, []),
            evidence,
            value,
            block_map,
            f"{object_id}.measurement_context.other_conditions.{key}",
        )
        for key, value in normalized_other.items()
    }
    return context.model_copy(update=updates)


def _normalize_series_coordinate(
    coordinate: PropertySeriesCoordinateCandidate,
    block_map: dict[str, Stage0Element],
    point_id: str,
) -> PropertySeriesCoordinateCandidate:
    anchors = [coordinate.name_raw, coordinate.value_raw]
    if coordinate.unit_raw:
        anchors.append(coordinate.unit_raw)
    evidence = _normalize_evidence(
        coordinate.evidence,
        block_map,
        anchors,
    )
    _require_stable_table_locator(evidence, block_map, point_id)
    evidence_items = [evidence]
    return coordinate.model_copy(update={
        "name_raw": _normalize_raw_across_evidence(
            coordinate.name_raw,
            evidence_items,
            block_map,
            f"{point_id}.coordinates.name_raw",
        ),
        "value_raw": _normalize_raw_across_evidence(
            coordinate.value_raw,
            evidence_items,
            block_map,
            f"{point_id}.coordinates.value_raw",
        ),
        "unit_raw": (
            _normalize_raw_across_evidence(
                coordinate.unit_raw,
                evidence_items,
                block_map,
                f"{point_id}.coordinates.unit_raw",
            )
            if coordinate.unit_raw
            else None
        ),
        "evidence": evidence,
    })


def _normalize_series_point(
    point: PropertySeriesPointCandidate,
    series: PropertySeriesCandidate,
    series_evidence: list[PropertyEvidenceCandidate],
    block_map: dict[str, Stage0Element],
) -> PropertySeriesPointCandidate:
    coordinates = [
        _normalize_series_coordinate(item, block_map, point.point_id)
        for item in point.coordinates
    ]
    anchors = [
        value
        for value in (point.value_raw, point.unit_raw)
        if value is not None
    ]
    evidence = [
        _normalize_evidence(item, block_map, anchors)
        for item in point.evidence
    ]
    for item in evidence:
        _require_stable_table_locator(item, block_map, point.point_id)
    all_evidence = _deduplicate_candidate_evidence([
        *series_evidence,
        *evidence,
        *(item.evidence for item in coordinates),
    ])
    context = point.measurement_context or series.measurement_context
    context = _normalize_measurement_context(
        context,
        all_evidence,
        block_map,
        point.point_id,
    )
    sample_id = point.sample_id
    entity_id = point.entity_id
    resolution = point.sample_resolution_status
    if resolution is None:
        sample_id = series.sample_id
        entity_id = series.entity_id
        resolution = series.sample_resolution_status
    return point.model_copy(update={
        "sample_id": sample_id,
        "entity_id": entity_id,
        "sample_resolution_status": resolution,
        "coordinates": coordinates,
        "value_raw": (
            _normalize_raw_across_evidence(
                point.value_raw,
                evidence,
                block_map,
                f"{point.point_id}.value_raw",
            )
            if point.value_raw is not None
            else None
        ),
        "unit_raw": (
            _normalize_raw_across_evidence(
                point.unit_raw,
                evidence,
                block_map,
                f"{point.point_id}.unit_raw",
            )
            if point.unit_raw is not None
            else None
        ),
        "measurement_context": context,
        "evidence": evidence,
    })


def _normalize_series(
    item: PropertySeriesCandidate,
    block_map: dict[str, Stage0Element],
    vocabulary: dict[str, tuple[str, str]],
    *,
    allow_incomplete_coverage: bool = False,
) -> PropertySeriesCandidate:
    if item.property_name_normalized is not None:
        entry = vocabulary.get(item.property_name_normalized)
        if entry is None:
            raise ValueError(
                f"未知 property_name_normalized：{item.property_name_normalized}"
            )
        if (item.property_code, item.property_category) != entry:
            raise ValueError("受控性质名称、代码和类别不匹配")
    series_anchors = [item.property_name_raw]
    if item.unit_raw:
        series_anchors.append(item.unit_raw)
    if item.determination_method_raw:
        series_anchors.append(item.determination_method_raw)
    series_evidence = [
        _normalize_evidence(
            evidence,
            block_map,
            series_anchors,
            allow_table_reference=True,
        )
        for evidence in item.evidence
    ]
    points = [
        _normalize_series_point(
            point,
            item,
            series_evidence,
            block_map,
        )
        for point in item.points
    ]
    all_evidence = _deduplicate_candidate_evidence([
        *series_evidence,
        *(evidence for point in points for evidence in point.evidence),
        *(
            coordinate.evidence
            for point in points
            for coordinate in point.coordinates
        ),
    ])
    if not all_evidence:
        raise ValueError(f"{item.series_id} 没有可保留的 evidence")
    value_cell_ids = [
        evidence.table_locator.cell_id
        for point in points
        for evidence in point.evidence
        if evidence.table_locator is not None
        and evidence.table_locator.cell_id is not None
    ]
    if len(value_cell_ids) != len(set(value_cell_ids)):
        raise ValueError(f"{item.series_id} 存在重复 Series point cell_id")
    if not allow_incomplete_coverage:
        _validate_series_table_coverage(item.series_id, points, block_map)
    counts = {
        status: sum(point.coverage_status == status for point in points)
        for status in ("covered", "missing", "not_applicable")
    }
    expected = counts["covered"] + counts["missing"]
    coverage = SeriesCoverage(
        expected=expected,
        covered=counts["covered"],
        missing=counts["missing"],
        not_applicable=counts["not_applicable"],
        ratio=counts["covered"] / expected if expected else 1.0,
    )
    context = _normalize_measurement_context(
        item.measurement_context,
        all_evidence,
        block_map,
        item.series_id,
    )
    payload = item.model_dump(mode="python")
    determination_method_raw, recovered_method = _normalize_determination_method(
        item.determination_method_raw,
        all_evidence,
        block_map,
        f"{item.series_id}.determination_method_raw",
        excluded_values=(
            item.property_name_raw,
            item.unit_raw,
            *(point.value_raw for point in points),
        ),
    )
    payload.update({
        "property_name_raw": _normalize_raw_across_evidence(
            item.property_name_raw,
            all_evidence,
            block_map,
            f"{item.series_id}.property_name_raw",
        ),
        "unit_raw": (
            _normalize_raw_across_evidence(
                item.unit_raw,
                all_evidence,
                block_map,
                f"{item.series_id}.unit_raw",
            )
            if item.unit_raw
            else None
        ),
        "determination_method_raw": determination_method_raw,
        "measurement_context": context,
        "points": points,
        "coverage": coverage,
        "evidence": all_evidence,
        "confidence": _mark_method_recovery_confidence(
            item,
            recovered_method,
        ),
    })
    return PropertySeriesCandidate.model_validate(payload)


def _validate_series_table_coverage(
    series_id: str,
    points: list[PropertySeriesPointCandidate],
    block_map: dict[str, Stage0Element],
) -> None:
    def property_symbol(value: str) -> str | None:
        projection, _ = _surface_projection(value)
        normalized = re.sub(r"[^a-z0-9]+", "", projection)
        match = re.search(r"(tg|tm|to|mn|mw)", normalized)
        return match.group(1) if match else None

    def explicit_other_property_value(
        value: str,
        column_labels: set[str],
    ) -> bool:
        projection, _ = _surface_projection(value)
        if "=" not in projection:
            return False
        value_symbol = property_symbol(value)
        label_symbols = {
            symbol
            for label in column_labels
            if (symbol := property_symbol(label)) is not None
        }
        return (
            value_symbol is not None
            and bool(label_symbols)
            and value_symbol not in label_symbols
        )

    grouped: dict[str, list[TableLocatorCandidate]] = {}
    for point in points:
        locator = next(
            (
                evidence.table_locator
                for evidence in point.evidence
                if evidence.table_locator is not None
                and evidence.table_locator.cell_id is not None
            ),
            None,
        )
        if locator is not None:
            grouped.setdefault(locator.table_id, []).append(locator)
    for table_id, locators in grouped.items():
        block = block_map[table_id]
        cells = table_cells_for(block)
        rows = {locator.row_index for locator in locators}
        columns = {locator.column_index for locator in locators}
        targets = []
        if len(columns) == 1:
            column = next(iter(columns))
            first_row = min(row for row in rows if row is not None)
            labels = {
                _normalized_table_label(locator.column_label)
                for locator in locators
            }
            headers = [
                cell
                for cell in cells
                if cell.row_index < first_row
                and cell.column_index <= column
                < cell.column_index + cell.column_span
                and _normalized_table_label(cell.text) in labels
            ]
            start = max(
                (
                    cell.row_index + cell.row_span
                    for cell in headers
                ),
                default=first_row,
            )
            last_row = max(row for row in rows if row is not None)
            barrier_rows = sorted({
                cell.row_index
                for cell in cells
                if cell.column_index < column
                and cell.text.strip()
                and not any(
                    target.column_index == column
                    and target.row_index <= cell.row_index
                    < target.row_index + target.row_span
                    and target.text.strip()
                    for target in cells
                )
            })
            scalar_unit_rows = {
                cell.row_index
                for cell in cells
                if cell.column_index < column
                and re.fullmatch(
                    r"\s*\([^)]*[A-Za-z/][^)]*\)\s*",
                    cell.text.strip(),
                )
                and any(
                    label.row_index == cell.row_index
                    and label.column_index < cell.column_index
                    and label.text.strip()
                    for label in cells
                )
            }
            barrier_rows = sorted({*barrier_rows, *scalar_unit_rows})
            represented_left_labels = [
                cell
                for cell in cells
                if cell.column_index < column
                and cell.row_index in rows
                and cell.text.strip()
            ]
            if any(
                cell.row_index == first_row
                for cell in represented_left_labels
            ):
                start = max(start, first_row)
            represented_stems = {
                re.sub(
                    r"[\W_]*\d+\s*$",
                    "",
                    cell.text.strip().casefold(),
                )
                for cell in represented_left_labels
            }
            represented_stems.discard("")
            if len(represented_stems) == 1 and len(represented_left_labels) >= 2:
                stem = next(iter(represented_stems))
                barrier_rows = sorted({
                    *barrier_rows,
                    *(
                        cell.row_index
                        for cell in cells
                        if cell.column_index < column
                        and cell.text.strip()
                        and re.sub(
                            r"[\W_]*\d+\s*$",
                            "",
                            cell.text.strip().casefold(),
                        ) != stem
                    ),
                })
            if not represented_left_labels:
                barrier_rows = sorted({
                    *barrier_rows,
                    *(
                        cell.row_index
                        for cell in cells
                        if cell.column_index < column
                        and cell.text.strip()
                    ),
                })
            lower_barriers = [row for row in barrier_rows if row < first_row]
            upper_barriers = [row for row in barrier_rows if row > last_row]
            if lower_barriers:
                start = max(start, max(lower_barriers) + 1)
            end = min(upper_barriers) if upper_barriers else None
            targets = [
                cell
                for cell in cells
                if cell.column_index == column
                and cell.row_index >= start
                and (end is None or cell.row_index < end)
                and cell.text.strip()
                and re.search(r"\d", cell.text)
                and not explicit_other_property_value(
                    cell.text,
                    {locator.column_label for locator in locators},
                )
            ]
        elif len(rows) == 1:
            row = next(iter(rows))
            first_column = min(
                column for column in columns if column is not None
            )
            labels = {
                _normalized_table_label(locator.row_label)
                for locator in locators
            }
            row_labels = [
                cell
                for cell in cells
                if cell.column_index < first_column
                and cell.row_index <= row < cell.row_index + cell.row_span
                and _normalized_table_label(cell.text) in labels
            ]
            start = max(
                (
                    cell.column_index + cell.column_span
                    for cell in row_labels
                ),
                default=first_column,
            )
            targets = [
                cell
                for cell in cells
                if cell.row_index == row
                and cell.column_index >= start
                and cell.text.strip()
            ]
        if not targets:
            continue
        represented = {locator.cell_id for locator in locators}
        missing = [
            cell.cell_id for cell in targets if cell.cell_id not in represented
        ]
        if missing:
            raise ValueError(
                f"{series_id} 未覆盖同一表格行/列中的目标单元格：{missing}"
            )


def _stable_value_cells(
    evidence: list[PropertyEvidenceCandidate],
) -> set[str]:
    return {
        item.table_locator.cell_id
        for item in evidence
        if item.table_locator is not None
        and item.table_locator.cell_id is not None
    }


def _series_covered_cells(
    property_series: list[PropertySeriesCandidate],
) -> tuple[set[str], set[str]]:
    point_cells: set[str] = set()
    coordinate_cells: set[str] = set()
    for series in property_series:
        for point in series.points:
            point_cells.update(_stable_value_cells(point.evidence))
            for coordinate in point.coordinates:
                coordinate_cells.update(
                    _stable_value_cells([coordinate.evidence])
                )
    return point_cells, coordinate_cells


def _looks_like_series_property_header(value: str) -> bool:
    projection, _ = _surface_projection(value)
    normalized = re.sub(r"[^a-z0-9]+", " ", projection).strip()
    if (
        re.search(r"\b(?:polymer|poly|blend|copolymer|resin)\b", normalized)
        and re.search(r"\bmw\s*\d", normalized)
    ):
        return False
    return bool(
        re.search(
            r"(?:\btg\b|\bt\s+[gm]\b|\bmn\b|\bmw\b|\bm\s+[nw]\b|"
            r"glass\s+trans(?:ition|formation)|melting\s+temperature)",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _validate_required_table_series(
    blocks: list[Stage0Element],
    property_series: list[PropertySeriesCandidate],
    *,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    """拒绝明确的多值性质列被完全静默省略为非 Series。"""

    point_cells, coordinate_cells = _series_covered_cells(property_series)
    missing: list[dict[str, Any]] = []
    coordinate_only: list[dict[str, Any]] = []
    for block in blocks:
        if block.type != "table" or not block.table_cells:
            continue
        cells_by_column: dict[int, list[Any]] = {}
        for cell in block.table_cells:
            cells_by_column.setdefault(cell.column_index, []).append(cell)
        for header in block.table_cells:
            if header.row_index > 2:
                continue
            if not _looks_like_series_property_header(header.text):
                continue
            values = [
                cell
                for cell in cells_by_column.get(header.column_index, [])
                if cell.row_index > header.row_index
                and re.search(r"[+\-]?\d+(?:[.,]\d+)?", cell.text)
                and not _looks_like_series_property_header(cell.text)
            ]
            if len(values) < 2:
                continue
            cell_ids = {cell.cell_id for cell in values}
            if cell_ids & point_cells:
                continue
            coordinate_cell_ids = cell_ids & coordinate_cells
            if coordinate_cell_ids:
                coordinate_only.append({
                    "table_id": block.block_id,
                    "column_index": header.column_index,
                    "column_label": header.text,
                    "value_count": len(values),
                    "coordinate_cell_count": len(coordinate_cell_ids),
                    "unrepresented_cell_count": len(
                        cell_ids - coordinate_cells
                    ),
                })
                continue
            missing.append({
                "table_id": block.block_id,
                "column_index": header.column_index,
                "column_label": header.text,
                "value_count": len(values),
            })
    if missing and not allow_missing:
        raise ValueError(
            "表格存在多行性质列，该列既未作为 PropertySeries point，"
            "也未作为 PropertySeries coordinate 输出："
            f"{missing}"
        )
    return [
        *coordinate_only,
        *(
            [{**item, "representation": "missing"} for item in missing]
            if allow_missing
            else []
        ),
    ]


def _candidate_property_key(item: dict[str, Any]) -> tuple[str, str]:
    normalized = item.get("property_name_normalized")
    if isinstance(normalized, str) and normalized.strip():
        return "controlled", normalized.strip().casefold()
    raw = str(item.get("property_name_raw") or "")
    return "raw", _normalized_table_label(raw)


def _mark_candidate_relation_uncertain(
    item: dict[str, Any],
    *fields: str,
) -> None:
    confidence = item.get("confidence")
    if not isinstance(confidence, dict):
        return
    confidence["score"] = min(float(confidence.get("score", 0.5)), 0.5)


def _repair_candidate_response_payload(
    data: dict[str, Any],
    process: Stage3Document,
    blocks: list[Stage0Element] | tuple[Stage0Element, ...] = (),
    vocabulary: dict[str, tuple[str, str]] | None = None,
    *,
    preview_relaxed: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """在 Pydantic 前修复可唯一判定的跨对象关系与 point confidence。"""

    payload, _ = compact_confidence_payload(data)
    repairs = {
        "aggregate_linked": 0,
        "aggregate_range_linked": 0,
        "aggregate_multi_linked": 0,
        "duplicate_unresolved_aggregates_removed": 0,
        "point_confidence_inherited": 0,
        "series_with_inherited_confidence": 0,
        "series_confidence_inherited_from_points": 0,
        "property_names_mapped_from_code_category": 0,
        "table_locator_ids_aligned_to_evidence": 0,
        "table_locator_surfaces_repaired": 0,
        "blank_table_cell_values_normalized": 0,
        "coordinate_table_locators_synthesized": 0,
        "coordinate_locators_aligned_to_point": 0,
        "condition_table_locators_synthesized": 0,
        "condition_evidence_surfaces_repaired": 0,
        "singleton_condition_evidence_unwrapped": 0,
        "confidence_paths_normalized": 0,
        "confidence_field_aliases_normalized": 0,
        "confidence_field_descriptions_normalized": 0,
        "redundant_confidence_descriptions_removed": 0,
        "series_point_confidence_fields_removed": 0,
        "point_locators_aligned_to_coordinates": 0,
        "point_compound_locators_aligned_to_coordinates": 0,
        "measurement_context_surfaces_repaired": 0,
        "series_method_evidence_supplemented": 0,
        "series_multimethod_downgraded": 0,
        "series_context_evidence_supplemented": 0,
        "series_entity_relinked_to_sample": 0,
        "point_entity_relinked_to_sample": 0,
        "multi_subject_series_normalized": 0,
        "single_subject_series_inherited_from_points": 0,
        "empty_reported_contexts_downgraded": 0,
        "embedded_measurement_conditions_promoted": 0,
        "missing_conditions_marked_not_reported": 0,
        "series_unit_surfaces_repaired": 0,
        "preview_unresolved_controlled_fields_cleared": 0,
        "preview_invalid_unresolved_properties_removed": 0,
        "preview_legacy_properties_moved_from_series": 0,
        "preview_unbound_aggregate_properties_downgraded": 0,
        "preview_unresolved_unit_surfaces_cleared": 0,
        "preview_unanchored_other_conditions_removed": 0,
        "preview_invalid_controlled_properties_cleared": 0,
        "preview_unscoped_scalar_properties_removed": 0,
        "preview_unauditable_series_removed": 0,
        "preview_unanchored_condition_fields_removed": 0,
        "preview_scalar_series_cell_duplicates_removed": 0,
        "preview_unstable_series_points_removed": 0,
        "preview_unresolved_point_status_filled": 0,
        "preview_missing_points_recovered_from_unique_rows": 0,
        "preview_missing_point_locators_recovered_from_unique_rows": 0,
        "preview_point_locators_recovered_from_unique_rows": 0,
        "preview_unresolved_series_points_synthesized": 0,
        "point_subject_inherited_from_series": 0,
        "preview_invalid_resolution_status_normalized": 0,
        "source_text_aliases_normalized": 0,
        "preview_missing_source_sentences_filled": 0,
        "preview_condition_quantity_ranges_compacted": 0,
        "preview_series_unsupported_fields_removed": 0,
        "preview_singleton_coordinate_evidence_unwrapped": 0,
        "preview_blank_table_locators_degraded": 0,
        "preview_ambiguous_condition_objects_removed": 0,
    }
    block_map = {block.block_id: block for block in blocks}

    def normalize_evidence_aliases(value: Any) -> None:
        if isinstance(value, dict):
            source_text = value.get("source_text")
            source_sentence = value.get("source_sentence")
            if (
                source_sentence is None
                and isinstance(value.get("block_id"), str)
                and isinstance(source_text, str)
                and source_text.strip()
            ):
                value["source_sentence"] = source_text
                value.pop("source_text", None)
                repairs["source_text_aliases_normalized"] += 1
            elif (
                preview_relaxed
                and isinstance(value.get("block_id"), str)
                and (
                    not isinstance(source_sentence, str)
                    or not source_sentence.strip()
                )
            ):
                block = block_map.get(value["block_id"])
                replacement = (
                    source_text.strip()
                    if isinstance(source_text, str) and source_text.strip()
                    else _element_source_text(block) if block is not None else ""
                )
                if replacement:
                    value["source_sentence"] = replacement
                    value.pop("source_text", None)
                    repairs["preview_missing_source_sentences_filled"] += 1
            for child in value.values():
                normalize_evidence_aliases(child)
        elif isinstance(value, list):
            for child in value:
                normalize_evidence_aliases(child)

    def compact_preview_condition_quantity_ranges(value: Any) -> None:
        if isinstance(value, dict):
            for field in (
                "temperature",
                "frequency",
                "humidity",
                "pressure",
                "wavelength",
            ):
                quantity = value.get(field)
                if not isinstance(quantity, dict):
                    continue
                has_min = "value_min" in quantity
                has_max = "value_max" in quantity
                value_min = quantity.pop("value_min", None)
                value_max = quantity.pop("value_max", None)
                if has_min or has_max:
                    if (
                        quantity.get("value") is None
                        and value_min is not None
                        and value_max is not None
                        and value_min == value_max
                    ):
                        quantity["value"] = value_min
                    repairs[
                        "preview_condition_quantity_ranges_compacted"
                    ] += 1
            for child in value.values():
                compact_preview_condition_quantity_ranges(child)
        elif isinstance(value, list):
            for child in value:
                compact_preview_condition_quantity_ranges(child)

    normalize_evidence_aliases(payload)
    if preview_relaxed:
        compact_preview_condition_quantity_ranges(payload)
        for series in payload.get("property_series", []):
            if not isinstance(series, dict):
                continue
            if "molecular_weight_type" in series:
                series.pop("molecular_weight_type", None)
                repairs["preview_series_unsupported_fields_removed"] += 1
            for point in series.get("points", []):
                if not isinstance(point, dict):
                    continue
                for coordinate in point.get("coordinates", []):
                    if not isinstance(coordinate, dict):
                        continue
                    evidence = coordinate.get("evidence")
                    if (
                        isinstance(evidence, list)
                        and len(evidence) == 1
                        and isinstance(evidence[0], dict)
                    ):
                        coordinate["evidence"] = evidence[0]
                        repairs[
                            "preview_singleton_coordinate_evidence_unwrapped"
                        ] += 1
    if preview_relaxed:
        for collection_name in ("properties", "property_series"):
            for item in payload.get(collection_name, []):
                if not isinstance(item, dict):
                    continue
                controlled = (
                    item.get("property_name_normalized"),
                    item.get("property_code"),
                    item.get("property_category"),
                )
                normalized = controlled[0]
                if (
                    any(value is not None for value in controlled)
                    and (
                        not all(value is not None for value in controlled)
                        or (
                            isinstance(normalized, str)
                            and vocabulary is not None
                            and normalized not in vocabulary
                        )
                    )
                ):
                    item["property_name_normalized"] = None
                    item["property_code"] = None
                    item["property_category"] = None
                    repairs[
                        "preview_invalid_controlled_properties_cleared"
                    ] += 1

        retained_unresolved = []
        for item in payload.get("unresolved_properties", []):
            if not isinstance(item, dict):
                continue
            changed = False
            for field in (
                "property_name_normalized",
                "property_code",
                "property_category",
                "molecular_weight_type",
                "value_min",
                "value_max",
                "unit_normalized",
                "measurement_condition_id",
            ):
                if item.get(field) is not None:
                    item[field] = None
                    changed = True
            if changed:
                repairs["preview_unresolved_controlled_fields_cleared"] += 1
            if str(item.get("property_name_raw") or "").strip().casefold() == str(
                item.get("value_raw") or ""
            ).strip().casefold():
                repairs["preview_invalid_unresolved_properties_removed"] += 1
                continue
            retained_unresolved.append(item)
        payload["unresolved_properties"] = retained_unresolved

        retained_series = []
        legacy_properties = []
        for item in payload.get("property_series", []):
            is_legacy_property = (
                isinstance(item, dict)
                and isinstance(item.get("property_id"), str)
                and not isinstance(item.get("series_id"), str)
                and not isinstance(item.get("points"), list)
                and isinstance(item.get("value_raw"), str)
            )
            if is_legacy_property:
                legacy_property = dict(item)
                for field in (
                    "coverage",
                    "entity_id",
                    "points",
                    "sample_resolution_status",
                ):
                    legacy_property.pop(field, None)
                if (
                    legacy_property.get("observation_role") == "aggregate"
                    and legacy_property.get("series_id") is None
                    and not legacy_property.get("series_ids")
                    and isinstance(legacy_property.get("sample_id"), str)
                ):
                    legacy_property["observation_role"] = "single"
                    repairs[
                        "preview_unbound_aggregate_properties_downgraded"
                    ] += 1
                legacy_properties.append(legacy_property)
                repairs["preview_legacy_properties_moved_from_series"] += 1
            else:
                retained_series.append(item)
        if legacy_properties:
            payload.setdefault("properties", []).extend(legacy_properties)
            payload["property_series"] = retained_series

        retained_properties = []
        for item in payload.get("properties", []):
            if (
                isinstance(item, dict)
                and not isinstance(item.get("sample_id"), str)
            ):
                repairs["preview_unscoped_scalar_properties_removed"] += 1
                continue
            retained_properties.append(item)
        payload["properties"] = retained_properties

        dropped_series_ids = set()
        retained_series = []
        for item in payload.get("property_series", []):
            if not isinstance(item, dict):
                continue
            if item.get("sample_id") is not None or item.get("entity_id") is not None:
                retained_series.append(item)
                continue
            points = [
                point for point in item.get("points", [])
                if isinstance(point, dict)
            ]
            if points and all(
                point.get("coverage_status") == "not_applicable"
                for point in points
            ):
                if isinstance(item.get("series_id"), str):
                    dropped_series_ids.add(item["series_id"])
                repairs["preview_unauditable_series_removed"] += 1
                continue
            explicit_subjects = {
                (point.get("sample_id"), point.get("entity_id"))
                for point in points
                if point.get("sample_id") is not None
                or point.get("entity_id") is not None
            }
            all_explicit = bool(points) and all(
                point.get("sample_id") is not None
                or point.get("entity_id") is not None
                for point in points
            )
            auditable_unresolved = bool(points) and all(
                point.get("sample_id") is None
                and point.get("entity_id") is None
                and
                bool(point.get("coordinates"))
                and bool(point.get("evidence"))
                for point in points
            )
            if (
                (all_explicit and len(explicit_subjects) >= 2)
                or auditable_unresolved
            ):
                retained_series.append(item)
                continue
            if isinstance(item.get("series_id"), str):
                dropped_series_ids.add(item["series_id"])
            repairs["preview_unauditable_series_removed"] += 1
        payload["property_series"] = retained_series
        if dropped_series_ids:
            for collection_name in ("properties", "unresolved_properties"):
                payload[collection_name] = [
                    item
                    for item in payload.get(collection_name, [])
                    if not isinstance(item, dict)
                    or not (
                        ({item.get("series_id")} | set(item.get("series_ids") or []))
                        & dropped_series_ids
                    )
                ]
    sample_entities = {
        sample.sample_id: sample.refers_to_entity
        for sample in process.samples
    }
    series_items = [
        item
        for item in payload.get("property_series", [])
        if isinstance(item, dict)
    ]
    def downgrade_empty_reported_context(value: Any) -> None:
        if isinstance(value, dict):
            status = value.get("condition_status")
            has_quantity = any(
                value.get(field) is not None
                for field in (
                    "temperature", "frequency", "humidity", "pressure",
                    "wavelength",
                )
            )
            has_other = bool(value.get("other_conditions"))
            if status == "reported" and not has_quantity and not has_other:
                value["condition_status"] = "not_reported"
                repairs["empty_reported_contexts_downgraded"] += 1
            for child in value.values():
                downgrade_empty_reported_context(child)
        elif isinstance(value, list):
            for child in value:
                downgrade_empty_reported_context(child)

    downgrade_empty_reported_context(payload)
    measurement_conditions = payload.setdefault("measurement_conditions", [])
    if isinstance(measurement_conditions, list):
        known_condition_ids = {
            item.get("condition_id")
            for item in measurement_conditions
            if isinstance(item, dict)
            and isinstance(item.get("condition_id"), str)
        }
        properties_by_missing_condition: dict[str, list[dict[str, Any]]] = {}
        for item in payload.get("properties", []):
            if not isinstance(item, dict):
                continue
            condition_id = item.get("measurement_condition_id")
            if (
                isinstance(condition_id, str)
                and condition_id not in known_condition_ids
            ):
                properties_by_missing_condition.setdefault(
                    condition_id, []
                ).append(item)
        for condition_id, properties in properties_by_missing_condition.items():
            contexts = [
                item.get("measurement_context") for item in properties
            ]
            if contexts and all(context is None for context in contexts):
                common_evidence: set[str] | None = None
                evidence_by_key: dict[str, dict[str, Any]] = {}
                for property_item in properties:
                    property_keys: set[str] = set()
                    for evidence_item in property_item.get("evidence", []):
                        if not isinstance(evidence_item, dict):
                            continue
                        key = json.dumps(
                            evidence_item,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        property_keys.add(key)
                        evidence_by_key[key] = evidence_item
                    common_evidence = (
                        property_keys
                        if common_evidence is None
                        else common_evidence & property_keys
                    )
                if common_evidence and len(common_evidence) == 1:
                    scores = [
                        float(item.get("confidence", {}).get("score", 0.5))
                        for item in properties
                        if isinstance(item.get("confidence"), dict)
                    ]
                    evidence_key = next(iter(common_evidence))
                    measurement_conditions.append({
                        "condition_id": condition_id,
                        "condition_status": "not_reported",
                        "temperature": None,
                        "frequency": None,
                        "humidity": None,
                        "pressure": None,
                        "wavelength": None,
                        "other_conditions": {},
                        "other_condition_evidence": {},
                        "evidence": copy.deepcopy(
                            evidence_by_key[evidence_key]
                        ),
                        "confidence": {"score": min(scores, default=0.5)},
                    })
                    known_condition_ids.add(condition_id)
                    repairs["missing_conditions_marked_not_reported"] += 1
                continue
            if not contexts or any(
                not isinstance(context, dict) for context in contexts
            ):
                continue
            canonical_contexts = {
                json.dumps(context, ensure_ascii=False, sort_keys=True)
                for context in contexts
            }
            if len(canonical_contexts) != 1:
                continue
            evidence_items: list[dict[str, Any]] = []
            context = contexts[0]
            for field in (
                "temperature", "frequency", "humidity", "pressure",
                "wavelength",
            ):
                quantity = context.get(field)
                if not isinstance(quantity, dict):
                    continue
                evidence_items.extend(
                    item for item in quantity.get("evidence", [])
                    if isinstance(item, dict)
                )
            other_evidence = context.get("other_condition_evidence")
            if isinstance(other_evidence, dict):
                for values in other_evidence.values():
                    if isinstance(values, list):
                        evidence_items.extend(
                            item for item in values if isinstance(item, dict)
                        )
            unique_evidence = {
                json.dumps(item, ensure_ascii=False, sort_keys=True): item
                for item in evidence_items
                if isinstance(item.get("block_id"), str)
                and isinstance(item.get("source_sentence"), str)
            }
            if len(unique_evidence) != 1:
                continue
            scores = [
                float(item.get("confidence", {}).get("score", 0.5))
                for item in properties
                if isinstance(item.get("confidence"), dict)
            ]
            promoted = {
                "condition_id": condition_id,
                **copy.deepcopy(context),
                "evidence": copy.deepcopy(next(iter(unique_evidence.values()))),
                "confidence": {"score": min(scores, default=0.5)},
            }
            measurement_conditions.append(promoted)
            known_condition_ids.add(condition_id)
            repairs["embedded_measurement_conditions_promoted"] += 1
    if vocabulary:
        names_by_code_category: dict[tuple[str, str], list[str]] = {}
        for name, code_category in vocabulary.items():
            names_by_code_category.setdefault(code_category, []).append(name)
        for collection_name in (
            "properties",
            "property_series",
            "unresolved_properties",
        ):
            for item in payload.get(collection_name, []):
                if not isinstance(item, dict):
                    continue
                normalized = item.get("property_name_normalized")
                if normalized is None or normalized in vocabulary:
                    continue
                candidates = names_by_code_category.get((
                    item.get("property_code"),
                    item.get("property_category"),
                ), [])
                if len(candidates) == 1:
                    item["property_name_normalized"] = candidates[0]
                    repairs["property_names_mapped_from_code_category"] += 1
    for series in series_items:
        sample_id = series.get("sample_id")
        expected_entity = sample_entities.get(sample_id)
        if isinstance(expected_entity, str) and (
            series.get("entity_id") != expected_entity
        ):
            series["entity_id"] = expected_entity
            repairs["series_entity_relinked_to_sample"] += 1
        series_resolution = series.get("sample_resolution_status")
        if preview_relaxed and series_resolution not in {"resolved", "unresolved"}:
            if series.get("sample_id") in sample_entities:
                series["entity_id"] = sample_entities[series["sample_id"]]
                series["sample_resolution_status"] = "resolved"
            else:
                series["sample_id"] = None
                if series.get("entity_id") not in set(sample_entities.values()):
                    series["entity_id"] = None
                series["sample_resolution_status"] = "unresolved"
            series_resolution = series["sample_resolution_status"]
            repairs["preview_invalid_resolution_status_normalized"] += 1
        for point in series.get("points", []):
            if not isinstance(point, dict):
                continue
            if (
                preview_relaxed
                and point.get("sample_resolution_status")
                not in {"resolved", "unresolved"}
            ):
                if point.get("sample_id") in sample_entities:
                    point["entity_id"] = sample_entities[point["sample_id"]]
                    point["sample_resolution_status"] = "resolved"
                elif (
                    point.get("sample_id") is None
                    and point.get("entity_id") is None
                    and series_resolution in {"resolved", "unresolved"}
                ):
                    point["sample_id"] = series.get("sample_id")
                    point["entity_id"] = series.get("entity_id")
                    point["sample_resolution_status"] = series_resolution
                    repairs["point_subject_inherited_from_series"] += 1
                    if series_resolution == "unresolved":
                        repairs["preview_unresolved_point_status_filled"] += 1
                else:
                    point["sample_id"] = None
                    if point.get("entity_id") not in set(sample_entities.values()):
                        point["entity_id"] = None
                    point["sample_resolution_status"] = "unresolved"
                repairs["preview_invalid_resolution_status_normalized"] += 1
            point_sample_id = point.get("sample_id")
            point_expected_entity = sample_entities.get(point_sample_id)
            if isinstance(point_expected_entity, str) and (
                point.get("entity_id") != point_expected_entity
            ):
                point["entity_id"] = point_expected_entity
                repairs["point_entity_relinked_to_sample"] += 1
        points = [
            point
            for point in series.get("points", [])
            if isinstance(point, dict)
        ]
        if not points or any(
            point.get("sample_id") is None
            and point.get("entity_id") is None
            for point in points
        ):
            continue
        subjects = list(dict.fromkeys(
            (point.get("sample_id"), point.get("entity_id"))
            for point in points
        ))
        if len(subjects) >= 2:
            series["sample_id"] = None
            series["entity_id"] = None
            series["sample_resolution_status"] = "unresolved"
            repairs["multi_subject_series_normalized"] += 1
        elif len(subjects) == 1 and (
            series.get("sample_id") is None
            and series.get("entity_id") is None
        ):
            series["sample_id"], series["entity_id"] = subjects[0]
            series["sample_resolution_status"] = (
                "resolved" if subjects[0][0] is not None else "unresolved"
            )
            repairs["single_subject_series_inherited_from_points"] += 1
    if preview_relaxed:
        retained_properties = []
        for item in payload.get("properties", []):
            if not isinstance(item, dict):
                continue
            value_raw = item.get("value_raw")
            evidence_sources = [
                _element_source_text(block_map[evidence.get("block_id")])
                for evidence in item.get("evidence", [])
                if isinstance(evidence, dict)
                and evidence.get("block_id") in block_map
            ]
            if (
                isinstance(value_raw, str)
                and evidence_sources
                and not any(
                    _resolve_surface_text(source, value_raw) is not None
                    for source in evidence_sources
                )
            ):
                repairs["preview_invalid_unresolved_properties_removed"] += 1
                continue
            retained_properties.append(item)
        payload["properties"] = retained_properties

        for item in payload.get("unresolved_properties", []):
            if not isinstance(item, dict):
                continue
            unit_raw = item.get("unit_raw")
            if not isinstance(unit_raw, str) or not unit_raw.strip():
                continue
            evidence_sources = [
                _element_source_text(block_map[evidence.get("block_id")])
                for evidence in item.get("evidence", [])
                if isinstance(evidence, dict)
                and evidence.get("block_id") in block_map
            ]
            if evidence_sources and not any(
                _resolve_surface_text(source, unit_raw) is not None
                for source in evidence_sources
            ):
                item["unit_raw"] = None
                repairs["preview_unresolved_unit_surfaces_cleared"] += 1

    condition_properties: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("properties", []):
        if not isinstance(item, dict):
            continue
        condition_id = item.get("measurement_condition_id")
        if isinstance(condition_id, str):
            condition_properties.setdefault(condition_id, []).append(item)

    for condition in payload.get("measurement_conditions", []):
        if not isinstance(condition, dict):
            continue
        evidence = condition.get("evidence")
        if isinstance(evidence, list):
            candidates = [item for item in evidence if isinstance(item, dict)]
            unique_candidates: list[dict[str, Any]] = []
            seen_candidates: set[str] = set()
            for candidate in candidates:
                key = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    unique_candidates.append(candidate)
            selected: dict[str, Any] | None = None
            if len(unique_candidates) == 1:
                selected = unique_candidates[0]
            elif unique_candidates:
                raw_values = {
                    raw.strip()
                    for field in (
                        "temperature", "frequency", "humidity", "pressure",
                        "wavelength",
                    )
                    if isinstance((quantity := condition.get(field)), dict)
                    for raw in (
                        quantity.get("value_raw"), quantity.get("unit_raw")
                    )
                    if isinstance(raw, str) and raw.strip()
                }
                raw_values.update(
                    raw.strip()
                    for raw in (condition.get("other_conditions") or {}).values()
                    if isinstance(raw, str) and raw.strip()
                )
                matching = [
                    candidate
                    for candidate in unique_candidates
                    if (block := block_map.get(candidate.get("block_id"))) is not None
                    and raw_values
                    and any(
                        _resolve_surface_text(_element_source_text(block), raw)
                        is not None
                        for raw in raw_values
                    )
                ]
                if len(matching) == 1:
                    selected = matching[0]
                else:
                    property_block_ids = {
                        item_evidence.get("block_id")
                        for item in condition_properties.get(
                            str(condition.get("condition_id")), []
                        )
                        for item_evidence in item.get("evidence", [])
                        if isinstance(item_evidence, dict)
                    }
                    linked = [
                        candidate
                        for candidate in unique_candidates
                        if candidate.get("block_id") in property_block_ids
                    ]
                    if len(linked) == 1:
                        selected = linked[0]
            if selected is not None:
                condition["evidence"] = selected
                repairs["singleton_condition_evidence_unwrapped"] += 1
        evidence = condition.get("evidence")
        if not isinstance(evidence, dict):
            continue
        block = block_map.get(evidence.get("block_id"))
        if block is None or block.type == "table":
            continue
        source = _element_source_text(block)
        source_sentence = evidence.get("source_sentence")
        if (
            isinstance(source_sentence, str)
            and _resolve_surface_text(source, source_sentence) is not None
        ):
            continue
        repaired = False
        for item in condition_properties.get(
            str(condition.get("condition_id") or ""),
            [],
        ):
            item_evidence = item.get("evidence")
            if not isinstance(item_evidence, list) or not any(
                isinstance(candidate, dict)
                and candidate.get("block_id") == block.block_id
                for candidate in item_evidence
            ):
                continue
            for field in (
                "property_name_raw",
                "value_raw",
                "determination_method_raw",
            ):
                anchor = item.get(field)
                if not isinstance(anchor, str) or not anchor.strip():
                    continue
                resolved = _resolve_surface_text(source, anchor)
                if resolved is None:
                    continue
                evidence["source_sentence"] = _source_excerpt(
                    source,
                    resolved,
                )
                repairs["condition_evidence_surfaces_repaired"] += 1
                repaired = True
                break
            if repaired:
                break

    def normalize_blank_cell_values(value: Any) -> None:
        if isinstance(value, dict):
            locator = value.get("table_locator")
            if isinstance(locator, dict):
                cell_value = locator.get("cell_value")
                if isinstance(cell_value, str) and not cell_value.strip():
                    locator["cell_value"] = None
                    repairs["blank_table_cell_values_normalized"] += 1
            for child in value.values():
                normalize_blank_cell_values(child)
        elif isinstance(value, list):
            for child in value:
                normalize_blank_cell_values(child)

    normalize_blank_cell_values(payload)

    if preview_relaxed:
        for series in series_items:
            for point in series.get("points", []):
                if (
                    not isinstance(point, dict)
                ):
                    continue
                evidence_items = point.get("evidence")
                if not isinstance(evidence_items, list):
                    continue
                for evidence in evidence_items:
                    if not isinstance(evidence, dict):
                        continue
                    locator = evidence.get("table_locator")
                    block = block_map.get(evidence.get("block_id"))
                    source_sentence = evidence.get("source_sentence")
                    if (
                        not isinstance(locator, dict)
                        or block is None
                        or block.type != "table"
                        or not isinstance(block.table_body, str)
                        or not isinstance(source_sentence, str)
                    ):
                        continue
                    if all(
                        locator.get(field) is not None
                        for field in ("cell_id", "row_index", "column_index")
                    ):
                        continue
                    rows = re.findall(
                        r"<tr\b[^>]*>.*?</tr>",
                        block.table_body,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    row_indices = [
                        index
                        for index, row in enumerate(rows)
                        if _resolve_surface_text(row, source_sentence)
                        is not None
                    ]
                    if len(row_indices) != 1:
                        continue
                    row_index = row_indices[0]
                    column_label = str(locator.get("column_label") or "")
                    column_key = _normalized_table_label(column_label)
                    column_indices = {
                        cell.column_index
                        for cell in table_cells_for(block)
                        if cell.row_index < row_index
                        and _normalized_table_label(cell.text) == column_key
                    }
                    if len(column_indices) != 1:
                        continue
                    column_index = next(iter(column_indices))
                    value_cells = [
                        cell
                        for cell in table_cells_for(block)
                        if cell.row_index == row_index
                        and cell.column_index == column_index
                    ]
                    if len(value_cells) != 1:
                        continue
                    cell = value_cells[0]
                    value_raw = cell.text.strip()
                    point_value = point.get("value_raw")
                    is_missing = (
                        point.get("coverage_status") == "missing"
                        and point_value is None
                    )
                    value_matches = (
                        isinstance(point_value, str)
                        and _resolve_surface_text(value_raw, point_value)
                        is not None
                    )
                    if not is_missing and not value_matches:
                        continue
                    locator.update({
                        "table_id": block.block_id,
                        "cell_value": value_raw or None,
                        "cell_id": cell.cell_id,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                    })
                    if not value_raw:
                        if not is_missing:
                            continue
                        repairs[
                            "preview_missing_point_locators_recovered_from_unique_rows"
                        ] += 1
                        break
                    if value_matches:
                        repairs[
                            "preview_point_locators_recovered_from_unique_rows"
                        ] += 1
                        break
                    point["coverage_status"] = "covered"
                    point["value_raw"] = value_raw
                    _mark_candidate_relation_uncertain(point, "value_raw")
                    repairs[
                        "preview_missing_points_recovered_from_unique_rows"
                    ] += 1
                    break

        for series in series_items:
            if series.get("sample_resolution_status") != "unresolved":
                continue
            points = series.get("points")
            if not isinstance(points, list):
                continue
            stable_evidence = [
                evidence
                for point in points
                if isinstance(point, dict)
                for evidence in point.get("evidence", [])
                if isinstance(evidence, dict)
                and isinstance(evidence.get("table_locator"), dict)
                and evidence["table_locator"].get("cell_id") is not None
                and evidence["table_locator"].get("row_index") is not None
                and evidence["table_locator"].get("column_index") is not None
            ]
            table_ids = {item.get("block_id") for item in stable_evidence}
            columns = {
                item["table_locator"]["column_index"]
                for item in stable_evidence
            }
            if len(table_ids) != 1 or len(columns) != 1:
                continue
            table_id = next(iter(table_ids))
            block = block_map.get(table_id)
            if (
                block is None
                or block.type != "table"
                or not isinstance(block.table_body, str)
            ):
                continue
            column_index = next(iter(columns))
            row_indices = {
                item["table_locator"]["row_index"]
                for item in stable_evidence
            }
            if len(row_indices) < 2:
                continue
            represented = {
                item["table_locator"]["cell_id"] for item in stable_evidence
            }
            rows = re.findall(
                r"<tr\b[^>]*>.*?</tr>",
                block.table_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            column_label = str(
                stable_evidence[0]["table_locator"].get("column_label") or ""
            )
            existing_ids = {
                str(point.get("point_id"))
                for point in points
                if isinstance(point, dict)
            }
            next_index = max(
                (
                    int(match.group(1))
                    for point_id in existing_ids
                    if (match := re.fullmatch(r"pt(\d+)", point_id))
                ),
                default=0,
            ) + 1
            for cell in table_cells_for(block):
                value_raw = cell.text.strip()
                if (
                    cell.column_index != column_index
                    or cell.row_index <= min(row_indices)
                    or cell.row_index >= max(row_indices)
                    or cell.cell_id in represented
                    or any(
                        isinstance(point, dict)
                        and isinstance(point.get("value_raw"), str)
                        and _resolve_surface_text(
                            value_raw,
                            point["value_raw"],
                        ) is not None
                        and any(
                            isinstance(evidence, dict)
                            and evidence.get("block_id") == block.block_id
                            and isinstance(
                                evidence.get("source_sentence"), str
                            )
                            and _resolve_surface_text(
                                rows[cell.row_index],
                                evidence["source_sentence"],
                            ) is not None
                            for evidence in point.get("evidence", [])
                        )
                        for point in points
                    )
                    or not re.fullmatch(
                        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[A-Za-z]?",
                        value_raw,
                    )
                    or cell.row_index >= len(rows)
                ):
                    continue
                row_cells = [
                    candidate
                    for candidate in table_cells_for(block)
                    if candidate.row_index == cell.row_index
                    and candidate.column_index < column_index
                    and candidate.text.strip()
                ]
                row_label = (
                    min(row_cells, key=lambda item: item.column_index).text.strip()
                    if row_cells
                    else ""
                )
                coordinate_cell = (
                    min(row_cells, key=lambda item: item.column_index)
                    if row_cells
                    else next(
                        (
                            candidate
                            for previous_row in range(cell.row_index - 1, -1, -1)
                            for candidate in sorted(
                                (
                                    item
                                    for item in table_cells_for(block)
                                    if item.row_index == previous_row
                                    and item.column_index < column_index
                                    and item.text.strip()
                                ),
                                key=lambda item: item.column_index,
                            )
                        ),
                        None,
                    )
                )
                if coordinate_cell is None:
                    continue
                row_label = coordinate_cell.text.strip()
                coordinate_header = next(
                    (
                        candidate
                        for candidate in table_cells_for(block)
                        if candidate.column_index <= coordinate_cell.column_index
                        < candidate.column_index + candidate.column_span
                        and candidate.row_index < min(row_indices)
                        and candidate.text.strip()
                    ),
                    None,
                )
                coordinate_name = (
                    coordinate_header.text.strip()
                    if coordinate_header is not None
                    else "row"
                )
                numeric = re.match(
                    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
                    value_raw,
                )
                point_id = f"pt{next_index:03d}"
                next_index += 1
                point_evidence = {
                    "block_id": block.block_id,
                    "source_sentence": rows[cell.row_index],
                    "table_locator": {
                        "table_id": block.block_id,
                        "row_label": row_label,
                        "column_label": column_label,
                        "cell_value": value_raw,
                        "cell_id": cell.cell_id,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                    },
                }
                points.append({
                    "point_id": point_id,
                    "coverage_status": "covered",
                    "value_raw": value_raw,
                    "value_min": float(numeric.group(0)) if numeric else None,
                    "value_max": float(numeric.group(0)) if numeric else None,
                    "unit_raw": series.get("unit_raw"),
                    "unit_normalized": series.get("unit_normalized"),
                    "sample_resolution_status": "unresolved",
                    "coordinates": [{
                        "name_raw": coordinate_name,
                        "value_raw": row_label,
                        "unit_raw": None,
                        "evidence": copy.deepcopy(point_evidence),
                    }],
                    "evidence": [point_evidence],
                    "confidence": {"score": 0.5},
                })
                repairs["preview_unresolved_series_points_synthesized"] += 1

    def repair_locator(
        evidence: dict[str, Any],
        *,
        row_hints: tuple[str, ...] = (),
        column_hints: tuple[str, ...] = (),
    ) -> bool:
        locator = evidence.get("table_locator")
        block = block_map.get(evidence.get("block_id"))
        if (
            not isinstance(locator, dict)
            or block is None
            or block.type != "table"
        ):
            return False
        source = _element_source_text(block)
        before = tuple(
            locator.get(field)
            for field in ("row_label", "column_label", "cell_value")
        )
        row_label = str(locator.get("row_label") or "")
        exact_row_cells = {
            cell.text.strip()
            for cell in table_cells_for(block)
            if cell.text.strip()
        }
        row_index_hint = locator.get("row_index")
        if (
            _resolve_surface_text(source, row_label) is None
            and not isinstance(row_index_hint, int)
            and isinstance(evidence.get("source_sentence"), str)
            and isinstance(block.table_body, str)
        ):
            rows = re.findall(
                r"<tr\b[^>]*>.*?</tr>",
                block.table_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            matching_rows = [
                index
                for index, row in enumerate(rows)
                if _resolve_surface_text(
                    row,
                    evidence["source_sentence"],
                ) is not None
            ]
            if len(matching_rows) == 1:
                row_index_hint = matching_rows[0]
        if (
            row_label not in exact_row_cells
            and isinstance(row_index_hint, int)
        ):
            row_cells = [
                cell
                for cell in table_cells_for(block)
                if cell.row_index <= row_index_hint
                < cell.row_index + cell.row_span
                and cell.text.strip()
            ]
            if row_cells:
                locator["row_label"] = min(
                    row_cells,
                    key=lambda cell: cell.column_index,
                ).text.strip()
                row_label = str(locator["row_label"])
        if _resolve_surface_text(source, row_label) is None:
            replacements = [
                _resolve_surface_text(source, hint)
                for hint in row_hints
                if hint
            ]
            replacements = [item for item in replacements if item is not None]
            if len(set(replacements)) == 1:
                locator["row_label"] = replacements[0]

        column_label = str(locator.get("column_label") or "")
        if _resolve_surface_text(source, column_label) is None:
            header_candidates = []
            for cell in table_cells_for(block):
                text = cell.text.strip()
                if not text or not any(
                    _resolve_surface_text(hint, text) is not None
                    for hint in column_hints
                    if hint
                ):
                    continue
                header_candidates.append((
                    cell.column_span,
                    -len(_normalized_table_label(text)),
                    text,
                ))
            if header_candidates:
                best_score = min(item[:2] for item in header_candidates)
                best = {
                    text
                    for span, length, text in header_candidates
                    if (span, length) == best_score
                }
                if len(best) == 1:
                    locator["column_label"] = next(iter(best))

        locator_payload = dict(locator)
        for field in ("cell_id", "row_index", "column_index"):
            locator_payload[field] = None
        stable = resolve_table_locator(block, locator_payload)
        if stable is None and isinstance(row_index_hint, int):
            column_key = _normalized_table_label(
                str(locator.get("column_label") or "")
            )
            value_key, _ = _surface_projection(
                str(locator.get("cell_value") or ""),
                compact_math=True,
            )
            candidates = [
                cell
                for cell in table_cells_for(block)
                if cell.row_index == row_index_hint
                and value_key
                and _surface_projection(
                    cell.text,
                    compact_math=True,
                )[0] == value_key
                and any(
                    header.row_index < row_index_hint
                    and header.column_index <= cell.column_index
                    < header.column_index + header.column_span
                    and _normalized_table_label(header.text) == column_key
                    for header in table_cells_for(block)
                )
            ]
            if len(candidates) == 1:
                stable = {
                    "cell_id": candidates[0].cell_id,
                    "row_index": candidates[0].row_index,
                    "column_index": candidates[0].column_index,
                }
        if stable is None:
            cell_value = str(locator.get("cell_value") or "")
            value_key, _ = _surface_projection(
                cell_value,
                compact_math=True,
            )
            value_cells = [
                cell
                for cell in table_cells_for(block)
                if value_key
                and _surface_projection(
                    cell.text,
                    compact_math=True,
                )[0] == value_key
            ]
            if len(value_cells) == 1:
                value_cell = value_cells[0]
                stable = {
                    "cell_id": value_cell.cell_id,
                    "row_index": value_cell.row_index,
                    "column_index": value_cell.column_index,
                }
            elif value_cells:
                unique_hint_rows = {
                    next(iter(rows))
                    for hint in row_hints
                    if hint
                    if len(rows := {
                        cell.row_index
                        for cell in table_cells_for(block)
                        if cell.text.strip()
                        and _resolve_surface_text(cell.text, hint) is not None
                    }) == 1
                }
                column_key = _normalized_table_label(
                    str(locator.get("column_label") or "")
                )
                columns = {
                    cell.column_index
                    for cell in table_cells_for(block)
                    if column_key
                    and _normalized_table_label(cell.text) == column_key
                }
                intersections = [
                    cell
                    for cell in value_cells
                    if cell.row_index in unique_hint_rows
                    and cell.column_index in columns
                ]
                if len(intersections) == 1:
                    value_cell = intersections[0]
                    stable = {
                        "cell_id": value_cell.cell_id,
                        "row_index": value_cell.row_index,
                        "column_index": value_cell.column_index,
                    }
        if stable is not None:
            cells = table_cells_for(block)
            value_cell = next(
                (
                    cell
                    for cell in cells
                    if cell.cell_id == stable.get("cell_id")
                ),
                None,
            )
            if value_cell is not None:
                if str(locator.get("row_label") or "") not in exact_row_cells:
                    row_cells = [
                        cell
                        for cell in cells
                        if cell.row_index <= value_cell.row_index
                        < cell.row_index + cell.row_span
                        and cell.text.strip()
                    ]
                    if row_cells:
                        locator["row_label"] = min(
                            row_cells,
                            key=lambda cell: cell.column_index,
                        ).text.strip()
                numeric_counts: dict[int, int] = {}
                for cell in cells:
                    text = cell.text.strip()
                    if text == "-" or re.fullmatch(
                        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
                        text,
                    ):
                        numeric_counts[cell.row_index] = (
                            numeric_counts.get(cell.row_index, 0) + 1
                        )
                data_rows = [
                    row_index
                    for row_index, count in numeric_counts.items()
                    if count >= 2
                ]
                data_start = min(data_rows) if data_rows else 1
                headers = [
                    cell
                    for cell in cells
                    if cell.text.strip()
                    and cell.row_index < data_start
                    and cell.column_index <= value_cell.column_index
                    < cell.column_index + cell.column_span
                ]
                if headers:
                    header = min(
                        headers,
                        key=lambda cell: (
                            cell.column_span,
                            -cell.row_index,
                            -len(_normalized_table_label(cell.text)),
                        ),
                    )
                    locator["column_label"] = header.text.strip()
                locator.update({
                    "cell_value": value_cell.text or None,
                    "cell_id": value_cell.cell_id,
                    "row_index": value_cell.row_index,
                    "column_index": value_cell.column_index,
                })
        after = tuple(
            locator.get(field)
            for field in ("row_label", "column_label", "cell_value")
        )
        return before != after

    def repair_measurement_context(
        context: Any,
        evidence_items: list[dict[str, Any]],
    ) -> None:
        if not isinstance(context, dict):
            return
        sources = [
            _element_source_text(block)
            for evidence in evidence_items
            if isinstance(evidence, dict)
            and (block := block_map.get(evidence.get("block_id"))) is not None
        ]
        for field in (
            "temperature",
            "frequency",
            "humidity",
            "pressure",
            "wavelength",
        ):
            quantity = context.get(field)
            if not isinstance(quantity, dict):
                continue
            raw = quantity.get("raw")
            if isinstance(raw, str):
                for evidence in quantity.get("evidence", []):
                    if not isinstance(evidence, dict):
                        continue
                    block = block_map.get(evidence.get("block_id"))
                    if (
                        block is None
                        or block.type != "table"
                        or isinstance(evidence.get("table_locator"), dict)
                    ):
                        continue
                    anchors = [
                        evidence.get("source_sentence"),
                        raw,
                    ]
                    matching_cells = []
                    for anchor in anchors:
                        if not isinstance(anchor, str) or not anchor:
                            continue
                        matching_cells = [
                            cell
                            for cell in table_cells_for(block)
                            if cell.text.strip()
                            if _resolve_surface_text(cell.text, anchor) is not None
                        ]
                        if not matching_cells:
                            matching_cells = [
                                cell
                                for cell in table_cells_for(block)
                                if _is_anchorable_cell_text(cell.text)
                                if _resolve_surface_text(anchor, cell.text) is not None
                            ]
                        if len(matching_cells) == 1:
                            break
                    if len(matching_cells) != 1:
                        continue
                    cell = matching_cells[0]
                    evidence["table_locator"] = {
                        "table_id": block.block_id,
                        "row_label": cell.text,
                        "column_label": cell.text,
                        "cell_value": cell.text,
                        "cell_id": cell.cell_id,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                    }
                    repairs["condition_table_locators_synthesized"] += 1
            if not isinstance(raw, str) or any(
                _resolve_surface_text(source, raw) is not None
                for source in sources
            ):
                continue
            variants = []
            if r"\pm" in raw:
                variants.append(raw.replace(r"\pm", r"\mp"))
            if r"\mp" in raw:
                variants.append(raw.replace(r"\mp", r"\pm"))
            if "±" in raw:
                variants.append(raw.replace("±", "∓"))
            if "∓" in raw:
                variants.append(raw.replace("∓", "±"))
            matches = {
                resolved
                for source in sources
                for variant in variants
                if (resolved := _resolve_surface_text(source, variant))
                is not None
            }
            if len(matches) == 1:
                quantity["raw"] = next(iter(matches))
                repairs["measurement_context_surfaces_repaired"] += 1
            if preview_relaxed:
                field_sources = [
                    _element_source_text(block)
                    for evidence in quantity.get("evidence", [])
                    if isinstance(evidence, dict)
                    and (
                        block := block_map.get(evidence.get("block_id"))
                    ) is not None
                ]
                current_raw = quantity.get("raw")
                if (
                    isinstance(current_raw, str)
                    and field_sources
                    and not any(
                        _resolve_surface_text(source, current_raw) is not None
                        for source in field_sources
                    )
                ):
                    context[field] = None
                    repairs["preview_unanchored_condition_fields_removed"] += 1
        if preview_relaxed:
            other_conditions = context.get("other_conditions")
            other_evidence = context.get("other_condition_evidence")
            if isinstance(other_conditions, dict):
                if not isinstance(other_evidence, dict):
                    other_evidence = {}
                    context["other_condition_evidence"] = other_evidence
                for key, raw in list(other_conditions.items()):
                    field_sources = [
                        _element_source_text(block)
                        for evidence in other_evidence.get(key, [])
                        if isinstance(evidence, dict)
                        and (
                            block := block_map.get(evidence.get("block_id"))
                        ) is not None
                    ]
                    if (
                        isinstance(raw, str)
                        and field_sources
                        and not any(
                            _resolve_surface_text(source, raw) is not None
                            for source in field_sources
                        )
                    ):
                        other_conditions.pop(key, None)
                        other_evidence.pop(key, None)
                        repairs[
                            "preview_unanchored_other_conditions_removed"
                        ] += 1

    for condition in payload.get("measurement_conditions", []):
        if not isinstance(condition, dict):
            continue
        condition_evidence = condition.get("evidence")
        evidence_items = (
            [condition_evidence]
            if isinstance(condition_evidence, dict)
            else []
        )
        repair_measurement_context(condition, evidence_items)
        evidence_candidates = [
            condition_evidence,
            *(
                evidence
                for field in (
                    "temperature",
                    "frequency",
                    "humidity",
                    "pressure",
                    "wavelength",
                )
                if isinstance((quantity := condition.get(field)), dict)
                for evidence in quantity.get("evidence", [])
            ),
        ]
        for evidence in evidence_candidates:
            if not isinstance(evidence, dict):
                continue
            block = block_map.get(evidence.get("block_id"))
            if (
                block is None
                or block.type != "table"
                or isinstance(evidence.get("table_locator"), dict)
            ):
                continue
            sentence = evidence.get("source_sentence")
            if not isinstance(sentence, str) or not sentence:
                continue
            matching_cells = [
                cell
                for cell in table_cells_for(block)
                if cell.text.strip()
                if _resolve_surface_text(cell.text, sentence) is not None
            ]
            if not matching_cells:
                matching_cells = [
                    cell
                    for cell in table_cells_for(block)
                    if _is_anchorable_cell_text(cell.text)
                    if _resolve_surface_text(sentence, cell.text) is not None
                ]
            if len(matching_cells) == 1:
                cell = matching_cells[0]
                evidence["table_locator"] = {
                    "table_id": block.block_id,
                    "row_label": cell.text,
                    "column_label": cell.text,
                    "cell_value": cell.text,
                    "cell_id": cell.cell_id,
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                }
            elif evidence is condition_evidence:
                field_locators = [
                    field_evidence.get("table_locator")
                    for field in (
                        "temperature",
                        "frequency",
                        "humidity",
                        "pressure",
                        "wavelength",
                    )
                    if isinstance((quantity := condition.get(field)), dict)
                    for field_evidence in quantity.get("evidence", [])
                    if isinstance(field_evidence, dict)
                    and isinstance(field_evidence.get("table_locator"), dict)
                ]
                stable_ids = {
                    locator.get("cell_id")
                    for locator in field_locators
                    if locator.get("cell_id") is not None
                }
                if len(stable_ids) == 1:
                    evidence["table_locator"] = copy.deepcopy(field_locators[0])
                else:
                    continue
            else:
                continue
            repairs["condition_table_locators_synthesized"] += 1

    for collection_name in ("properties", "unresolved_properties"):
        for item in payload.get(collection_name, []):
            if not isinstance(item, dict):
                continue
            for evidence in item.get("evidence", []):
                if not isinstance(evidence, dict):
                    continue
                block = block_map.get(evidence.get("block_id"))
                locator = evidence.get("table_locator")
                if (
                    block is not None
                    and block.type == "table"
                    and isinstance(locator, dict)
                    and locator.get("table_id") != block.block_id
                ):
                    locator["table_id"] = block.block_id
                    repairs["table_locator_ids_aligned_to_evidence"] += 1
                if repair_locator(
                    evidence,
                    row_hints=(str(item.get("property_name_raw") or ""),),
                    column_hints=(
                        str(item.get("determination_method_raw") or ""),
                    ),
                ):
                    repairs["table_locator_surfaces_repaired"] += 1
                locator = evidence.get("table_locator")
                block = block_map.get(evidence.get("block_id"))
                property_name = item.get("property_name_raw")
                if (
                    not preview_relaxed
                    or
                    not isinstance(locator, dict)
                    or block is None
                    or block.type != "table"
                    or not isinstance(property_name, str)
                    or not isinstance(locator.get("row_index"), int)
                ):
                    continue
                row_index = locator["row_index"]
                matching_row_labels = [
                    cell
                    for cell in table_cells_for(block)
                    if cell.row_index <= row_index < cell.row_index + cell.row_span
                    and _resolve_surface_text(cell.text, property_name) is not None
                ]
                if len(matching_row_labels) == 1 and (
                    locator.get("row_label") != matching_row_labels[0].text
                ):
                    locator["row_label"] = matching_row_labels[0].text
                    for field in ("cell_id", "row_index", "column_index"):
                        locator[field] = None
                    repair_locator(
                        evidence,
                        row_hints=(property_name,),
                        column_hints=(
                            str(item.get("determination_method_raw") or ""),
                        ),
                    )
                    repairs["table_locator_surfaces_repaired"] += 1
    for series in series_items:
        method_raw = series.get("determination_method_raw")
        series_evidence = series.setdefault("evidence", [])
        if (
            isinstance(method_raw, str)
            and not any(
                _resolve_surface_text(
                    _element_source_text(block_map[evidence.get("block_id")]),
                    method_raw,
                ) is not None
                for evidence in series_evidence
                if isinstance(evidence, dict)
                and evidence.get("block_id") in block_map
            )
        ):
            method_blocks = [
                (block, resolved)
                for block in blocks
                if block.type in {"text", "title", "equation", "footnote"}
                and (
                    resolved := _resolve_surface_text(
                        _element_source_text(block),
                        method_raw,
                    )
                ) is not None
            ]
            preferred_method_blocks = method_blocks
            if len(method_blocks) > 1:
                methods_matches = [
                    item
                    for item in method_blocks
                    if item[0].section == "Methods"
                ]
                if len(methods_matches) == 1:
                    preferred_method_blocks = methods_matches
            if len(preferred_method_blocks) == 1:
                block, resolved = preferred_method_blocks[0]
                series_evidence.append({
                    "block_id": block.block_id,
                    "source_sentence": _source_excerpt(
                        _element_source_text(block),
                        resolved,
                    ),
                    "table_locator": None,
                })
                repairs["series_method_evidence_supplemented"] += 1
            elif "/" in method_raw:
                method_parts = [
                    part.strip()
                    for part in method_raw.split("/")
                    if part.strip()
                ]
                multi_method_blocks = [
                    block
                    for block in blocks
                    if block.type in {"text", "title", "equation", "footnote"}
                    and all(
                        _resolve_surface_text(
                            _element_source_text(block),
                            part,
                        ) is not None
                        for part in method_parts
                    )
                ]
                if len(multi_method_blocks) == 1:
                    block = multi_method_blocks[0]
                    source = _element_source_text(block)
                    anchor = _resolve_surface_text(source, method_parts[0])
                    series_evidence.append({
                        "block_id": block.block_id,
                        "source_sentence": _source_excerpt(source, anchor),
                        "table_locator": None,
                    })
                    series["determination_method_raw"] = None
                    confidence = series.get("confidence")
                    if isinstance(confidence, dict):
                        confidence["score"] = min(
                            float(confidence.get("score", 0.5)),
                            0.5,
                        )
                    repairs["series_multimethod_downgraded"] += 1
        series_point_evidence = [
            evidence
            for point in series.get("points", [])
            if isinstance(point, dict)
            for evidence in point.get("evidence", [])
            if isinstance(evidence, dict)
        ]
        repair_measurement_context(
            series.get("measurement_context"),
            [
                *[
                    evidence
                    for evidence in series.get("evidence", [])
                    if isinstance(evidence, dict)
                ],
                *series_point_evidence,
            ],
        )
        context_raw_values = []
        for context in [
            series.get("measurement_context"),
            *[
                point.get("measurement_context")
                for point in series.get("points", [])
                if isinstance(point, dict)
            ],
        ]:
            if not isinstance(context, dict):
                continue
            for field in (
                "temperature",
                "frequency",
                "humidity",
                "pressure",
                "wavelength",
            ):
                quantity = context.get(field)
                if isinstance(quantity, dict) and isinstance(
                    quantity.get("raw"), str
                ):
                    context_raw_values.append(quantity["raw"])
            context_raw_values.extend(
                value
                for value in (context.get("other_conditions") or {}).values()
                if isinstance(value, str)
            )
        for raw_value in dict.fromkeys(context_raw_values):
            current_evidence = [*series_evidence, *series_point_evidence]
            if any(
                _resolve_surface_text(
                    _element_source_text(block_map[evidence.get("block_id")]),
                    raw_value,
                ) is not None
                for evidence in current_evidence
                if evidence.get("block_id") in block_map
            ):
                continue
            context_anchors = [
                value
                for value in (
                    series.get("property_name_raw"),
                    series.get("determination_method_raw"),
                )
                if isinstance(value, str) and value
            ]
            context_blocks = [
                (
                    sum(
                        _resolve_surface_text(
                            _element_source_text(block),
                            anchor,
                        ) is not None
                        for anchor in context_anchors
                    ),
                    block,
                    resolved,
                )
                for block in blocks
                if (
                    resolved := _resolve_surface_text(
                        _element_source_text(block),
                        raw_value,
                    )
                ) is not None
            ]
            if context_blocks:
                best_score = max(item[0] for item in context_blocks)
                best_blocks = [
                    (block, resolved)
                    for score, block, resolved in context_blocks
                    if score == best_score
                ]
            else:
                best_blocks = []
            if len(best_blocks) == 1:
                block, resolved = best_blocks[0]
                if not any(
                    evidence.get("block_id") == block.block_id
                    for evidence in series_evidence
                ):
                    series_evidence.append({
                        "block_id": block.block_id,
                        "source_sentence": _source_excerpt(
                            _element_source_text(block),
                            resolved,
                        ),
                        "table_locator": None,
                    })
                    repairs["series_context_evidence_supplemented"] += 1
        for point in series.get("points", []):
            if not isinstance(point, dict):
                continue
            for evidence in point.get("evidence", []):
                if not isinstance(evidence, dict):
                    continue
                block = block_map.get(evidence.get("block_id"))
                locator = evidence.get("table_locator")
                if (
                    block is not None
                    and block.type == "table"
                    and isinstance(locator, dict)
                    and locator.get("table_id") != block.block_id
                ):
                    locator["table_id"] = block.block_id
                    repairs["table_locator_ids_aligned_to_evidence"] += 1
                if repair_locator(
                    evidence,
                    row_hints=tuple(
                        str(coordinate.get(field) or "")
                        for coordinate in point.get("coordinates", [])
                        if isinstance(coordinate, dict)
                        for field in ("name_raw", "value_raw")
                    ),
                    column_hints=(
                        str(series.get("property_name_raw") or ""),
                    ),
                ):
                    repairs["table_locator_surfaces_repaired"] += 1
            repair_measurement_context(
                point.get("measurement_context"),
                [
                    evidence
                    for evidence in point.get("evidence", [])
                    if isinstance(evidence, dict)
                ],
            )
            for coordinate in point.get("coordinates", []):
                if not isinstance(coordinate, dict):
                    continue
                evidence = coordinate.get("evidence")
                if not isinstance(evidence, dict):
                    continue
                block = block_map.get(evidence.get("block_id"))
                locator = evidence.get("table_locator")
                if (
                    block is not None
                    and block.type == "table"
                    and isinstance(locator, dict)
                    and locator.get("table_id") != block.block_id
                ):
                    locator["table_id"] = block.block_id
                    repairs["table_locator_ids_aligned_to_evidence"] += 1
                coordinate_value = str(coordinate.get("value_raw") or "")
                if (
                    block is not None
                    and block.type == "table"
                    and (
                        not isinstance(evidence.get("table_locator"), dict)
                        or evidence["table_locator"].get("cell_value") is None
                    )
                ):
                    locator = evidence.get("table_locator")
                    if not isinstance(locator, dict):
                        locator = {
                            "table_id": block.block_id,
                            "row_label": coordinate_value,
                            "column_label": "",
                        }
                        evidence["table_locator"] = locator
                    locator["cell_value"] = coordinate_value
                    repairs["coordinate_table_locators_synthesized"] += 1
                if repair_locator(
                    evidence,
                    column_hints=(
                        str(coordinate.get("name_raw") or ""),
                        str(coordinate.get("unit_raw") or ""),
                    ),
                ):
                    repairs["table_locator_surfaces_repaired"] += 1
                locator = evidence.get("table_locator")
                block = block_map.get(evidence.get("block_id"))
                if not isinstance(locator, dict) or block is None:
                    continue
                if locator.get("cell_id") is None:
                    point_rows = {
                        point_evidence.get("table_locator", {}).get("row_index")
                        for point_evidence in point.get("evidence", [])
                        if isinstance(point_evidence, dict)
                        and point_evidence.get("block_id") == block.block_id
                        and isinstance(
                            point_evidence.get("table_locator"), dict
                        )
                        and point_evidence["table_locator"].get("row_index")
                        is not None
                    }
                    if len(point_rows) == 1:
                        point_row = next(iter(point_rows))
                        candidates = [
                            cell
                            for cell in table_cells_for(block)
                            if cell.row_index <= point_row
                            < cell.row_index + cell.row_span
                            and _resolve_surface_text(
                                cell.text,
                                coordinate_value,
                            ) is not None
                        ]
                        if len(candidates) == 1:
                            cell = candidates[0]
                            locator.update({
                                "cell_value": cell.text or None,
                                "cell_id": cell.cell_id,
                                "row_index": cell.row_index,
                                "column_index": cell.column_index,
                            })
                            repairs[
                                "coordinate_locators_aligned_to_point"
                            ] += 1
                source = _element_source_text(block)
                header = str(locator.get("column_label") or "")
                unit_match = re.search(r"\([^()]+\)", header)
                name_from_header = (
                    header[:unit_match.start()].strip()
                    if unit_match is not None
                    else header.strip()
                )
                if _resolve_surface_text(
                    source,
                    str(coordinate.get("name_raw") or ""),
                ) is None and name_from_header:
                    coordinate["name_raw"] = name_from_header
                unit_raw = coordinate.get("unit_raw")
                if (
                    isinstance(unit_raw, str)
                    and _resolve_surface_text(source, unit_raw) is None
                ):
                    coordinate["unit_raw"] = (
                        unit_match.group(0) if unit_match is not None else None
                    )
            coordinate_rows = {
                evidence.get("table_locator", {}).get("row_index")
                for coordinate in point.get("coordinates", [])
                if isinstance(coordinate, dict)
                and isinstance((evidence := coordinate.get("evidence")), dict)
                and isinstance(evidence.get("table_locator"), dict)
                and evidence["table_locator"].get("row_index") is not None
            }
            if len(coordinate_rows) == 1:
                coordinate_row = next(iter(coordinate_rows))
                for evidence in point.get("evidence", []):
                    if not isinstance(evidence, dict):
                        continue
                    locator = evidence.get("table_locator")
                    block = block_map.get(evidence.get("block_id"))
                    if (
                        not isinstance(locator, dict)
                        or block is None
                        or block.type != "table"
                    ):
                        continue
                    point_value = str(point.get("value_raw") or "").strip()
                    if point_value:
                        coordinate_cells = []
                        for coordinate in point.get("coordinates", []):
                            if not isinstance(coordinate, dict):
                                continue
                            coordinate_evidence = coordinate.get("evidence")
                            if (
                                not isinstance(coordinate_evidence, dict)
                                or coordinate_evidence.get("block_id")
                                != block.block_id
                            ):
                                continue
                            coordinate_locator = coordinate_evidence.get(
                                "table_locator"
                            )
                            if not isinstance(coordinate_locator, dict):
                                continue
                            coordinate_cell_id = coordinate_locator.get(
                                "cell_id"
                            )
                            coordinate_cells.extend(
                                cell
                                for cell in table_cells_for(block)
                                if cell.cell_id == coordinate_cell_id
                            )
                        covered_rows = {
                            row
                            for cell in coordinate_cells
                            for row in range(
                                cell.row_index,
                                cell.row_index + cell.row_span,
                            )
                        }
                        exact_candidates = [
                            cell
                            for cell in table_cells_for(block)
                            if cell.row_index in covered_rows
                            and cell.text.strip() == point_value
                        ]
                        spans_multiple_rows = any(
                            cell.row_span > 1 for cell in coordinate_cells
                        )
                        if spans_multiple_rows and len(exact_candidates) == 1:
                            cell = exact_candidates[0]
                            if locator.get("cell_id") != cell.cell_id:
                                locator.update({
                                    "cell_value": cell.text or None,
                                    "cell_id": cell.cell_id,
                                    "row_index": cell.row_index,
                                    "column_index": cell.column_index,
                                })
                                repairs[
                                    "point_compound_locators_aligned_to_coordinates"
                                ] += 1
                            continue
                    if (
                        locator.get("cell_id") is not None
                        and locator.get("row_index") == coordinate_row
                    ):
                        continue
                    candidates = [
                        cell
                        for cell in table_cells_for(block)
                        if cell.row_index == coordinate_row
                        and _resolve_surface_text(
                            cell.text,
                            str(locator.get("cell_value") or ""),
                        ) is not None
                    ]
                    if len(candidates) != 1:
                        continue
                    cell = candidates[0]
                    locator.update({
                        "cell_value": cell.text or None,
                        "cell_id": cell.cell_id,
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                    })
                    repairs["point_locators_aligned_to_coordinates"] += 1

    def unit_surface_key(value: str) -> str:
        projected, _ = _surface_projection(value, compact_math=True)
        return projected.replace("(", "").replace(")", "")

    def repair_series_unit_surface(
        item: dict[str, Any],
        evidence_items: list[dict[str, Any]],
        property_name: str,
    ) -> None:
        unit_raw = item.get("unit_raw")
        if not isinstance(unit_raw, str) or not unit_raw.strip():
            return
        sources = [
            _element_source_text(block_map[evidence.get("block_id")])
            for evidence in evidence_items
            if evidence.get("block_id") in block_map
        ]
        if any(
            _resolve_surface_text(source, unit_raw) is not None
            for source in sources
        ):
            return
        target_key = unit_surface_key(unit_raw)
        candidates: set[str] = set()
        for evidence in evidence_items:
            locator = evidence.get("table_locator")
            if not isinstance(locator, dict):
                continue
            header = locator.get("column_label")
            if not isinstance(header, str):
                continue
            resolved_name = _resolve_surface_text(header, property_name)
            if resolved_name is None:
                continue
            position = header.find(resolved_name)
            if position < 0:
                continue
            suffix = header[position + len(resolved_name):].strip()
            if suffix and unit_surface_key(suffix) == target_key:
                candidates.add(suffix)
        if len(candidates) == 1:
            item["unit_raw"] = next(iter(candidates))
            repairs["series_unit_surfaces_repaired"] += 1

    for series in series_items:
        property_name = series.get("property_name_raw")
        if not isinstance(property_name, str) or not property_name:
            continue
        series_evidence = [
            evidence
            for evidence in series.get("evidence", [])
            if isinstance(evidence, dict)
        ]
        point_evidence = [
            evidence
            for point in series.get("points", [])
            if isinstance(point, dict)
            for evidence in point.get("evidence", [])
            if isinstance(evidence, dict)
        ]
        repair_series_unit_surface(
            series,
            [*series_evidence, *point_evidence],
            property_name,
        )
        for point in series.get("points", []):
            if not isinstance(point, dict):
                continue
            repair_series_unit_surface(
                point,
                [
                    *series_evidence,
                    *[
                        evidence
                        for evidence in point.get("evidence", [])
                        if isinstance(evidence, dict)
                    ],
                ],
                property_name,
            )

    def candidate_unit_keys(item: dict[str, Any]) -> set[str]:
        values = [item.get("unit_normalized"), item.get("unit_raw")]
        for point in item.get("points", []):
            if isinstance(point, dict):
                values.extend([
                    point.get("unit_normalized"),
                    point.get("unit_raw"),
                ])
        keys = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            projected, _ = _surface_projection(value, compact_math=True)
            if projected:
                keys.add(projected)
        return keys

    def candidate_evidence_blocks(item: dict[str, Any]) -> set[str]:
        block_ids = {
            evidence.get("block_id")
            for evidence in item.get("evidence", [])
            if isinstance(evidence, dict)
            and isinstance(evidence.get("block_id"), str)
        }
        for point in item.get("points", []):
            if not isinstance(point, dict):
                continue
            block_ids.update(
                evidence.get("block_id")
                for evidence in point.get("evidence", [])
                if isinstance(evidence, dict)
                and isinstance(evidence.get("block_id"), str)
            )
            for coordinate in point.get("coordinates", []):
                if not isinstance(coordinate, dict):
                    continue
                evidence = coordinate.get("evidence")
                if (
                    isinstance(evidence, dict)
                    and isinstance(evidence.get("block_id"), str)
                ):
                    block_ids.add(evidence["block_id"])
        return block_ids

    def compatible_series(item: dict[str, Any]) -> list[dict[str, Any]]:
        property_matches = [
            series
            for series in series_items
            if _candidate_property_key(series) == _candidate_property_key(item)
        ]

        def same_subject(series: dict[str, Any]) -> bool:
            item_sample = item.get("sample_id")
            series_sample = series.get("sample_id")
            if isinstance(item_sample, str) and isinstance(series_sample, str):
                return item_sample == series_sample
            item_entity = (
                sample_entities.get(item_sample)
                if isinstance(item_sample, str)
                else item.get("entity_id")
            )
            series_entity = (
                sample_entities.get(series_sample)
                if isinstance(series_sample, str)
                else series.get("entity_id")
            )
            return (
                isinstance(item_entity, str)
                and item_entity == series_entity
            )

        property_matches = [
            series for series in property_matches if same_subject(series)
        ]
        item_units = candidate_unit_keys(item)
        item_blocks = candidate_evidence_blocks(item)
        item_group = item.get("observation_group_id")
        compatible = []
        for series in property_matches:
            series_units = candidate_unit_keys(series)
            if item_units and series_units and not item_units & series_units:
                continue
            series_group = series.get("observation_group_id")
            same_group = (
                isinstance(item_group, str)
                and item_group == series_group
            )
            if not same_group and not (
                item_blocks & candidate_evidence_blocks(series)
            ):
                continue
            compatible.append(series)
        return compatible

    def range_compatible_series(item: dict[str, Any]) -> list[dict[str, Any]]:
        if item.get("observation_role") != "aggregate":
            return []
        value_min = item.get("value_min")
        value_max = item.get("value_max")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (value_min, value_max)
        ):
            return []
        item_units = candidate_unit_keys(item)
        matches: list[dict[str, Any]] = []
        for series in series_items:
            if _candidate_property_key(series) != _candidate_property_key(item):
                continue
            if any(
                item.get(field) is not None
                and series.get(field) is not None
                and item.get(field) != series.get(field)
                for field in ("property_code", "property_category")
            ):
                continue
            series_units = candidate_unit_keys(series)
            if item_units and series_units and not item_units & series_units:
                continue
            values = [
                point.get("value_min")
                for point in series.get("points", [])
                if isinstance(point, dict)
                and isinstance(point.get("value_min"), (int, float))
                and not isinstance(point.get("value_min"), bool)
            ] + [
                point.get("value_max")
                for point in series.get("points", [])
                if isinstance(point, dict)
                and isinstance(point.get("value_max"), (int, float))
                and not isinstance(point.get("value_max"), bool)
            ]
            if values and min(values) == value_min and max(values) == value_max:
                matches.append(series)
        return matches

    for collection_name in ("properties", "unresolved_properties"):
        for item in payload.get(collection_name, []):
            if (
                not isinstance(item, dict)
                or item.get("observation_role") != "aggregate"
            ):
                continue
            matching_series = compatible_series(item)
            linked_by_range = False
            if not matching_series:
                range_matches = range_compatible_series(item)
                if len(range_matches) == 1:
                    matching_series = range_matches
                    linked_by_range = True
            matching_ids = list(dict.fromkeys(
                series.get("series_id")
                for series in matching_series
                if isinstance(series.get("series_id"), str)
            ))
            matching_id_set = set(matching_ids)
            current = item.get("series_id")
            current_multiple = item.get("series_ids")
            if isinstance(current_multiple, list):
                if not set(current_multiple) <= matching_id_set:
                    raise ValueError(
                        "aggregate series_ids 包含不兼容的 PropertySeries"
                    )
                continue
            if len(matching_ids) == 1:
                target = matching_ids[0]
                if current != target:
                    item["series_id"] = target
                    _mark_candidate_relation_uncertain(item, "series_id")
                    repairs["aggregate_linked"] += 1
                    if linked_by_range:
                        repairs["aggregate_range_linked"] += 1
            elif len(matching_ids) >= 2:
                item_group = item.get("observation_group_id")
                explicitly_grouped = (
                    isinstance(item_group, str)
                    and all(
                        series.get("observation_group_id") == item_group
                        for series in matching_series
                    )
                )
                if explicitly_grouped:
                    item.pop("series_id", None)
                    item["series_ids"] = matching_ids
                    _mark_candidate_relation_uncertain(item, "series_ids")
                    repairs["aggregate_multi_linked"] += 1

    resolved_aggregates = [
        item
        for item in payload.get("properties", [])
        if isinstance(item, dict)
        and item.get("observation_role") == "aggregate"
        and (
            isinstance(item.get("series_id"), str)
            or isinstance(item.get("series_ids"), list)
        )
    ]
    retained_unresolved = []
    for item in payload.get("unresolved_properties", []):
        if (
            not isinstance(item, dict)
            or item.get("observation_role") != "aggregate"
            or isinstance(item.get("series_id"), str)
            or isinstance(item.get("series_ids"), list)
        ):
            retained_unresolved.append(item)
            continue
        item_entity = item.get("entity_id")
        duplicates = [
            resolved
            for resolved in resolved_aggregates
            if isinstance(resolved.get("sample_id"), str)
            and sample_entities.get(resolved["sample_id"]) == item_entity
            and _normalized_table_label(str(
                resolved.get("property_name_raw") or ""
            )) == _normalized_table_label(str(
                item.get("property_name_raw") or ""
            ))
            and resolved.get("value_raw") == item.get("value_raw")
            and candidate_unit_keys(resolved) == candidate_unit_keys(item)
            and candidate_evidence_blocks(resolved)
            == candidate_evidence_blocks(item)
        ]
        if len(duplicates) == 1:
            repairs["duplicate_unresolved_aggregates_removed"] += 1
            continue
        retained_unresolved.append(item)
    payload["unresolved_properties"] = retained_unresolved

    if preview_relaxed:
        for series in payload.get("property_series", []):
            if not isinstance(series, dict):
                continue
            retained_points = []
            for point in series.get("points", []):
                if not isinstance(point, dict):
                    continue
                unstable_composite_locator = False
                for evidence in point.get("evidence", []):
                    if not isinstance(evidence, dict):
                        continue
                    locator = evidence.get("table_locator")
                    block = block_map.get(evidence.get("block_id"))
                    if (
                        not isinstance(locator, dict)
                        or block is None
                        or block.type != "table"
                        or locator.get("cell_id") is not None
                    ):
                        continue
                    exact_cells = {
                        cell.text.strip()
                        for cell in table_cells_for(block)
                        if cell.text.strip()
                    }
                    if locator.get("row_label") not in exact_cells:
                        unstable_composite_locator = True
                        break
                if unstable_composite_locator:
                    repairs["preview_unstable_series_points_removed"] += 1
                    continue
                retained_points.append(point)
                evidence_items = [
                    *[
                        evidence
                        for evidence in point.get("evidence", [])
                        if isinstance(evidence, dict)
                    ],
                    *[
                        coordinate.get("evidence")
                        for coordinate in point.get("coordinates", [])
                        if isinstance(coordinate, dict)
                        and isinstance(coordinate.get("evidence"), dict)
                    ],
                ]
                for evidence in evidence_items:
                    locator = evidence.get("table_locator")
                    block = block_map.get(evidence.get("block_id"))
                    if (
                        not isinstance(locator, dict)
                        or block is None
                        or block.type != "table"
                        or not isinstance(locator.get("row_index"), int)
                    ):
                        continue
                    exact_cells = {
                        cell.text.strip()
                        for cell in table_cells_for(block)
                        if cell.text.strip()
                    }
                    if locator.get("row_label") in exact_cells:
                        continue
                    row_cells = [
                        cell
                        for cell in table_cells_for(block)
                        if cell.row_index <= locator["row_index"]
                        < cell.row_index + cell.row_span
                        and cell.text.strip()
                    ]
                    if row_cells:
                        locator["row_label"] = min(
                            row_cells,
                            key=lambda cell: cell.column_index,
                        ).text.strip()
                        repairs["table_locator_surfaces_repaired"] += 1
            series["points"] = retained_points

        series_cell_ids = {
            locator.get("cell_id")
            for series in payload.get("property_series", [])
            if isinstance(series, dict)
            for point in series.get("points", [])
            if isinstance(point, dict)
            for evidence in point.get("evidence", [])
            if isinstance(evidence, dict)
            and isinstance((locator := evidence.get("table_locator")), dict)
            and locator.get("cell_id") is not None
        }
        retained_properties = []
        for item in payload.get("properties", []):
            if not isinstance(item, dict):
                continue
            scalar_cells = {
                locator.get("cell_id")
                for evidence in item.get("evidence", [])
                if isinstance(evidence, dict)
                and isinstance(
                    (locator := evidence.get("table_locator")), dict
                )
                and locator.get("cell_id") is not None
            }
            if scalar_cells & series_cell_ids:
                repairs[
                    "preview_scalar_series_cell_duplicates_removed"
                ] += 1
                continue
            retained_properties.append(item)
        payload["properties"] = retained_properties

    if preview_relaxed:
        def degrade_blank_table_locators(value: Any) -> None:
            if isinstance(value, dict):
                locator = value.get("table_locator")
                if isinstance(locator, dict) and any(
                    isinstance(locator.get(field), str)
                    and not locator[field].strip()
                    for field in ("row_label", "column_label")
                ):
                    value["table_locator"] = None
                    repairs["preview_blank_table_locators_degraded"] += 1
                for child in value.values():
                    degrade_blank_table_locators(child)
            elif isinstance(value, list):
                for child in value:
                    degrade_blank_table_locators(child)

        degrade_blank_table_locators(payload)

    # 锚定修复可能再次清空 reported condition，返回前必须再降级一次。
    downgrade_empty_reported_context(payload)

    for series in series_items:
        series_confidence = series.get("confidence")
        if not isinstance(series_confidence, dict):
            points = series.get("points")
            point_scores = [
                point.get("confidence", {}).get("score")
                for point in points
                if isinstance(point, dict)
                and isinstance(point.get("confidence"), dict)
            ] if isinstance(points, list) else []
            if (
                points
                and len(point_scores) == len(points)
                and all(
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and 0 <= score <= 1
                    for score in point_scores
                )
            ):
                series_confidence = {"score": min(point_scores)}
                series["confidence"] = series_confidence
                repairs["series_confidence_inherited_from_points"] += 1
        if not isinstance(series_confidence, dict):
            continue
        inherited = 0
        for point in series.get("points", []):
            if (
                not isinstance(point, dict)
                or isinstance(point.get("confidence"), dict)
            ):
                continue
            point["confidence"] = {
                "score": min(float(series_confidence.get("score", 0.5)), 0.5)
            }
            inherited += 1
        if inherited:
            repairs["point_confidence_inherited"] += inherited
            repairs["series_with_inherited_confidence"] += 1
    return payload, repairs


def _candidate_repair_warnings(
    repairs: dict[str, int],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if repairs["preview_unresolved_controlled_fields_cleared"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_unresolved_controlled_fields_cleared",
            "message": (
                "Preview 模式已清空 unresolved property 中仅 resolved "
                "条目允许的受控名称与代码；raw 值和 evidence 保持不变"
            ),
            "properties": repairs[
                "preview_unresolved_controlled_fields_cleared"
            ],
        })
    if repairs["preview_unscoped_scalar_properties_removed"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_unscoped_scalar_properties_removed",
            "message": (
                "Preview 模式已移除没有合法 sample_id 的 resolved scalar property；"
                "未猜测主体"
            ),
            "properties": repairs[
                "preview_unscoped_scalar_properties_removed"
            ],
        })
    if repairs["preview_missing_points_recovered_from_unique_rows"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_missing_point_recovered_from_unique_table_row",
            "message": (
                "Preview 模式已用唯一完整表格行与唯一列头恢复误标为 missing "
                "的非空 Series point，并降低 confidence"
            ),
            "points": repairs[
                "preview_missing_points_recovered_from_unique_rows"
            ],
        })
    if repairs[
        "preview_missing_point_locators_recovered_from_unique_rows"
    ]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_missing_point_locator_recovered_from_unique_table_row",
            "message": (
                "Preview 模式已用唯一完整表格行与唯一列头补全 missing "
                "Series point 的空单元格稳定坐标；missing 状态保持不变"
            ),
            "points": repairs[
                "preview_missing_point_locators_recovered_from_unique_rows"
            ],
        })
    if repairs["preview_point_locators_recovered_from_unique_rows"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_point_locator_recovered_from_unique_table_row",
            "message": (
                "Preview 模式已用唯一完整表格行、唯一列头及一致 value_raw "
                "补全 Series point 的稳定坐标"
            ),
            "points": repairs[
                "preview_point_locators_recovered_from_unique_rows"
            ],
        })
    if repairs["preview_unresolved_series_points_synthesized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_unresolved_series_points_synthesized",
            "message": (
                "Preview 模式已补入 unresolved Series 同一表格列、已有行范围内"
                "唯一遗漏的非空数值单元格；confidence 降为 0.5"
            ),
            "points": repairs["preview_unresolved_series_points_synthesized"],
        })
    if repairs["series_unit_surfaces_repaired"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_unit_surface_repaired",
            "message": (
                "Series/point 的 unit_raw 缺少表头括号；已从同一表格"
                "列名中恢复唯一等价的原文单位表面文本"
            ),
            "fields": repairs["series_unit_surfaces_repaired"],
        })
    if (
        repairs["series_entity_relinked_to_sample"]
        or repairs["point_entity_relinked_to_sample"]
    ):
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_entity_relinked_to_sample",
            "message": (
                "PropertySeries/point 的 entity_id 与已解析 Sample 冲突；"
                "已按 Stage 3 Sample.refers_to_entity 确定性纠正"
            ),
            "series": repairs["series_entity_relinked_to_sample"],
            "points": repairs["point_entity_relinked_to_sample"],
        })
    if repairs["multi_subject_series_normalized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "multi_subject_series_normalized",
            "message": (
                "PropertySeries 的 points 明确属于多个 Sample/PolymerEntity；"
                "已清除错误的单一顶层主体，主体关系保留在各 point"
            ),
            "series": repairs["multi_subject_series_normalized"],
        })
    if repairs["single_subject_series_inherited_from_points"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "single_subject_series_inherited_from_points",
            "message": (
                "PropertySeries 顶层主体缺失；已从全部 point 的唯一共同主体补全"
            ),
            "series": repairs["single_subject_series_inherited_from_points"],
        })
    if repairs["empty_reported_contexts_downgraded"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "empty_reported_context_downgraded",
            "message": (
                "measurement_context 标记为 reported 但没有任何条件值；"
                "已确定性改为 not_reported"
            ),
            "contexts": repairs["empty_reported_contexts_downgraded"],
        })
    if repairs["aggregate_linked"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "candidate_aggregate_relation_repaired",
            "message": (
                "已按性质、Sample/PolymerEntity、单位与 evidence 的"
                "唯一兼容关系补充 aggregate series_id"
            ),
            "linked": repairs["aggregate_linked"],
        })
    if repairs["aggregate_range_linked"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "candidate_aggregate_range_relation_repaired",
            "message": (
                "aggregate 的性质、代码、类别和单位一致，且数值上下界"
                "与唯一 PropertySeries 的有效 points 边界完全一致；"
                "已补充 series_id"
            ),
            "linked": repairs["aggregate_range_linked"],
        })
    if repairs["duplicate_unresolved_aggregates_removed"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "duplicate_unresolved_aggregates_removed",
            "message": (
                "unresolved aggregate 与已解析 aggregate 的主体、名称、"
                "值、单位和证据完全相同；已删除重复表达"
            ),
            "removed": repairs["duplicate_unresolved_aggregates_removed"],
        })
    if repairs["aggregate_multi_linked"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "candidate_aggregate_multi_relation_repaired",
            "message": (
                "已按性质、Sample/PolymerEntity、单位、evidence 与明确 "
                "observation_group_id 补充 aggregate series_ids"
            ),
            "linked": repairs["aggregate_multi_linked"],
        })
    if repairs["point_confidence_inherited"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_point_confidence_inherited",
            "message": (
                "模型未逐点重复 confidence；已保守继承 Series confidence，"
                "并标记 indirect_relation"
            ),
            "points": repairs["point_confidence_inherited"],
            "series": repairs["series_with_inherited_confidence"],
        })
    if repairs["series_confidence_inherited_from_points"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_confidence_inherited_from_points",
            "message": (
                "模型遗漏 Series confidence；已取该 Series 全部 point "
                "confidence.score 的最低值"
            ),
            "series": repairs["series_confidence_inherited_from_points"],
        })
    if repairs["property_names_mapped_from_code_category"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "property_names_mapped_from_code_category",
            "message": (
                "模型将 property_name_normalized 输出为非词表名称；"
                "已按唯一的 property_code 与 property_category 映射到规范键"
            ),
            "properties": repairs["property_names_mapped_from_code_category"],
        })
    if repairs["table_locator_ids_aligned_to_evidence"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "table_locator_ids_aligned_to_evidence",
            "message": (
                "模型将论文表格显示名填入 table_locator.table_id；"
                "已对齐到 evidence 引用的 Stage 0 表格 block_id"
            ),
            "evidence": repairs["table_locator_ids_aligned_to_evidence"],
        })
    if repairs["table_locator_surfaces_repaired"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "candidate_table_locator_surface_repaired",
            "message": (
                "模型 locator 表面文本不在表格原文；已用 property raw 与"
                " determination method 中唯一匹配的 Stage 0 表头恢复"
            ),
            "evidence": repairs["table_locator_surfaces_repaired"],
        })
    if repairs["blank_table_cell_values_normalized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "blank_table_cell_values_normalized",
            "message": (
                "空单元格 locator 的 cell_value 已规范为 null，"
                "保留表格行列定位"
            ),
            "evidence": repairs["blank_table_cell_values_normalized"],
        })
    if repairs["coordinate_table_locators_synthesized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "coordinate_table_locators_synthesized",
            "message": "已按 coordinate 原文值与表头生成稳定单元格定位",
            "evidence": repairs["coordinate_table_locators_synthesized"],
        })
    if repairs["coordinate_locators_aligned_to_point"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "coordinate_locators_aligned_to_point",
            "message": (
                "重复 coordinate 标签已按同一 point 的唯一性质值行"
                "绑定到对应单元格（支持 rowspan）"
            ),
            "evidence": repairs["coordinate_locators_aligned_to_point"],
        })
    if repairs["condition_table_locators_synthesized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "condition_table_locators_synthesized",
            "message": "已将唯一包含条件 raw 的表头单元格绑定为条件 evidence",
            "evidence": repairs["condition_table_locators_synthesized"],
        })
    if repairs["condition_evidence_surfaces_repaired"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "condition_evidence_surfaces_repaired",
            "message": (
                "条件 evidence 的候选摘要无法逐字定位；"
                "已用同一条件下性质字段的原文 anchor 恢复"
            ),
            "evidence": repairs["condition_evidence_surfaces_repaired"],
        })
    if repairs["singleton_condition_evidence_unwrapped"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "singleton_condition_evidence_unwrapped",
            "message": (
                "MeasurementCondition evidence 误输出为单项数组；"
                "已确定性解包为唯一 evidence 对象"
            ),
            "conditions": repairs["singleton_condition_evidence_unwrapped"],
        })
    if repairs["embedded_measurement_conditions_promoted"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "embedded_measurement_conditions_promoted",
            "message": (
                "property 引用了未定义 condition，但携带一致且有逐字证据的 "
                "measurement_context；已确定性提升为顶层 MeasurementCondition"
            ),
            "conditions": repairs[
                "embedded_measurement_conditions_promoted"
            ],
        })
    if repairs["missing_conditions_marked_not_reported"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "missing_conditions_marked_not_reported",
            "message": (
                "property 引用了未定义 condition 且未提供条件内容；"
                "已在唯一共同性质证据下明确标记为 not_reported"
            ),
            "conditions": repairs[
                "missing_conditions_marked_not_reported"
            ],
        })
    if repairs["confidence_paths_normalized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_paths_normalized",
            "message": "confidence 字段路径中的数组标记 [] 已规范为 Schema 字段路径",
            "fields": repairs["confidence_paths_normalized"],
        })
    if repairs["confidence_field_aliases_normalized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_field_aliases_normalized",
            "message": (
                "confidence 使用了明确旧字段别名；"
                "已将 value/unit 规范为 value_raw/unit_raw"
            ),
            "fields": repairs["confidence_field_aliases_normalized"],
        })
    if repairs["confidence_field_descriptions_normalized"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_field_descriptions_normalized",
            "message": (
                "confidence 字段路径附带了说明性 for 后缀；"
                "已在唯一合法 sample_id 前缀下规范化"
            ),
            "fields": repairs["confidence_field_descriptions_normalized"],
        })
    if repairs["redundant_confidence_descriptions_removed"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "redundant_confidence_descriptions_removed",
            "message": (
                "confidence uncertain_fields 中的非字段关联说明已由 "
                "indirect_relation 原因码完整表达；已删除重复说明元数据"
            ),
            "fields": repairs["redundant_confidence_descriptions_removed"],
        })
    if repairs["series_point_confidence_fields_removed"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_point_confidence_fields_removed",
            "message": (
                "Series confidence 误引用了 point-only 字段；"
                "已删除无效的 value_min/value_max 元数据"
            ),
            "fields": repairs["series_point_confidence_fields_removed"],
        })
    if repairs["point_locators_aligned_to_coordinates"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "point_locators_aligned_to_coordinates",
            "message": "重复值单元格已按同一 point 的唯一坐标行绑定",
            "evidence": repairs["point_locators_aligned_to_coordinates"],
        })
    if repairs["point_compound_locators_aligned_to_coordinates"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "point_compound_locators_aligned_to_coordinates",
            "message": (
                "point locator 指向包含目标值的复合单元格；已按 coordinate "
                "rowspan 范围内唯一精确值单元格重新绑定"
            ),
            "evidence": repairs[
                "point_compound_locators_aligned_to_coordinates"
            ],
        })
    if repairs["measurement_context_surfaces_repaired"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "candidate_measurement_context_surface_repaired",
            "message": (
                "measurement context raw 仅在 pm/mp 符号上与原文不同；"
                "已恢复为同一 evidence block 的原文形式"
            ),
            "fields": repairs["measurement_context_surfaces_repaired"],
        })
    if repairs["series_method_evidence_supplemented"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_method_evidence_supplemented",
            "message": (
                "Series determination method 不在 point 表格 evidence；"
                "已补入全文中唯一的完全匹配原文 block"
            ),
            "series": repairs["series_method_evidence_supplemented"],
        })
    if repairs["series_multimethod_downgraded"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_multimethod_downgraded",
            "message": (
                "Series 跨多种测定方法，无法保存为单一逐字 raw 字段；"
                "已置 null 并保留方法原文证据"
            ),
            "series": repairs["series_multimethod_downgraded"],
        })
    if repairs["series_context_evidence_supplemented"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "series_context_evidence_supplemented",
            "message": (
                "Series/point measurement context 不在数值表格 evidence；"
                "已补入全文中唯一的完全匹配原文 block"
            ),
            "series_evidence": repairs[
                "series_context_evidence_supplemented"
            ],
        })
    if repairs["preview_series_unsupported_fields_removed"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_series_unsupported_fields_removed",
            "message": (
                "PropertySeries 包含仅适用于标量性质的字段；"
                "Preview 已删除该字段，Strict 模式仍会报错"
            ),
            "fields": repairs["preview_series_unsupported_fields_removed"],
        })
    if repairs["preview_singleton_coordinate_evidence_unwrapped"]:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_singleton_coordinate_evidence_unwrapped",
            "message": (
                "PropertySeriesCoordinate evidence 误输出为单项数组；"
                "Preview 已确定性解包为唯一 evidence 对象，Strict 模式仍会报错"
            ),
            "coordinates": repairs[
                "preview_singleton_coordinate_evidence_unwrapped"
            ],
        })
    return warnings


def _validate_response(
    response: LLMJSONResponse,
    entities: Stage2Document,
    process: Stage3Document,
    blocks: list[Stage0Element],
    vocabulary: dict[str, tuple[str, str]],
    *,
    preview_relaxed: bool = False,
) -> tuple[
    PropertyStageResponse,
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[str],
    list[dict[str, Any]],
]:
    cleaned_data, _ = compact_confidence_payload(response.data)
    parsed = PropertyStageResponse.model_validate(cleaned_data)
    entity_ids = {item.entity_id for item in entities.polymer_entities}
    sample_ids = {item.sample_id for item in process.samples}
    block_map = {block.block_id: block for block in blocks}

    referenced_condition_ids = {
        item.measurement_condition_id for item in parsed.properties
    }
    dropped_condition_ids = [
        item.condition_id
        for item in parsed.measurement_conditions
        if item.condition_id not in referenced_condition_ids
    ]
    conditions = []
    degraded_condition_evidence: list[tuple[str, str]] = []
    supplemented_condition_evidence: list[tuple[str, str]] = []
    properties_by_condition: dict[
        str, list[PropertyObservationCandidate]
    ] = {}
    for item in parsed.properties:
        properties_by_condition.setdefault(
            item.measurement_condition_id,
            [],
        ).append(item)
    for item in parsed.measurement_conditions:
        if item.condition_id not in referenced_condition_ids:
            continue
        normalized, degraded, supplemented = _normalize_condition(
            item,
            block_map,
            properties_by_condition.get(item.condition_id, []),
        )
        conditions.append(normalized)
        if degraded:
            degraded_condition_evidence.append(
                (item.condition_id, item.evidence.block_id)
            )
        if supplemented:
            supplemented_condition_evidence.append(
                (item.condition_id, normalized.evidence.block_id)
            )
    properties = []
    dropped_table_evidence: list[tuple[str, str]] = []
    dropped_unanchored_evidence: list[tuple[str, str]] = []
    degraded_table_evidence: list[tuple[str, str]] = []
    supplemented_property_evidence: list[tuple[str, str]] = []
    seen_properties: set[tuple[str, str, str, str, str | None]] = set()
    for item in parsed.properties:
        if item.sample_id not in sample_ids:
            raise ValueError(
                f"{item.property_id} 引用了未知 sample：{item.sample_id}"
            )
        (
            normalized,
            dropped_table_blocks,
            dropped_unanchored_blocks,
            degraded_table_blocks,
            supplemented_blocks,
        ) = _normalize_property(item, block_map, vocabulary)
        dropped_table_evidence.extend(
            (item.property_id, block_id)
            for block_id in dropped_table_blocks
        )
        dropped_unanchored_evidence.extend(
            (item.property_id, block_id)
            for block_id in dropped_unanchored_blocks
        )
        degraded_table_evidence.extend(
            (item.property_id, block_id)
            for block_id in degraded_table_blocks
        )
        supplemented_property_evidence.extend(
            (item.property_id, block_id)
            for block_id in supplemented_blocks
        )
        key = (
            normalized.sample_id,
            normalized.property_name_raw.casefold(),
            normalized.value_raw,
            normalized.measurement_condition_id,
            (
                normalized.determination_method_raw.casefold()
                if normalized.determination_method_raw
                else None
            ),
        )
        if key in seen_properties:
            raise ValueError("存在重复 PropertyObservation")
        seen_properties.add(key)
        properties.append(normalized)

    unresolved = []
    degraded_unresolved_evidence: list[tuple[str, str]] = []
    supplemented_unresolved_evidence: list[tuple[str, str]] = []
    for item in parsed.unresolved_properties:
        if item.entity_id not in entity_ids:
            raise ValueError(
                f"{item.unresolved_id} 引用了未知 entity：{item.entity_id}"
            )
        (
            normalized,
            dropped_table_blocks,
            dropped_unanchored_blocks,
            degraded_table_blocks,
            supplemented_blocks,
        ) = _normalize_unresolved(item, block_map)
        unresolved.append(normalized)
        degraded_unresolved_evidence.extend(
            (item.unresolved_id, block_id)
            for block_id in degraded_table_blocks
        )
        supplemented_unresolved_evidence.extend(
            (item.unresolved_id, block_id)
            for block_id in supplemented_blocks
        )
    property_series = []
    for item in parsed.property_series:
        if item.sample_id and item.sample_id not in sample_ids:
            raise ValueError(
                f"{item.series_id} 引用了未知 sample：{item.sample_id}"
            )
        if item.entity_id and item.entity_id not in entity_ids:
            raise ValueError(
                f"{item.series_id} 引用了未知 entity：{item.entity_id}"
            )
        normalized = _normalize_series(
            item,
            block_map,
            vocabulary,
            allow_incomplete_coverage=preview_relaxed,
        )
        for point in normalized.points:
            if point.sample_id and point.sample_id not in sample_ids:
                raise ValueError(
                    f"{point.point_id} 引用了未知 sample：{point.sample_id}"
                )
            if point.entity_id and point.entity_id not in entity_ids:
                raise ValueError(
                    f"{point.point_id} 引用了未知 entity：{point.entity_id}"
                )
        property_series.append(normalized)

    coordinate_only_table_columns = _validate_required_table_series(
        blocks,
        property_series,
        allow_missing=preview_relaxed,
    )

    scalar_cells = set().union(*(
        _stable_value_cells(item.evidence)
        for item in [*properties, *unresolved]
    )) if (properties or unresolved) else set()
    series_cells = {
        cell_id
        for series in property_series
        for point in series.points
        for cell_id in _stable_value_cells(point.evidence)
    }
    duplicate_cells = sorted(scalar_cells & series_cells)
    if duplicate_cells:
        raise ValueError(
            "同一 table cell 不得同时输出 scalar property 和 series point："
            f"{duplicate_cells}"
        )
    return (
        parsed.model_copy(update={
            "measurement_conditions": conditions,
            "properties": properties,
            "unresolved_properties": unresolved,
            "property_series": property_series,
        }),
        dropped_table_evidence,
        dropped_unanchored_evidence,
        degraded_table_evidence,
        degraded_condition_evidence,
        supplemented_condition_evidence,
        supplemented_property_evidence,
        degraded_unresolved_evidence,
        supplemented_unresolved_evidence,
        dropped_condition_ids,
        coordinate_only_table_columns,
    )


def _materialize_evidence(
    item: PropertyEvidenceCandidate,
    block_map: dict[str, Stage0Element],
) -> Evidence:
    block = block_map[item.block_id]
    return Evidence(
        block_id=block.block_id,
        page=block.page,
        bbox=block.bbox,
        source_type=block.type,
        source_sentence=item.source_sentence,
        table_locator=(
            item.table_locator.model_dump(mode="json")
            if item.table_locator is not None
            else None
        ),
    )


def _materialize_condition_quantity(
    item: ConditionQuantityCandidate | None,
    block_map: dict[str, Stage0Element],
) -> ConditionQuantity | None:
    if item is None:
        return None
    return ConditionQuantity(
        raw=item.raw,
        value=item.value,
        unit=item.unit,
        evidence=[
            _materialize_evidence(evidence, block_map)
            for evidence in item.evidence
        ],
    )


def _materialize_measurement_context(
    item: MeasurementContextCandidate | MeasurementConditionCandidate,
    block_map: dict[str, Stage0Element],
) -> MeasurementContext:
    return MeasurementContext(
        temperature=_materialize_condition_quantity(
            item.temperature, block_map
        ),
        frequency=_materialize_condition_quantity(item.frequency, block_map),
        humidity=_materialize_condition_quantity(item.humidity, block_map),
        pressure=_materialize_condition_quantity(item.pressure, block_map),
        wavelength=_materialize_condition_quantity(
            item.wavelength, block_map
        ),
        other_conditions=item.other_conditions,
        other_condition_evidence={
            key: [
                _materialize_evidence(evidence, block_map)
                for evidence in candidates
            ]
            for key, candidates in item.other_condition_evidence.items()
        },
        condition_status=item.condition_status,
    )


def _materialize(
    parsed: PropertyStageResponse,
    blocks: list[Stage0Element],
) -> tuple[
    list[MeasurementCondition],
    list[PropertyObservation],
    list[UnresolvedPropertyObservation],
    list[PropertySeries],
]:
    block_map = {block.block_id: block for block in blocks}
    condition_id_map = {
        item.condition_id: f"mc{index:03d}"
        for index, item in enumerate(parsed.measurement_conditions, start=1)
    }
    group_id_map = {
        group_id: f"pog{index:03d}"
        for index, group_id in enumerate(
            dict.fromkeys(
                item.observation_group_id
                for item in (
                    parsed.properties
                    + parsed.unresolved_properties
                    + parsed.property_series
                )
                if item.observation_group_id is not None
            ),
            start=1,
        )
    }
    conditions = []
    for item in parsed.measurement_conditions:
        context = _materialize_measurement_context(item, block_map)
        conditions.append(MeasurementCondition(
            condition_id=condition_id_map[item.condition_id],
            temperature=context.temperature,
            frequency=context.frequency,
            humidity=context.humidity,
            pressure=context.pressure,
            wavelength=context.wavelength,
            other_conditions=context.other_conditions,
            other_condition_evidence=context.other_condition_evidence,
            condition_status=context.condition_status,
            evidence=_materialize_evidence(item.evidence, block_map),
            confidence=item.confidence,
        ))
    condition_contexts = {
        item.condition_id: _materialize_measurement_context(item, block_map)
        for item in parsed.measurement_conditions
    }
    series_id_map = {
        item.series_id: f"series{index:03d}"
        for index, item in enumerate(parsed.property_series, start=1)
    }
    properties = [
        PropertyObservation(
            property_id=f"prop{index:03d}",
            sample_id=item.sample_id,
            property_name_raw=item.property_name_raw,
            property_name_normalized=item.property_name_normalized,
            property_code=item.property_code,
            property_category=item.property_category,
            molecular_weight_type=item.molecular_weight_type,
            determination_method_raw=item.determination_method_raw,
            observation_group_id=(
                group_id_map[item.observation_group_id]
                if item.observation_group_id is not None
                else None
            ),
            observation_role=item.observation_role,
            series_id=(
                series_id_map[item.series_id]
                if item.series_id is not None
                else None
            ),
            series_ids=(
                [series_id_map[series_id] for series_id in item.series_ids]
                if item.series_ids is not None
                else None
            ),
            value_raw=item.value_raw,
            value_min=item.value_min,
            value_max=item.value_max,
            unit_raw=item.unit_raw,
            unit_normalized=item.unit_normalized,
            measurement_condition_id=condition_id_map[
                item.measurement_condition_id
            ],
            measurement_context=(
                _materialize_measurement_context(
                    item.measurement_context, block_map
                )
                if item.measurement_context is not None
                else condition_contexts[item.measurement_condition_id]
            ),
            source_type=block_map[item.evidence[0].block_id].type,
            evidence=[
                _materialize_evidence(evidence, block_map)
                for evidence in item.evidence
            ],
            confidence=item.confidence,
        )
        for index, item in enumerate(parsed.properties, start=1)
    ]
    unresolved = [
        UnresolvedPropertyObservation(
            unresolved_id=f"uprop{index:03d}",
            entity_id=item.entity_id,
            sample_id=item.sample_id,
            property_name_raw=item.property_name_raw,
            property_name_normalized=item.property_name_normalized,
            property_code=item.property_code,
            property_category=item.property_category,
            molecular_weight_type=item.molecular_weight_type,
            determination_method_raw=item.determination_method_raw,
            observation_group_id=(
                group_id_map[item.observation_group_id]
                if item.observation_group_id is not None
                else None
            ),
            observation_role=item.observation_role,
            series_id=(
                series_id_map[item.series_id]
                if item.series_id is not None
                else None
            ),
            series_ids=(
                [series_id_map[series_id] for series_id in item.series_ids]
                if item.series_ids is not None
                else None
            ),
            value_raw=item.value_raw,
            value_min=item.value_min,
            value_max=item.value_max,
            unit_raw=item.unit_raw,
            unit_normalized=item.unit_normalized,
            measurement_condition_id=item.measurement_condition_id,
            measurement_context=(
                _materialize_measurement_context(
                    item.measurement_context, block_map
                )
                if item.measurement_context is not None
                else MeasurementContext(condition_status="not_reported")
            ),
            reason=item.reason,
            evidence=[
                _materialize_evidence(evidence, block_map)
                for evidence in item.evidence
            ],
            confidence=item.confidence,
        )
        for index, item in enumerate(parsed.unresolved_properties, start=1)
    ]
    point_index = 0
    property_series = []
    for item in parsed.property_series:
        points = []
        for point in item.points:
            point_index += 1
            points.append(PropertySeriesPoint(
                point_id=f"pt{point_index:03d}",
                observation_role=point.observation_role,
                sample_id=point.sample_id,
                entity_id=point.entity_id,
                sample_resolution_status=point.sample_resolution_status,
                coordinates=[
                    PropertySeriesCoordinate(
                        name_raw=coordinate.name_raw,
                        value_raw=coordinate.value_raw,
                        unit_raw=coordinate.unit_raw,
                        evidence=_materialize_evidence(
                            coordinate.evidence,
                            block_map,
                        ),
                    )
                    for coordinate in point.coordinates
                ],
                value_raw=point.value_raw,
                value_min=point.value_min,
                value_max=point.value_max,
                unit_raw=point.unit_raw,
                unit_normalized=point.unit_normalized,
                measurement_context=(
                    _materialize_measurement_context(
                        point.measurement_context or item.measurement_context,
                        block_map,
                    )
                ),
                coverage_status=point.coverage_status,
                evidence=[
                    _materialize_evidence(evidence, block_map)
                    for evidence in point.evidence
                ],
                confidence=point.confidence,
            ))
        property_series.append(PropertySeries(
            series_id=series_id_map[item.series_id],
            sample_id=item.sample_id,
            entity_id=item.entity_id,
            sample_resolution_status=item.sample_resolution_status,
            property_name_raw=item.property_name_raw,
            property_name_normalized=item.property_name_normalized,
            property_code=item.property_code,
            property_category=item.property_category,
            determination_method_raw=item.determination_method_raw,
            observation_group_id=(
                group_id_map[item.observation_group_id]
                if item.observation_group_id is not None
                else None
            ),
            unit_raw=item.unit_raw,
            unit_normalized=item.unit_normalized,
            measurement_context=_materialize_measurement_context(
                item.measurement_context, block_map
            ),
            points=points,
            coverage=item.coverage,
            evidence=[
                _materialize_evidence(evidence, block_map)
                for evidence in item.evidence
            ],
            confidence=item.confidence,
        ))
    return conditions, properties, unresolved, property_series



def _preview_salvage_materialization(
    parsed: PropertyStageResponse,
    blocks: list[Stage0Element],
) -> tuple[
    PropertyStageResponse,
    tuple[
        list[MeasurementCondition],
        list[PropertyObservation],
        list[UnresolvedPropertyObservation],
        list[PropertySeries],
    ],
    dict[str, Any],
]:
    """Preview 按对象保留可物化候选，避免单条坏数据清空整篇。"""

    block_ids = {block.block_id for block in blocks}
    report: dict[str, Any] = {
        "dropped_conditions": [],
        "dropped_properties": [],
        "dropped_unresolved_properties": [],
        "dropped_series": [],
        "dropped_points": [],
        "dropped_evidence": 0,
        "dropped_coordinates": 0,
    }

    def valid_evidence(item: Any) -> bool:
        return getattr(item, "block_id", None) in block_ids

    def clean_context_payload(payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = copy.deepcopy(payload)
        for field in (
            "temperature",
            "frequency",
            "humidity",
            "pressure",
            "wavelength",
        ):
            quantity = cleaned.get(field)
            if not isinstance(quantity, dict):
                continue
            evidence = quantity.get("evidence")
            if isinstance(evidence, list):
                kept = [
                    item for item in evidence
                    if isinstance(item, dict)
                    and item.get("block_id") in block_ids
                ]
                report["dropped_evidence"] += len(evidence) - len(kept)
                quantity["evidence"] = kept
        evidence_map = cleaned.get("other_condition_evidence")
        if isinstance(evidence_map, dict):
            kept_map: dict[str, list[dict[str, Any]]] = {}
            for key, evidence in evidence_map.items():
                if not isinstance(evidence, list):
                    continue
                kept = [
                    item for item in evidence
                    if isinstance(item, dict)
                    and item.get("block_id") in block_ids
                ]
                report["dropped_evidence"] += len(evidence) - len(kept)
                if kept:
                    kept_map[key] = kept
            cleaned["other_condition_evidence"] = kept_map
        return cleaned

    def clean_evidence_list(items: list[Any]) -> list[Any]:
        kept = [item for item in items if valid_evidence(item)]
        report["dropped_evidence"] += len(items) - len(kept)
        return kept

    def coverage_payload(points: list[PropertySeriesPointCandidate]) -> dict[str, Any]:
        covered = sum(item.coverage_status == "covered" for item in points)
        missing = sum(item.coverage_status == "missing" for item in points)
        not_applicable = sum(
            item.coverage_status == "not_applicable" for item in points
        )
        expected = covered + missing
        return {
            "expected": expected,
            "covered": covered,
            "missing": missing,
            "not_applicable": not_applicable,
            "ratio": covered / expected if expected else 1.0,
        }

    def candidate_from_payload(payload: dict[str, Any]) -> PropertyStageResponse:
        return PropertyStageResponse.model_validate(payload)

    safe_payload: dict[str, list[Any]] = {
        "measurement_conditions": [],
        "properties": [],
        "unresolved_properties": [],
        "property_series": [],
    }

    # Condition 是 resolved property 的必需引用；自身无法物化时只删除它，
    # 后续引用清扫会删除对应 property，不猜测其他 condition。
    for item in parsed.measurement_conditions:
        payload = clean_context_payload(item.model_dump(mode="python"))
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("block_id") not in block_ids:
            report["dropped_conditions"].append(item.condition_id)
            report["dropped_evidence"] += 1
            continue
        try:
            trial_payload = copy.deepcopy(safe_payload)
            trial_payload["measurement_conditions"].append(payload)
            trial = candidate_from_payload(trial_payload)
            _materialize(trial, blocks)
        except (KeyError, ValidationError, ValueError):
            report["dropped_conditions"].append(item.condition_id)
            continue
        safe_payload = trial.model_dump(mode="python")

    # Series 先逐 point 清理；point 坏只删 point，全部 point 坏才删 series。
    for item in parsed.property_series:
        series_payload = clean_context_payload(item.model_dump(mode="python"))
        series_evidence = clean_evidence_list(list(item.evidence))
        kept_points: list[PropertySeriesPointCandidate] = []
        for point in item.points:
            point_payload = point.model_dump(mode="python")
            point_payload["evidence"] = [
                evidence.model_dump(mode="python")
                for evidence in clean_evidence_list(list(point.evidence))
            ]
            if not point_payload["evidence"]:
                report["dropped_points"].append({
                    "series_id": item.series_id,
                    "point_id": point.point_id,
                })
                continue
            coordinates = []
            for coordinate in point.coordinates:
                if valid_evidence(coordinate.evidence):
                    coordinates.append(coordinate.model_dump(mode="python"))
                else:
                    report["dropped_coordinates"] += 1
                    report["dropped_evidence"] += 1
            point_payload["coordinates"] = coordinates
            if point.measurement_context is not None:
                point_payload["measurement_context"] = clean_context_payload(
                    point.measurement_context.model_dump(mode="python")
                )
            try:
                point_candidate = PropertySeriesPointCandidate.model_validate(
                    point_payload
                )
                point_series_payload = copy.deepcopy(series_payload)
                point_series_payload["points"] = [
                    point_candidate.model_dump(mode="python")
                ]
                point_series_payload["coverage"] = coverage_payload(
                    [point_candidate]
                )
                if (
                    point_series_payload.get("sample_id") is None
                    and point_series_payload.get("entity_id") is None
                    and (
                        point_candidate.sample_id is not None
                        or point_candidate.entity_id is not None
                    )
                ):
                    # 多主体 series 拆成单 point 试验时会暂时失去
                    # multiple_subjects 条件；仅在试验副本继承当前 point
                    # 的主体，避免把本来可物化的 point 误删。
                    point_series_payload["sample_id"] = point_candidate.sample_id
                    point_series_payload["entity_id"] = point_candidate.entity_id
                    point_series_payload["sample_resolution_status"] = (
                        point_candidate.sample_resolution_status
                    )
                if not series_evidence:
                    point_series_payload["evidence"] = point_payload["evidence"]
                else:
                    point_series_payload["evidence"] = [
                        evidence.model_dump(mode="python")
                        for evidence in series_evidence
                    ]
                point_series = PropertySeriesCandidate.model_validate(
                    point_series_payload
                )
                trial = PropertyStageResponse.model_validate({
                    "property_series": [
                        point_series.model_dump(mode="python")
                    ]
                })
                _materialize(trial, blocks)
            except (KeyError, ValidationError, ValueError):
                report["dropped_points"].append({
                    "series_id": item.series_id,
                    "point_id": point.point_id,
                })
                continue
            kept_points.append(point_candidate)

        if not kept_points:
            report["dropped_series"].append(item.series_id)
            continue
        series_payload["points"] = [
            point.model_dump(mode="python") for point in kept_points
        ]
        series_payload["coverage"] = coverage_payload(kept_points)
        if series_evidence:
            series_payload["evidence"] = [
                evidence.model_dump(mode="python")
                for evidence in series_evidence
            ]
        else:
            # Series evidence 是最终对象必填项；仅复用其保留 point 的原始 evidence，
            # 不创建新文本、不选择实体，也不改变 subject 归属。
            deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
            for point in kept_points:
                for evidence in point.evidence:
                    if valid_evidence(evidence):
                        key = (evidence.block_id, evidence.source_sentence)
                        deduplicated.setdefault(
                            key,
                            evidence.model_dump(mode="python"),
                        )
            series_payload["evidence"] = list(deduplicated.values())
        try:
            series_candidate = PropertySeriesCandidate.model_validate(
                series_payload
            )
            trial_payload = copy.deepcopy(safe_payload)
            trial_payload["property_series"].append(
                series_candidate.model_dump(mode="python")
            )
            trial = candidate_from_payload(trial_payload)
            _materialize(trial, blocks)
        except (KeyError, ValidationError, ValueError):
            report["dropped_series"].append(item.series_id)
            continue
        safe_payload = trial.model_dump(mode="python")

    known_conditions = {
        item["condition_id"] for item in safe_payload["measurement_conditions"]
    }
    known_series = {
        item["series_id"] for item in safe_payload["property_series"]
    }

    def references_known_series(item: Any) -> bool:
        references = set(item.series_ids or [])
        if item.series_id is not None:
            references.add(item.series_id)
        return references <= known_series

    for item in parsed.properties:
        if (
            item.measurement_condition_id not in known_conditions
            or not references_known_series(item)
        ):
            report["dropped_properties"].append(item.property_id)
            continue
        payload = item.model_dump(mode="python")
        payload["evidence"] = [
            evidence.model_dump(mode="python")
            for evidence in clean_evidence_list(list(item.evidence))
        ]
        if not payload["evidence"]:
            report["dropped_properties"].append(item.property_id)
            continue
        if item.measurement_context is not None:
            payload["measurement_context"] = clean_context_payload(
                item.measurement_context.model_dump(mode="python")
            )
        try:
            candidate = PropertyObservationCandidate.model_validate(payload)
            trial_payload = copy.deepcopy(safe_payload)
            trial_payload["properties"].append(
                candidate.model_dump(mode="python")
            )
            trial = candidate_from_payload(trial_payload)
            _materialize(trial, blocks)
        except (KeyError, ValidationError, ValueError):
            report["dropped_properties"].append(item.property_id)
            continue
        safe_payload = trial.model_dump(mode="python")

    for item in parsed.unresolved_properties:
        if not references_known_series(item):
            report["dropped_unresolved_properties"].append(item.unresolved_id)
            continue
        payload = item.model_dump(mode="python")
        payload["evidence"] = [
            evidence.model_dump(mode="python")
            for evidence in clean_evidence_list(list(item.evidence))
        ]
        if not payload["evidence"]:
            report["dropped_unresolved_properties"].append(item.unresolved_id)
            continue
        if item.measurement_context is not None:
            payload["measurement_context"] = clean_context_payload(
                item.measurement_context.model_dump(mode="python")
            )
        try:
            candidate = UnresolvedPropertyCandidate.model_validate(payload)
            trial_payload = copy.deepcopy(safe_payload)
            trial_payload["unresolved_properties"].append(
                candidate.model_dump(mode="python")
            )
            trial = candidate_from_payload(trial_payload)
            _materialize(trial, blocks)
        except (KeyError, ValidationError, ValueError):
            report["dropped_unresolved_properties"].append(item.unresolved_id)
            continue
        safe_payload = trial.model_dump(mode="python")

    salvaged = candidate_from_payload(safe_payload)
    materialized = _materialize(salvaged, blocks)
    return salvaged, materialized, report


def _normalized_table_label(value: str) -> str:
    projected, _ = _surface_projection(value, compact_math=True)
    return projected


def _recover_grouped_table_methods(
    items: list[UnresolvedPropertyObservation],
) -> tuple[
    list[UnresolvedPropertyObservation],
    list[dict[str, str]],
]:
    recovered_items = list(items)
    recovered: list[dict[str, str]] = []
    group_indices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        if item.observation_group_id is not None:
            group_indices.setdefault(
                item.observation_group_id,
                [],
            ).append(index)

    generic_columns = {
        "value",
        "values",
        "result",
        "results",
        "property",
        "properties",
    }
    for indices in group_indices.values():
        if (
            len(indices) < 2
            or any(items[index].determination_method_raw for index in indices)
        ):
            continue
        locators: list[tuple[Evidence, dict[str, str]]] = []
        valid_group = True
        for index in indices:
            table_evidence = [
                evidence
                for evidence in items[index].evidence
                if evidence.source_type == "table"
                and isinstance(evidence.table_locator, dict)
            ]
            if len(table_evidence) != 1:
                valid_group = False
                break
            evidence = table_evidence[0]
            locator = evidence.table_locator or {}
            if not all(
                isinstance(locator.get(field), str)
                and locator[field].strip()
                for field in (
                    "table_id",
                    "row_label",
                    "column_label",
                    "cell_value",
                )
            ):
                valid_group = False
                break
            locators.append((evidence, locator))
        if not valid_group:
            continue
        if len({
            locator["table_id"] for _, locator in locators
        }) != 1:
            continue
        if len({
            _normalized_table_label(locator["row_label"])
            for _, locator in locators
        }) != 1:
            continue
        column_labels = [
            locator["column_label"] for _, locator in locators
        ]
        normalized_columns = [
            _normalized_table_label(label) for label in column_labels
        ]
        if (
            len(set(normalized_columns)) != len(normalized_columns)
            or any(
                re.sub(r"[^a-z0-9]+", "", label.casefold())
                in generic_columns
                for label in column_labels
            )
        ):
            continue
        if any(
            normalized_columns[offset]
            in {
                _normalized_table_label(
                    items[index].property_name_raw
                ),
                _normalized_table_label(items[index].value_raw),
                _normalized_table_label(items[index].unit_raw or ""),
            }
            for offset, index in enumerate(indices)
        ):
            continue

        for offset, index in enumerate(indices):
            item = items[index]
            evidence, locator = locators[offset]
            confidence = item.confidence
            if confidence is not None:
                confidence = confidence.model_copy(
                    update={"score": min(confidence.score, 0.5)}
                )
            recovered_items[index] = item.model_copy(update={
                "determination_method_raw": locator["column_label"],
                "confidence": confidence,
            })
            recovered.append({
                "unresolved_id": item.unresolved_id,
                "block_id": evidence.block_id,
                "column_label": locator["column_label"],
            })
    return recovered_items, recovered


def _recovered_method_warning(
    recovered: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "stage": STAGE_ID,
        "code": "recovered_grouped_table_methods",
        "message": (
            "同组 unresolved property 位于同一表格行且列头唯一；"
            "已用原文列头恢复 determination_method_raw"
        ),
        "items": recovered,
    }


def _cache_components(
    document: Stage0Document,
    entities: Stage2Document,
    process: Stage3Document,
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
        "cache_revision": CACHE_REVISION,
        "preview_relaxed": preview_relaxed,
    })
    return input_hash, model_config_hash, cache_key


def extract_properties(
    document: Stage0Document,
    entities: Stage2Document,
    process: Stage3Document,
    client: LLMClient,
    prompt: RenderedPrompt,
    vocabulary: dict[str, tuple[str, str]],
    vocabulary_sha256: str,
    *,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 90000,
    max_validation_retries: int = 1,
    max_tokens: int = 32768,
    preview_relaxed: bool = False,
) -> Stage4Document:
    history_start = len(getattr(client, "call_history", []))
    if not (
        document.document_id == entities.document_id == process.document_id
    ):
        raise Stage4Error("Stage 0、Stage 2 与 Stage 3 document_id 不一致")
    blocks, warnings, context_chars = select_context_blocks(
        document,
        entities,
        process,
        input_sections=input_sections,
        max_input_chars=max_input_chars,
    )
    actual_models: list[str] = []
    dropped_table_evidence: list[tuple[str, str]] = []
    dropped_unanchored_evidence: list[tuple[str, str]] = []
    degraded_table_evidence: list[tuple[str, str]] = []
    degraded_condition_evidence: list[tuple[str, str]] = []
    supplemented_condition_evidence: list[tuple[str, str]] = []
    supplemented_property_evidence: list[tuple[str, str]] = []
    degraded_unresolved_evidence: list[tuple[str, str]] = []
    supplemented_unresolved_evidence: list[tuple[str, str]] = []
    dropped_condition_ids: list[str] = []
    candidate_repairs: dict[str, int] = {}
    dropped_confidence_fields: list[str] = []
    coordinate_only_table_columns: list[dict[str, Any]] = []
    preview_semantic_bypass_reason: str | None = None
    preview_degraded_reason: str | None = None

    if entities.polymer_entities:
        feedback = None
        last_error: Exception | None = None
        validation_errors: list[str] = []
        parsed: PropertyStageResponse | None = None
        for attempt in range(max_validation_retries + 1):
            try:
                response = client.call_json(
                    prompt.text,
                    _user_message(
                        document.document_id,
                        entities,
                        process,
                        blocks,
                        vocabulary,
                        feedback,
                    ),
                    max_tokens=max_tokens,
                )
                compacted_data, dropped_confidence_fields = (
                    compact_confidence_payload(response.data)
                )
                repaired_data, candidate_repairs = (
                    _repair_candidate_response_payload(
                        compacted_data,
                        process,
                        blocks,
                        vocabulary,
                        preview_relaxed=preview_relaxed,
                    )
                )
                response = LLMJSONResponse(
                    data=repaired_data,
                    provider=response.provider,
                    model=response.model,
                    usage=response.usage,
                    cost=response.cost,
                )
                try:
                    (
                        parsed,
                        dropped_table_evidence,
                        dropped_unanchored_evidence,
                        degraded_table_evidence,
                        degraded_condition_evidence,
                        supplemented_condition_evidence,
                        supplemented_property_evidence,
                        degraded_unresolved_evidence,
                        supplemented_unresolved_evidence,
                        dropped_condition_ids,
                        coordinate_only_table_columns,
                    ) = _validate_response(
                        response,
                        entities,
                        process,
                        blocks,
                        vocabulary,
                        preview_relaxed=preview_relaxed,
                    )
                except ValueError as exc:
                    if not preview_relaxed:
                        raise
                    # Preview 只要求结构可用。原文/evidence 语义校验失败时，
                    # 保留已通过 Schema 的候选，不为跑通而删除或改写对象。
                    parsed = PropertyStageResponse.model_validate(repaired_data)
                    preview_semantic_bypass_reason = _validation_feedback(exc)
                actual_models.append(response.model)
                last_error = None
                break
            except (LLMRequestError, ValidationError, ValueError) as exc:
                last_error = exc
                feedback = _validation_feedback(exc)
                validation_errors.append(feedback)
                if attempt >= max_validation_retries:
                    break
        if last_error is not None or parsed is None:
            failure_reason = (
                "；".join(validation_errors)
                or _validation_feedback(last_error or ValueError("empty"))
            )
            if preview_relaxed:
                parsed = PropertyStageResponse()
                preview_degraded_reason = failure_reason
            else:
                raise Stage4Error(
                    f"{document.document_id} 响应校验失败：{failure_reason}"
                ) from last_error
    else:
        parsed = PropertyStageResponse()

    preview_salvage_report: dict[str, Any] | None = None
    try:
        conditions, properties, unresolved, property_series = _materialize(
            parsed,
            blocks,
        )
    except (KeyError, ValidationError, ValueError) as exc:
        if not preview_relaxed:
            raise Stage4Error(
                f"{document.document_id} 响应物化失败：{_validation_feedback(exc)}"
            ) from exc
        try:
            parsed, materialized, preview_salvage_report = (
                _preview_salvage_materialization(parsed, blocks)
            )
            conditions, properties, unresolved, property_series = materialized
        except (KeyError, ValidationError, ValueError) as salvage_exc:
            parsed = PropertyStageResponse()
            conditions, properties, unresolved, property_series = _materialize(
                parsed,
                blocks,
            )
            preview_degraded_reason = (
                "结构化候选无法安全物化："
                + _validation_feedback(exc)
                + "；逐对象保留失败："
                + _validation_feedback(salvage_exc)
            )
    if coordinate_only_table_columns:
        warnings.append({
            "stage": STAGE_ID,
            "code": "table_property_column_represented_as_coordinate",
            "message": (
                "候选模式发现未完整表示的多值性质列：部分列仅作为其他 "
                "Series 的 coordinate 留存，或尚未抽取；已有结果保留"
            ),
            "columns": coordinate_only_table_columns,
        })
    if isinstance(client, _FailureReplayClient):
        warnings.append({
            "stage": STAGE_ID,
            "code": "failure_response_replayed",
            "message": "已离线回放保存的 Stage 4 响应，未请求模型",
            "source": client.failure_path.name,
        })
    if candidate_repairs:
        warnings.extend(_candidate_repair_warnings(candidate_repairs))
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
    if preview_salvage_report is not None:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_objects_salvaged",
            "message": (
                "Preview 已逐对象保留可安全物化的数据；坏字段、坏对象或"
                "失效引用已局部删除，Strict 模式仍会报错"
            ),
            "details": preview_salvage_report,
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
    if dropped_confidence_fields:
        warnings.append({
            "stage": STAGE_ID,
            "code": "confidence_fields_compacted",
            "message": "confidence 已确定性收敛为仅保留 score",
            "fields": list(dict.fromkeys(dropped_confidence_fields)),
        })
    unresolved, recovered_methods = _recover_grouped_table_methods(unresolved)
    if unresolved:
        warnings.append({
            "stage": STAGE_ID,
            "code": "unresolved_properties",
            "message": f"{len(unresolved)} 条性质无法可靠关联具体 Sample",
            "unresolved_ids": [item.unresolved_id for item in unresolved],
        })
    if recovered_methods:
        warnings.append(_recovered_method_warning(recovered_methods))
    incomplete_series = [
        item
        for item in property_series
        if item.coverage.missing > 0
    ]
    if incomplete_series:
        warnings.append({
            "stage": STAGE_ID,
            "code": "property_series_incomplete",
            "message": (
                f"{len(incomplete_series)} 个 PropertySeries 存在缺失数据点"
            ),
            "series_ids": [item.series_id for item in incomplete_series],
        })
    if dropped_table_evidence:
        property_id_map = {
            item.property_id: properties[index].property_id
            for index, item in enumerate(parsed.properties)
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "dropped_table_evidence",
            "message": (
                "缺少 table_locator 的附加表格 evidence 已舍弃；"
                "对应 property 仍有其他完整证据"
            ),
            "items": [
                {
                    "property_id": property_id_map[property_id],
                    "block_id": block_id,
                }
                for property_id, block_id in dropped_table_evidence
            ],
        })
    if dropped_condition_ids:
        warnings.append({
            "stage": STAGE_ID,
            "code": "dropped_unused_conditions",
            "message": (
                "未被任何 PropertyObservation 引用的 "
                "MeasurementCondition 已舍弃"
            ),
            "candidate_condition_ids": dropped_condition_ids,
        })
    if dropped_unanchored_evidence:
        property_id_map = {
            item.property_id: properties[index].property_id
            for index, item in enumerate(parsed.properties)
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "dropped_unanchored_evidence",
            "message": (
                "无法定位句子或 property raw 锚点的附加 evidence 已舍弃；"
                "对应 property 仍有其他完整证据"
            ),
            "items": [
                {
                    "property_id": property_id_map[property_id],
                    "block_id": block_id,
                }
                for property_id, block_id in dropped_unanchored_evidence
            ],
        })
    if degraded_table_evidence:
        property_id_map = {
            item.property_id: properties[index].property_id
            for index, item in enumerate(parsed.properties)
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "table_evidence_without_locator",
            "message": (
                "唯一表格证据的 raw 锚点均可定位，但模型未提供 "
                "table_locator；已降级保留块级证据"
            ),
            "items": [
                {
                    "property_id": property_id_map[property_id],
                    "block_id": block_id,
                }
                for property_id, block_id in degraded_table_evidence
            ],
        })
    if degraded_condition_evidence:
        condition_id_map = {
            candidate.condition_id: materialized.condition_id
            for candidate, materialized in zip(
                [
                    item
                    for item in parsed.measurement_conditions
                    if item.condition_id not in dropped_condition_ids
                ],
                conditions,
                strict=True,
            )
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "table_condition_evidence_without_locator",
            "message": (
                "MeasurementCondition 的表格 evidence 可定位原文，"
                "但 locator 缺失或无效；已降级保留块级证据"
            ),
            "items": [
                {
                    "condition_id": condition_id_map[condition_id],
                    "block_id": block_id,
                }
                for condition_id, block_id in degraded_condition_evidence
            ],
        })
    if supplemented_condition_evidence:
        condition_id_map = {
            candidate.condition_id: materialized.condition_id
            for candidate, materialized in zip(
                [
                    item
                    for item in parsed.measurement_conditions
                    if item.condition_id not in dropped_condition_ids
                ],
                conditions,
                strict=True,
            )
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "supplemented_condition_evidence",
            "message": (
                "模型 condition evidence 未覆盖 condition raw；"
                "已改用同时包含条件值与关联 property raw 的原文块"
            ),
            "items": [
                {
                    "condition_id": condition_id_map[condition_id],
                    "block_id": block_id,
                }
                for condition_id, block_id
                in supplemented_condition_evidence
            ],
        })
    if supplemented_property_evidence:
        property_id_map = {
            item.property_id: properties[index].property_id
            for index, item in enumerate(parsed.properties)
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "supplemented_property_evidence",
            "message": (
                "模型 evidence 未覆盖 property raw 字段；"
                "已补入同时包含该 raw 与 value_raw 的正文块"
            ),
            "items": [
                {
                    "property_id": property_id_map[property_id],
                    "block_id": block_id,
                }
                for property_id, block_id in supplemented_property_evidence
            ],
        })
    if degraded_unresolved_evidence:
        unresolved_id_map = {
            item.unresolved_id: unresolved[index].unresolved_id
            for index, item in enumerate(parsed.unresolved_properties)
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "table_unresolved_evidence_without_locator",
            "message": (
                "Unresolved property 的表格 raw 锚点可定位，"
                "但 locator 缺失或无效；已降级保留块级证据"
            ),
            "items": [
                {
                    "unresolved_id": unresolved_id_map[unresolved_id],
                    "block_id": block_id,
                }
                for unresolved_id, block_id in degraded_unresolved_evidence
            ],
        })
    if supplemented_unresolved_evidence:
        unresolved_id_map = {
            item.unresolved_id: unresolved[index].unresolved_id
            for index, item in enumerate(parsed.unresolved_properties)
        }
        warnings.append({
            "stage": STAGE_ID,
            "code": "supplemented_unresolved_evidence",
            "message": (
                "模型 evidence 未覆盖 unresolved property raw；"
                "已补入同时包含该 raw 与 value_raw 的正文块"
            ),
            "items": [
                {
                    "unresolved_id": unresolved_id_map[unresolved_id],
                    "block_id": block_id,
                }
                for unresolved_id, block_id
                in supplemented_unresolved_evidence
            ],
        })

    input_hash, model_config_hash, cache_key = _cache_components(
        document,
        entities,
        process,
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
    provenance = Stage4Provenance(
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
    return Stage4Document(
        document_id=document.document_id,
        measurement_conditions=conditions,
        properties=properties,
        unresolved_properties=unresolved,
        property_series=property_series,
        provenance=provenance,
        warnings=warnings,
    )


def _stage4_raw_response_artifact(
    client: LLMClient,
    *,
    document_id: str,
    history_start: int,
) -> dict[str, Any] | None:
    """保存不含请求、凭据和 HTTP headers 的最近一次模型响应。"""

    raw = getattr(client, "last_raw_response", None)
    if raw is None:
        return None
    history = getattr(client, "call_history", [])
    call_count = max(0, len(history) - history_start)
    usage, cost = summarize_client_calls(
        client,
        history_start,
        call_count=call_count,
    )
    def json_safe(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    return json_safe({
        "status": "received",
        "stage": STAGE_ID,
        "document_id": document_id,
        "call_count": call_count,
        "usage": usage,
        "cost": cost,
        "raw_response": {
            "provider": raw.provider,
            "model": raw.model,
            "finish_reason": raw.finish_reason,
            "content": raw.content,
            "usage": asdict(raw.usage),
            "cost": asdict(raw.cost) if raw.cost is not None else None,
        },
    })


def run_stage4(
    stage0_path: Path,
    stage2_path: Path,
    stage3_path: Path,
    output_path: Path,
    client: LLMClient,
    prompt: RenderedPrompt,
    vocabulary: dict[str, tuple[str, str]],
    vocabulary_sha256: str,
    *,
    force: bool = False,
    input_sections: tuple[str, ...] = DEFAULT_INPUT_SECTIONS,
    max_input_chars: int = 90000,
    max_validation_retries: int = 1,
    max_tokens: int = 32768,
    preview_relaxed: bool = False,
) -> tuple[Path, bool]:
    document = load_stage0_document(stage0_path)
    entities = load_stage2_document(stage2_path)
    process = load_stage3_document(stage3_path)
    _, _, expected_cache_key = _cache_components(
        document,
        entities,
        process,
        prompt,
        vocabulary_sha256,
        client,
        preview_relaxed=preview_relaxed,
    )
    if output_path.is_file() and not force:
        try:
            cached = Stage4Document.model_validate_json(
                output_path.read_text(encoding="utf-8-sig")
            )
            if cached.provenance.cache_key == expected_cache_key:
                return output_path, True
            for compatible_version in COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS:
                _, _, compatible_cache_key = _cache_components(
                    document,
                    entities,
                    process,
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
            _, _, legacy_cache_key = _cache_components(
                document,
                entities,
                process,
                prompt,
                vocabulary_sha256,
                client,
                implementation_version="1.4.0",
                preview_relaxed=preview_relaxed,
            )
            if (
                cached.provenance.implementation_version == "1.4.0"
                and cached.provenance.cache_key == legacy_cache_key
            ):
                unresolved, recovered = _recover_grouped_table_methods(
                    cached.unresolved_properties
                )
                warnings = list(cached.warnings)
                if recovered:
                    warnings.append(_recovered_method_warning(recovered))
                upgraded = cached.model_copy(update={
                    "unresolved_properties": unresolved,
                    "provenance": cached.provenance.model_copy(update={
                        "implementation_version": IMPLEMENTATION_VERSION,
                        "cache_key": expected_cache_key,
                    }),
                    "warnings": warnings,
                })
                write_json_atomic(
                    output_path,
                    upgraded.model_dump(mode="json", exclude_none=True),
                )
                return output_path, True
        except (OSError, ValidationError):
            pass

    history_start = len(getattr(client, "call_history", []))
    result = extract_properties(
        document,
        entities,
        process,
        client,
        prompt,
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
    raw_artifact = _stage4_raw_response_artifact(
        client,
        document_id=document.document_id,
        history_start=history_start,
    )
    if raw_artifact is not None:
        write_json_atomic(
            output_path.with_name("stage4_llm_response.json"),
            raw_artifact,
        )
    return output_path, False


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages") or {}
    stage = stages.get(STAGE_ID) or {}
    if not isinstance(stage, dict):
        raise Stage4Error(f"配置 {STAGE_ID} 必须是对象")
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 Stage 4 性质与条件抽取")
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
        help="演示模式：保留 raw/evidence 并清理 unresolved 禁用字段",
    )
    parser.add_argument(
        "--replay-failure",
        action="store_true",
        help="离线回放现有 stage4_failure.json，不请求模型",
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
        else _resolve_vocabulary_path(
            stage_config.get("vocabulary_path") or DEFAULT_VOCABULARY_PATH,
            config_path=config_path,
        )
    )
    vocabulary, vocabulary_sha256 = load_property_vocabulary(vocabulary_path)
    prompt_id = str(
        stage_config.get("prompt_id") or "polymer.stage4.property"
    )
    prompt = PromptLoader().render_stage_prompt(
        prompt_id,
        PropertyStageResponse,
        expected_stage=STAGE_ID,
        expected_output_schema=OUTPUT_SCHEMA_VERSION,
    )
    if args.replay_failure and not args.ref_no:
        raise Stage4Error("--replay-failure 必须与单篇 --ref-no 配合使用")
    client = (
        _failure_replay_client(
            input_root / args.ref_no / "stage4_failure.json",
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
    max_tokens = int(stage_config.get("max_tokens") or 32768)

    if args.ref_no:
        ref_nos = [args.ref_no]
    else:
        ref_nos = sorted(
            path.parent.name
            for path in input_root.glob("reference_no_*/stage3_process.json")
        )
    if not ref_nos:
        raise Stage4Error(f"未找到 Stage 3 输出：{input_root}")

    failures: list[tuple[str, str]] = []
    for ref_no in ref_nos:
        history_start = len(client.call_history)
        try:
            output_path, cached = run_stage4(
                input_root / ref_no / "stage0_blocks.json",
                input_root / ref_no / "stage2_entities.json",
                input_root / ref_no / "stage3_process.json",
                output_root / ref_no / "stage4_properties.json",
                client,
                prompt,
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
                    output_root / ref_no / "stage4_failure.json",
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
    print(f"Stage 4 完成：成功 {len(ref_nos) - len(failures)}，失败 {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
