"""Stage 1：使用 LLM 识别 MaterialMention。"""

from __future__ import annotations

import argparse
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
PROJECT_ROOT = EXTRACTION_ROOT.parent.parent
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMCallCost,
    LLMCallRecord,
    LLMClient,
    LLMJSONResponse,
    LLMOutputTruncatedError,
    LLMRequestError,
    LLMTokenUsage,
    extract_json_object,
    llm_config_cache_payload,
    llm_failure_artifact,
    load_pipeline_config,
    resolve_llm_config,
    resolve_pricing_config,
    summarize_client_calls,
)
from prompt_loader import PromptLoader, RenderedPrompt
from schema.polymer_schema import (
    compact_confidence_payload,
    Evidence,
    MaterialMention,
    MentionCandidate,
    MentionChunkResponse,
    Stage0Document,
    Stage0Element,
    Stage1Document,
    Stage1Provenance,
)
from stages.table_grid import table_cells_for


STAGE_ID = "stage1_material_mention"
OUTPUT_SCHEMA_VERSION = "material_mention_schema.v2"
IMPLEMENTATION_VERSION = "1.2.6"
# 1.2.6 新增的标记容忍恢复只在旧版本会硬失败的路径上触发，
# 不改变任何既有成功产物，故旧缓存仍可复用。
COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS = (
    "1.2.5",
    "1.2.4",
    "1.2.3",
    "1.2.2",
)
DEFAULT_PRIMARY_SECTIONS = ("Abstract", "Methods", "Results")
DEFAULT_FALLBACK_SECTIONS = ("Introduction", "Conclusion")
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？]\s+|\n+")


class Stage1Error(RuntimeError):
    """Stage 1 输入、LLM 响应或输出验证失败。"""


class _FailureReplayClient:
    """仅返回 Stage 1 failure 中保存的一次响应，不发起网络请求。"""

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
            raise Stage1Error("Stage 1 failure 响应只允许离线回放一次")
        self.calls += 1
        self.call_history.append(self.record)
        return self.response


def _failure_replay_client(
    failure_path: Path,
    config: dict[str, Any],
    *,
    stage0_path: Path,
    max_chunk_chars: int,
    primary_sections: tuple[str, ...],
    fallback_sections: tuple[str, ...],
) -> _FailureReplayClient:
    """构造单响应回放客户端；多 chunk 输入不可安全回放。"""
    document = load_stage0_document(stage0_path)
    blocks, _ = select_input_blocks(
        document,
        primary_sections,
        fallback_sections,
    )
    chunk_count = len(chunk_blocks(blocks, max_chunk_chars))
    if chunk_count != 1:
        raise Stage1Error(
            "Stage 1 failure 仅保存最后一次响应，"
            f"当前输入为 {chunk_count} 个 chunk，不可安全回放"
        )
    if not failure_path.is_file():
        raise Stage1Error(f"缺少 Stage 1 failure 文件：{failure_path}")
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1Error(f"Stage 1 failure 文件无效：{failure_path}") from exc
    raw = failure.get("raw_response") if isinstance(failure, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
        raise Stage1Error("Stage 1 failure 未保存可回放的 raw response")
    try:
        data = extract_json_object(raw["content"])
    except LLMRequestError as exc:
        raise Stage1Error(
            f"Stage 1 failure raw response 无法解析为 JSON 对象：{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise Stage1Error("Stage 1 failure raw response 必须是 JSON 对象")

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


def _element_input_text(element: Stage0Element) -> str:
    if element.type == "table":
        return "\n".join(
            value.strip()
            for value in (element.caption, element.table_body)
            if value and value.strip()
        )
    return (element.text or "").strip()


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


def load_stage0_document(path: Path) -> Stage0Document:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return Stage0Document.model_validate(raw)
    except OSError as exc:
        raise Stage1Error(f"无法读取 Stage 0：{path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage1Error(f"Stage 0 JSON 无效：{path}") from exc
    except ValidationError as exc:
        raise Stage1Error(f"Stage 0 未通过 Schema：{path.name}") from exc


def select_input_blocks(
    document: Stage0Document,
    primary_sections: tuple[str, ...] = DEFAULT_PRIMARY_SECTIONS,
    fallback_sections: tuple[str, ...] = DEFAULT_FALLBACK_SECTIONS,
) -> tuple[list[Stage0Element], list[dict[str, Any]]]:
    def selected(sections: tuple[str, ...]) -> list[Stage0Element]:
        return [
            element
            for element in document.elements
            if element.type in {"text", "title", "table"}
            and element.section in sections
            and bool(_element_input_text(element))
        ]

    blocks = selected(primary_sections)
    if blocks:
        return blocks, []
    fallback = selected(fallback_sections)
    if fallback:
        return fallback, [{
            "stage": STAGE_ID,
            "code": "section_fallback",
            "message": (
                "Abstract/Methods/Results 为空，改用 Introduction/Conclusion；"
                "结果需人工复核"
            ),
        }]
    unsectioned = [
        element
        for element in document.elements
        if element.type in {"text", "title", "table"}
        and bool(_element_input_text(element))
    ]
    if not unsectioned:
        raise Stage1Error(
            f"{document.document_id} 没有可供 Stage 1 使用的 text/title/table block"
        )
    return unsectioned, [{
        "stage": STAGE_ID,
        "code": "unsectioned_blocks_fallback",
        "message": (
            "未识别到受支持的章节标签，改用全部有内容的 text/title/table；"
            "结果需人工复核"
        ),
    }]


def chunk_blocks(
    blocks: list[Stage0Element],
    max_chunk_chars: int,
) -> list[list[Stage0Element]]:
    if max_chunk_chars < 2000:
        raise ValueError("max_chunk_chars 不得小于 2000")
    chunks: list[list[Stage0Element]] = []
    current: list[Stage0Element] = []
    current_size = 0
    for block in blocks:
        estimated = len(_element_input_text(block)) + 160
        if current and current_size + estimated > max_chunk_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(block)
        current_size += estimated
    if current:
        chunks.append(current)
    return chunks


def _chunk_user_message(
    document_id: str,
    chunk: list[Stage0Element],
    chunk_index: int,
    chunk_count: int,
    validation_feedback: str | None = None,
) -> str:
    blocks = [
        {
            "block_id": block.block_id,
            "page": block.page,
            "type": block.type,
            "section": block.section,
            "text": block.text,
            "caption": block.caption,
            "table_body": block.table_body,
        }
        for block in chunk
    ]
    message = (
        f"document_id: {document_id}\n"
        f"chunk: {chunk_index}/{chunk_count}\n"
        "--- BEGIN UNTRUSTED DOCUMENT BLOCKS ---\n"
        + json.dumps(blocks, ensure_ascii=False, indent=2)
        + "\n--- END UNTRUSTED DOCUMENT BLOCKS ---"
    )
    if validation_feedback:
        message += (
            "\n\n上一次响应未通过校验。请重新输出完整 JSON。"
            f"错误类型：{validation_feedback}"
        )
    return message


def _validate_chunk_candidates(
    response: LLMJSONResponse,
    chunk: list[Stage0Element],
    dropped_confidence_fields: list[str] | None = None,
    surface_repairs: list[dict[str, str]] | None = None,
    preview_invalid_mentions_removed: list[dict[str, str]] | None = None,
) -> MentionChunkResponse:
    cleaned_data, dropped = compact_confidence_payload(response.data)
    parsed = MentionChunkResponse.model_validate(cleaned_data)
    block_map = {block.block_id: block for block in chunk}
    resolved_candidates: list[MentionCandidate] = []
    for candidate in parsed.mentions:
        block = block_map.get(candidate.block_id)
        if block is None:
            raise ValueError(f"未知 block_id：{candidate.block_id}")
        resolved_text = _resolve_surface_text(
            _element_input_text(block),
            candidate.text,
        )
        if resolved_text is None:
            if preview_invalid_mentions_removed is not None:
                preview_invalid_mentions_removed.append({
                    "block_id": candidate.block_id,
                    "model_text": candidate.text,
                    "reason": "not_source_substring",
                })
                continue
            raise ValueError(
                f"mention text {candidate.text!r} 不是 "
                f"{candidate.block_id} 的原文子串"
            )
        if surface_repairs is not None and resolved_text != candidate.text:
            surface_repairs.append({
                "block_id": candidate.block_id,
                "model_text": candidate.text,
                "source_text": resolved_text,
            })
        resolved_candidates.append(candidate.model_copy(
            update={"text": resolved_text}
        ))
    if dropped_confidence_fields is not None:
        dropped_confidence_fields.extend(dropped)
    return MentionChunkResponse(mentions=resolved_candidates)


def _resolve_surface_text(source: str, candidate: str) -> str | None:
    if re.fullmatch(r"[A-Za-z0-9]+", candidate):
        token_match = re.search(
            rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])",
            source,
            flags=re.IGNORECASE,
        )
        if token_match:
            return token_match.group(0)
        if re.search(r"[A-Za-z]", candidate) and re.search(r"\d", candidate):
            embedded = list(re.finditer(
                re.escape(candidate),
                source,
                flags=re.IGNORECASE,
            ))
            if len(embedded) == 1:
                return embedded[0].group(0)
        if candidate.isalpha() and len(candidate) >= 8:
            embedded = list(re.finditer(
                re.escape(candidate),
                source,
                flags=re.IGNORECASE,
            ))
            if len(embedded) == 1:
                match = embedded[0]
                has_left_boundary = (
                    match.start() == 0
                    or not source[match.start() - 1].isalnum()
                )
                has_right_boundary = (
                    match.end() == len(source)
                    or not source[match.end()].isalnum()
                )
                if has_left_boundary or has_right_boundary:
                    return match.group(0)
        if re.search(r"[A-Za-z]", candidate) and re.search(r"\d", candidate):
            spaced_pattern = r"\s*".join(
                re.escape(character) for character in candidate
            )
            character_spaced = re.search(
                rf"(?<![A-Za-z0-9]){spaced_pattern}(?![A-Za-z0-9])",
                source,
                flags=re.IGNORECASE,
            )
            if character_spaced:
                return character_spaced.group(0)
            latex_sequence = _resolve_formula_sequence_surface(
                source,
                candidate,
            )
            if latex_sequence is not None:
                return latex_sequence
        return None
    if candidate in source:
        return candidate
    direct = re.search(re.escape(candidate), source, flags=re.IGNORECASE)
    if direct:
        return direct.group(0)
    decoded_match = re.search(
        re.escape(candidate),
        html.unescape(source),
        flags=re.IGNORECASE,
    )
    if decoded_match:
        return decoded_match.group(0)
    if candidate.startswith("$") and candidate.endswith("$"):
        inner = candidate[1:-1]
        inner_matches = list(re.finditer(
            re.escape(inner),
            source,
            flags=re.IGNORECASE,
        ))
        if len(inner_matches) == 1:
            return inner_matches[0].group(0)
    latex_surface = _resolve_latex_group_surface(source, candidate)
    if latex_surface is not None:
        return latex_surface
    formula_sequence = _resolve_formula_sequence_surface(source, candidate)
    if formula_sequence is not None:
        return formula_sequence
    tokens = candidate.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    pattern = pattern.replace(r"\-", "[-‐‑‒–—]")
    tolerant = re.search(pattern, source, flags=re.IGNORECASE)
    if tolerant:
        return tolerant.group(0)
    return _resolve_markup_tolerant_surface(source, candidate)


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


def _resolve_formula_sequence_surface(
    source: str,
    candidate: str,
) -> str | None:
    """恢复 LaTeX/标点差异，但要求完整字母数字序列和短原文片段。"""

    candidate_key = _formula_key(candidate)
    mixed_letter_digit = (
        any(character.isalpha() for character in candidate)
        and any(character.isdigit() for character in candidate)
    )
    minimum_length = 2 if mixed_letter_digit else 4
    if len(candidate_key) < minimum_length:
        return None
    projected: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(source):
        if source[index] == "\\":
            command = re.match(r"\\[A-Za-z]+", source[index:])
            if command is not None:
                index += len(command.group(0))
                continue
        if source[index] == "<":
            tag = re.match(r"</?[^>]+>", source[index:])
            if tag is not None:
                index += len(tag.group(0))
                continue
        character = source[index]
        if character.isalnum():
            for folded in character.casefold():
                projected.append(folded)
                spans.append((index, index + 1))
        index += 1
    projected_text = "".join(projected)
    fragments = []
    start = 0
    while True:
        position = projected_text.find(candidate_key, start)
        if position < 0:
            break
        fragment = source[
            spans[position][0]:spans[position + len(candidate_key) - 1][1]
        ]
        if len(fragment) <= 256:
            fragments.append(fragment)
        start = position + 1
    if not fragments:
        return None
    shortest_length = min(len(fragment) for fragment in fragments)
    shortest = {
        fragment for fragment in fragments if len(fragment) == shortest_length
    }
    return next(iter(shortest)) if len(shortest) == 1 else None


# 原文中的上下标标记（MinerU 保留的 <sup>/<sub>）。模型常在引用时略去，
# 契约 §8.1 允许"上下标形式差异"的确定性恢复。
_MARKUP_GAP = r"(?:\s|<su[pb]>.*?</su[pb]>)*"


def _resolve_markup_tolerant_surface(
    source: str,
    candidate: str,
) -> str | None:
    """容忍原文中夹杂 <sup>/<sub> 标记的表面恢复。

    仅在**匹配唯一**时返回，否则返回 None 交由上层硬失败——
    契约 §3.1 要求确定性修复的结果必须唯一，
    多候选时不得选择"最像"的一个。
    """
    if "<" in candidate:
        return None
    units = [unit for unit in re.split(r"\s+", candidate) if unit]
    if not units:
        return None
    # 词内也可能被标记切断（poly( - <sup>R</sup>olefin)s），
    # 因此逐字符允许插入标记，而不只在空格处。
    pattern = _MARKUP_GAP.join(
        _MARKUP_GAP.join(
            re.escape(character) for character in unit
        ).replace(r"\-", "[-‐‑‒–—]")
        for unit in units
    )
    matches = list(re.finditer(pattern, source, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None
    resolved = matches[0].group(0)
    # 只有真的跨越了标记才算恢复；否则前面的分支应已命中
    if "<" not in resolved:
        return None
    return resolved


def _resolve_html_entity_source_fragment(
    source: str,
    candidate: str,
) -> str | None:
    decoded: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for match in re.finditer(
        r"&(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);",
        source,
        flags=re.IGNORECASE,
    ):
        for index in range(position, match.start()):
            decoded.append(source[index])
            spans.append((index, index + 1))
        replacement = html.unescape(match.group(0))
        if replacement == match.group(0):
            replacement = ""
        for character in replacement:
            decoded.append(character)
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


def _validation_feedback(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for item in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in item.get("loc") or ())
            parts.append(f"{location}: {item.get('msg', 'validation error')}")
        return "; ".join(parts)[:800]
    return str(error)[:800]


def _source_sentence(text: str, mention: str, max_chars: int = 500) -> str:
    resolved = _resolve_surface_text(text, mention)
    boundary_match = (
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(mention)}(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
        if re.fullmatch(r"[A-Za-z0-9]+", mention)
        else None
    )
    match = boundary_match or (
        re.search(re.escape(resolved), text, flags=re.IGNORECASE)
        if resolved is not None
        else None
    )
    if match is None:
        raise ValueError("mention 不在 evidence block 中")
    position = match.start()
    mention_end = match.end()
    start = 0
    end = len(text)
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        if match.end() <= position:
            start = match.end()
        elif match.start() >= mention_end:
            end = match.start() + 1
            break
    sentence = text[start:end].strip()
    if len(sentence) > max_chars:
        mention_in_sentence = position - start
        left = max(0, mention_in_sentence - max_chars // 2)
        right = min(len(sentence), left + max_chars)
        left = max(0, right - max_chars)
        sentence = sentence[left:right].strip()
    if _resolve_surface_text(sentence, mention) is None:
        raise ValueError("mention 不在最终 evidence sentence 中")
    return sentence


def _materialize_mentions(
    candidates: list[MentionCandidate],
    block_map: dict[str, Stage0Element],
) -> list[MaterialMention]:
    seen: set[tuple[str, str, str]] = set()
    mentions: list[MaterialMention] = []
    for candidate in candidates:
        key = (candidate.block_id, candidate.text, candidate.mention_role)
        if key in seen:
            continue
        seen.add(key)
        block = block_map[candidate.block_id]
        source = _element_input_text(block)
        if block.type == "table":
            matching_cells = [
                cell.text
                for cell in table_cells_for(block)
                if _resolve_surface_text(cell.text, candidate.text) is not None
            ]
            if len(matching_cells) == 1:
                entity_surface = _resolve_html_entity_source_fragment(
                    source,
                    candidate.text,
                )
                source_sentence = (
                    entity_surface
                    if entity_surface is not None
                    and html.unescape(entity_surface) != entity_surface
                    else matching_cells[0]
                )
            else:
                source_sentence = _source_sentence(source, candidate.text)
        else:
            source_sentence = _source_sentence(source, candidate.text)
        evidence = Evidence(
            block_id=block.block_id,
            page=block.page,
            bbox=block.bbox,
            source_type=block.type,
            source_sentence=source_sentence,
        )
        mentions.append(MaterialMention(
            mention_id=f"m{len(mentions) + 1:03d}",
            text=candidate.text,
            mention_role=candidate.mention_role,
            evidence=evidence,
            confidence=candidate.confidence,
        ))
    return mentions


def _cache_components(
    document: Stage0Document,
    prompt: RenderedPrompt,
    client: LLMClient,
    *,
    preview_relaxed: bool = False,
) -> tuple[str, str, str]:
    input_hash = _sha256_json(document.model_dump(mode="json"))
    model_config_hash = _sha256_json(
        llm_config_cache_payload(client.resolved)
    )
    cache_payload = {
        "input_hash": input_hash,
        "rendered_prompt_hash": prompt.sha256,
        "model_config_hash": model_config_hash,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
    }
    if preview_relaxed:
        cache_payload["preview_relaxed"] = True
    cache_key = _sha256_json(cache_payload)
    return input_hash, model_config_hash, cache_key


def extract_material_mentions(
    document: Stage0Document,
    client: LLMClient,
    prompt: RenderedPrompt,
    *,
    max_chunk_chars: int = 8000,
    max_tokens: int = 8192,
    max_validation_retries: int = 1,
    primary_sections: tuple[str, ...] = DEFAULT_PRIMARY_SECTIONS,
    fallback_sections: tuple[str, ...] = DEFAULT_FALLBACK_SECTIONS,
    preview_relaxed: bool = False,
) -> Stage1Document:
    history_start = len(getattr(client, "call_history", []))
    blocks, warnings = select_input_blocks(
        document,
        primary_sections,
        fallback_sections,
    )
    chunks = chunk_blocks(blocks, max_chunk_chars)
    all_candidates: list[MentionCandidate] = []
    actual_models: list[str] = []
    dropped_confidence_fields: list[str] = []
    surface_repairs: list[dict[str, str]] = []
    preview_invalid_mentions_removed: list[dict[str, str]] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        feedback = None
        last_error: Exception | None = None
        for attempt in range(max_validation_retries + 1):
            try:
                response = client.call_json(
                    prompt.text,
                    _chunk_user_message(
                        document.document_id,
                        chunk,
                        chunk_index,
                        len(chunks),
                        feedback,
                    ),
                    max_tokens=max_tokens,
                )
                parsed = _validate_chunk_candidates(
                    response,
                    chunk,
                    dropped_confidence_fields,
                    surface_repairs,
                    (
                        preview_invalid_mentions_removed
                        if preview_relaxed
                        else None
                    ),
                )
                all_candidates.extend(parsed.mentions)
                actual_models.append(response.model)
                last_error = None
                break
            except LLMOutputTruncatedError as exc:
                last_error = exc
                break
            except (LLMRequestError, ValidationError, ValueError) as exc:
                last_error = exc
                feedback = _validation_feedback(exc)
                if attempt >= max_validation_retries:
                    break
        if last_error is not None:
            raise Stage1Error(
                f"{document.document_id} chunk {chunk_index} 响应校验失败："
                f"{_validation_feedback(last_error)}"
            ) from last_error

    block_map = {block.block_id: block for block in blocks}
    mentions = _materialize_mentions(all_candidates, block_map)
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
            "code": "mention_surface_recovered",
            "message": (
                "mention 表面文本已恢复为 evidence 中的原文形式"
                "（唯一匹配，多候选时不修复）"
            ),
            "items": surface_repairs,
        })
    if preview_invalid_mentions_removed:
        warnings.append({
            "stage": STAGE_ID,
            "code": "preview_invalid_mentions_removed",
            "message": "Preview 模式已丢弃无法恢复为原文子串的 mention 候选",
            "items": preview_invalid_mentions_removed,
        })
    input_hash, model_config_hash, cache_key = _cache_components(
        document,
        prompt,
        client,
        preview_relaxed=preview_relaxed,
    )
    unique_models = list(dict.fromkeys(actual_models))
    usage, cost = summarize_client_calls(
        client,
        history_start,
        call_count=len(actual_models),
    )
    provenance = Stage1Provenance(
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
        chunk_count=len(chunks),
        usage=usage,
        cost=cost,
    )
    return Stage1Document(
        document_id=document.document_id,
        material_mentions=mentions,
        provenance=provenance,
        warnings=warnings,
    )


def run_stage1(
    stage0_path: Path,
    output_path: Path,
    client: LLMClient,
    prompt: RenderedPrompt,
    *,
    force: bool = False,
    max_chunk_chars: int = 8000,
    max_tokens: int = 8192,
    max_validation_retries: int = 1,
    primary_sections: tuple[str, ...] = DEFAULT_PRIMARY_SECTIONS,
    fallback_sections: tuple[str, ...] = DEFAULT_FALLBACK_SECTIONS,
    record_failure: bool = True,
    preview_relaxed: bool = False,
) -> tuple[Path, bool]:
    document = load_stage0_document(stage0_path)
    history_start = len(getattr(client, "call_history", []))
    input_hash, model_config_hash, expected_cache_key = _cache_components(
        document,
        prompt,
        client,
        preview_relaxed=preview_relaxed,
    )
    if output_path.is_file() and not force:
        try:
            cached = Stage1Document.model_validate_json(
                output_path.read_text(encoding="utf-8-sig")
            )
            if cached.provenance.cache_key == expected_cache_key:
                return output_path, True
            for compatible_version in (
                ()
                if preview_relaxed
                else COMPATIBLE_CACHE_IMPLEMENTATION_VERSIONS
            ):
                compatible_key = _sha256_json({
                    "input_hash": input_hash,
                    "rendered_prompt_hash": prompt.sha256,
                    "model_config_hash": model_config_hash,
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "implementation_version": compatible_version,
                })
                if (
                    cached.provenance.implementation_version
                    == compatible_version
                    and cached.provenance.cache_key == compatible_key
                ):
                    return output_path, True
        except (OSError, ValidationError):
            pass

    try:
        result = extract_material_mentions(
            document,
            client,
            prompt,
            max_chunk_chars=max_chunk_chars,
            max_tokens=max_tokens,
            max_validation_retries=max_validation_retries,
            primary_sections=primary_sections,
            fallback_sections=fallback_sections,
            preview_relaxed=preview_relaxed,
        )
    except Exception as exc:
        if record_failure:
            write_json_atomic(
                output_path.with_name("stage1_failure.json"),
                llm_failure_artifact(
                    client,
                    stage=STAGE_ID,
                    document_id=document.document_id,
                    error=exc,
                    history_start=history_start,
                ),
            )
        raise
    write_json_atomic(
        output_path,
        result.model_dump(mode="json", exclude_none=True),
    )
    return output_path, False


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    stages = config.get("stages") or {}
    stage = stages.get(STAGE_ID) or {}
    if not isinstance(stage, dict):
        raise Stage1Error(f"配置 {STAGE_ID} 必须是对象")
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 Stage 1 MaterialMention 识别")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ref-no")
    mode.add_argument("--batch", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-chunk-chars", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--replay-failure",
        action="store_true",
        help="仅用 stage1_failure.json 中保存的响应离线重放（只支持单 chunk）",
    )
    parser.add_argument(
        "--preview-relaxed",
        action="store_true",
        help="演示模式：丢弃无法恢复为原文子串的单个 mention 候选并记录 warning",
    )
    parser.add_argument("--force", action="store_true")
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
        stage_config.get("prompt_id") or "polymer.stage1.material_mention"
    )
    prompt = PromptLoader().render_stage_prompt(
        prompt_id,
        MentionChunkResponse,
        expected_stage=STAGE_ID,
        expected_output_schema=OUTPUT_SCHEMA_VERSION,
    )
    max_chunk_chars = int(
        args.max_chunk_chars
        or stage_config.get("max_chunk_chars")
        or 8000
    )
    max_tokens = int(
        args.max_tokens
        or stage_config.get("max_tokens")
        or 8192
    )
    max_validation_retries = int(
        stage_config.get("max_validation_retries", 1)
    )
    primary_sections = tuple(
        stage_config.get("input_sections") or DEFAULT_PRIMARY_SECTIONS
    )
    fallback_sections = tuple(
        stage_config.get("fallback_sections") or DEFAULT_FALLBACK_SECTIONS
    )

    if args.ref_no:
        ref_nos = [args.ref_no]
    else:
        ref_nos = sorted(
            path.parent.name
            for path in input_root.glob("reference_no_*/stage0_blocks.json")
        )
    if not ref_nos:
        raise Stage1Error(f"未找到 Stage 0 输出：{input_root}")
    if args.replay_failure and not args.ref_no:
        raise Stage1Error("--replay-failure 必须与单个 --ref-no 一起使用")

    if args.replay_failure:
        replay_ref = str(args.ref_no)
        client = _failure_replay_client(
            output_root / replay_ref / "stage1_failure.json",
            config,
            stage0_path=input_root / replay_ref / "stage0_blocks.json",
            max_chunk_chars=max_chunk_chars,
            primary_sections=primary_sections,
            fallback_sections=fallback_sections,
        )
        max_validation_retries = 0
    else:
        client = LLMClient.from_pipeline_config(
            stage=STAGE_ID,
            config_path=config_path,
        )

    failures: list[tuple[str, str]] = []
    for ref_no in ref_nos:
        try:
            output_path, cached = run_stage1(
                input_root / ref_no / "stage0_blocks.json",
                output_root / ref_no / "stage1_mentions.json",
                client,
                prompt,
                force=args.force,
                max_chunk_chars=max_chunk_chars,
                max_tokens=max_tokens,
                max_validation_retries=max_validation_retries,
                primary_sections=primary_sections,
                fallback_sections=fallback_sections,
                record_failure=not args.replay_failure,
                preview_relaxed=args.preview_relaxed,
            )
            state = "cached" if cached else "done"
            print(f"[{state}] {ref_no} -> {output_path}")
        except Exception as exc:
            failures.append((ref_no, type(exc).__name__))
            print(f"[failed] {ref_no}: {exc}", file=sys.stderr)
    print(f"Stage 1 完成：成功 {len(ref_nos) - len(failures)}，失败 {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
