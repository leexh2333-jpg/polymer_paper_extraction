"""将 MinerU 产物融合为抽取阶段使用的标准化 document JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
EXTRACTION_ROOT = PACKAGE_ROOT / "extraction"
DEFAULT_WORK_ROOT = PACKAGE_ROOT / "work_pdf_pipeline"
DEFAULT_MINERU_OUTPUT = DEFAULT_WORK_ROOT / "mineru"
DEFAULT_ORGANIZED_ROOT = DEFAULT_WORK_ROOT / "organized"
DEFAULT_PROCESSED_OUTPUT = DEFAULT_WORK_ROOT / "processed"
META_PROMPT_PATH = EXTRACTION_ROOT / "prompts" / "meta_extract.md"
DEFAULT_PIPELINE_CONFIG = EXTRACTION_ROOT / "config" / "pipeline.yaml"

HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\((?P<path>[^)\s]+)")
SKIPPED_BLOCK_TYPES = {"header", "footer", "page_header", "page_footer", "page_number"}
META_FIELDS = {"doi", "title", "authors", "journal", "year"}
META_OUTPUT_FIELDS = META_FIELDS | {"confidence"}


class DocumentGateError(RuntimeError):
    """MinerU 完成状态或必需产物不满足转换门禁。"""


@dataclass(frozen=True)
class MineruFiles:
    ref_no: str
    paper_dir: Path
    markdown: Path
    content_v1: Path
    content_v2: Path | None
    pdf: Path
    images_dir: Path | None


@dataclass(frozen=True)
class BatchContext:
    batch_id: str
    state: str
    manifest_path: Path
    status_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class MetaExtractionResponse:
    data: dict[str, Any]
    provider: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    cost: dict[str, Any] | None = None


class MetaExtractor(Protocol):
    """后续统一 LLMClient 可通过此小接口接入。"""

    def extract(self, prompt: str, source_text: str) -> MetaExtractionResponse:
        ...


class ConfiguredMetaExtractor:
    """把 extraction.llm_client 适配为预处理元数据接口。"""

    def __init__(self, config_path: Path) -> None:
        if str(PACKAGE_ROOT) not in sys.path:
            sys.path.insert(0, str(PACKAGE_ROOT))
        from extraction.llm_client import LLMClient

        self.client = LLMClient.from_pipeline_config(
            stage="meta_extract",
            config_path=config_path,
        )
        self.last_usage: dict[str, int] | None = None
        self.last_cost: dict[str, Any] | None = None

    def extract(self, prompt: str, source_text: str) -> MetaExtractionResponse:
        from extraction.llm_client import summarize_client_calls

        history_start = len(self.client.call_history)
        try:
            response = self.client.call_json(
                prompt,
                "以下是首页及第二页的 MinerU OCR 文本：\n\n" + source_text,
            )
        finally:
            self.last_usage, self.last_cost = summarize_client_calls(
                self.client,
                history_start,
                call_count=1,
            )
        return MetaExtractionResponse(
            data=response.data,
            provider=response.provider,
            model=response.model,
            usage=self.last_usage,
            cost=self.last_cost,
        )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocumentGateError(f"无法读取有效 JSON：{path}") from exc


def write_json_atomic(path: Path, data: Any) -> None:
    def json_default(value: Any) -> str:
        if isinstance(value, Decimal):
            return str(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _find_single(
    paper_dir: Path,
    pattern: str,
    *,
    required: bool,
    label: str,
) -> Path | None:
    candidates = sorted(paper_dir.glob(pattern))
    if len(candidates) > 1:
        raise DocumentGateError(f"{label} 文件不唯一：{paper_dir}")
    if not candidates:
        if required:
            raise DocumentGateError(f"缺少 {label}：{paper_dir}")
        return None
    return candidates[0]


def discover_mineru_files(mineru_output: Path, ref_no: str) -> MineruFiles:
    paper_dir = mineru_output / ref_no
    if not paper_dir.is_dir():
        raise DocumentGateError(f"缺少 MinerU 文献目录：{paper_dir}")

    markdown = _find_single(paper_dir, "*.md", required=True, label="Markdown")
    content_v1 = _find_single(
        paper_dir,
        "*_content_list.json",
        required=True,
        label="content_list v1",
    )
    content_v2 = _find_single(
        paper_dir,
        "*_content_list_v2.json",
        required=False,
        label="content_list v2",
    )
    pdf = _find_single(paper_dir, "*_origin.pdf", required=True, label="origin PDF")
    image_dirs = sorted(
        path for path in paper_dir.iterdir()
        if path.is_dir() and path.name.endswith("_images")
    )
    if len(image_dirs) > 1:
        raise DocumentGateError(f"图片目录不唯一：{paper_dir}")

    return MineruFiles(
        ref_no=ref_no,
        paper_dir=paper_dir,
        markdown=markdown,
        content_v1=content_v1,
        content_v2=content_v2,
        pdf=pdf,
        images_dir=image_dirs[0] if image_dirs else None,
    )


def _matching_result(status: dict[str, Any], ref_no: str) -> dict[str, Any] | None:
    results = ((status.get("data") or {}).get("extract_result") or [])
    for item in results:
        file_name = str(item.get("file_name") or "")
        if Path(file_name).stem == ref_no:
            return item
    return None


def find_batch_context(mineru_output: Path, ref_no: str) -> BatchContext:
    matches: list[tuple[float, Path, dict[str, Any], dict[str, Any]]] = []
    for status_path in mineru_output.glob("batch_*_status.json"):
        status = read_json(status_path)
        if not isinstance(status, dict):
            continue
        result = _matching_result(status, ref_no)
        if result is not None:
            matches.append((status_path.stat().st_mtime, status_path, status, result))
    if not matches:
        raise DocumentGateError(f"批次状态中找不到文献：{ref_no}")

    _, status_path, status, result = max(matches, key=lambda item: item[0])
    state = str(result.get("state") or "unknown")
    if state != "done":
        raise DocumentGateError(f"MinerU 状态不是 done：{ref_no} ({state})")

    batch_id = str((status.get("data") or {}).get("batch_id") or "")
    if not batch_id:
        match = re.match(r"batch_(.+)_status\.json$", status_path.name)
        batch_id = match.group(1) if match else ""
    if not batch_id:
        raise DocumentGateError(f"批次状态缺少 batch_id：{status_path.name}")

    manifest_path = mineru_output / f"batch_{batch_id}_manifest.json"
    if not manifest_path.is_file():
        raise DocumentGateError(f"缺少批次 manifest：{manifest_path.name}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise DocumentGateError(f"批次 manifest 不是 JSON 对象：{manifest_path.name}")

    return BatchContext(
        batch_id=batch_id,
        state=state,
        manifest_path=manifest_path,
        status_path=status_path,
        manifest=manifest,
    )


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _resolve_raw_image(files: MineruFiles, raw_path: str) -> Path | None:
    clean_path = raw_path.strip().strip("<>")
    if not clean_path or re.match(r"^(?:https?:|data:)", clean_path, re.IGNORECASE):
        return None
    direct = files.paper_dir / Path(clean_path)
    if direct.is_file():
        return direct
    if files.images_dir:
        by_name = files.images_dir / Path(clean_path).name
        if by_name.is_file():
            return by_name
    return None


def validate_image_references(
    files: MineruFiles,
    blocks: list[dict[str, Any]],
    markdown_text: str,
) -> None:
    referenced_paths = [
        str(block["img_path"])
        for block in blocks
        if block.get("img_path")
    ]
    referenced_paths.extend(
        match.group("path") for match in MARKDOWN_IMAGE_RE.finditer(markdown_text)
    )
    missing = sorted({
        Path(path.strip().strip("<>")).name
        for path in referenced_paths
        if _resolve_raw_image(files, path) is None
    })
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise DocumentGateError(f"缺少正文明确引用的图片：{preview}{suffix}")


def validate_organized_files(organized_root: Path, ref_no: str) -> None:
    organized_dir = organized_root / ref_no
    required = (
        organized_dir / "content.json",
        organized_dir / f"{ref_no}.md",
        organized_dir / "origin.pdf",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise DocumentGateError(
            f"Stage -1 产物不完整：{ref_no} 缺少 " + ", ".join(missing)
        )


def _content_sequence(v2_block: dict[str, Any]) -> list[dict[str, Any]]:
    content = v2_block.get("content")
    if not isinstance(content, dict):
        return []
    for key, value in content.items():
        if key.endswith("_content") and isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _render_v2_text(v2_block: dict[str, Any]) -> tuple[str, list[int]]:
    parts: list[str] = []
    inline_indexes: list[int] = []
    for index, item in enumerate(_content_sequence(v2_block)):
        item_type = str(item.get("type") or "")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if item_type == "equation_inline":
            parts.append(f"${content}$")
            inline_indexes.append(index)
        elif item_type == "text":
            parts.append(content)
    return "".join(parts), inline_indexes


def build_v2_index(
    content_v2: Any,
) -> dict[tuple[int, tuple[Any, ...]], list[tuple[int, dict[str, Any]]]]:
    index: dict[tuple[int, tuple[Any, ...]], list[tuple[int, dict[str, Any]]]] = {}
    if not isinstance(content_v2, list):
        return index

    if content_v2 and all(isinstance(page, list) for page in content_v2):
        page_groups = enumerate(content_v2)
    else:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for block in content_v2:
            if isinstance(block, dict):
                grouped.setdefault(int(block.get("page_idx", 0)), []).append(block)
        page_groups = sorted(grouped.items())

    for page_idx, page in page_groups:
        for block_index, block in enumerate(page):
            if not isinstance(block, dict):
                continue
            bbox = block.get("bbox")
            if not isinstance(bbox, list):
                continue
            key = (int(page_idx), tuple(bbox))
            index.setdefault(key, []).append((block_index, block))
    return index


def _normalized_html(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _caption_text(block: dict[str, Any]) -> str | None:
    for key in ("table_caption", "chart_caption", "image_caption"):
        value = block.get(key)
        if isinstance(value, list):
            text = " ".join(str(item) for item in value if item is not None).strip()
            if text:
                return text
        elif isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _block_prefix(element_type: str) -> str:
    return {
        "table": "T",
        "image": "I",
        "equation": "E",
        "references": "R",
        "footnote": "F",
    }.get(element_type, "P")


def _organized_blocks(organized_root: Path, ref_no: str) -> list[dict[str, Any]]:
    path = organized_root / ref_no / "content.json"
    if not path.is_file():
        return []
    data = read_json(path)
    blocks = data.get("blocks") if isinstance(data, dict) else None
    return blocks if isinstance(blocks, list) else []


def _organized_image_path(
    organized_root: Path,
    ref_no: str,
    organized_blocks: list[dict[str, Any]],
    block_index: int,
    raw_path: str | None,
) -> str | None:
    normalized_path = None
    if block_index < len(organized_blocks):
        candidate = organized_blocks[block_index]
        if isinstance(candidate, dict) and candidate.get("img_path"):
            normalized_path = str(candidate["img_path"])
    normalized_path = normalized_path or raw_path
    if not normalized_path:
        return None
    return str(PurePosixPath(organized_root.name) / ref_no / normalized_path)


def validate_organized_image_paths(
    elements: list[dict[str, Any]],
    organized_root: Path,
    ref_no: str,
) -> None:
    prefix = PurePosixPath(organized_root.name) / ref_no
    missing: set[str] = set()
    for element in elements:
        image_path = element.get("image_path")
        if not image_path:
            continue
        try:
            relative = PurePosixPath(str(image_path)).relative_to(prefix)
        except ValueError:
            missing.add(str(image_path))
            continue
        target = organized_root / ref_no / Path(*relative.parts)
        if not target.is_file():
            missing.add(str(image_path))
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise DocumentGateError(f"Stage -1 图片引用无效：{preview}{suffix}")


def standardize_elements(
    blocks: list[dict[str, Any]],
    content_v2: Any,
    markdown_text: str,
    organized_root: Path,
    ref_no: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    elements: list[dict[str, Any]] = []
    v2_index = build_v2_index(content_v2)
    markdown_tables = HTML_TABLE_RE.findall(markdown_text)
    used_markdown_tables: set[int] = set()
    organized_blocks = _organized_blocks(organized_root, ref_no)
    current_section: str | None = None
    document_title_seen = False

    for block_index, block in enumerate(blocks):
        block_type = str(block.get("type") or "")
        if block_type in SKIPPED_BLOCK_TYPES:
            continue
        page_id = int(block.get("page_idx", 0))
        bbox = block.get("bbox")
        bbox_value = bbox if isinstance(bbox, list) else None
        key = (page_id, tuple(bbox_value or []))
        v2_matches = v2_index.get(key, [])
        alignment_status = "matched" if v2_matches else "unresolved"

        if block_type == "text":
            element_type = "title" if block.get("text_level") is not None else "text"
        elif block_type == "table":
            element_type = "table"
        elif block_type in {"chart", "image"}:
            element_type = "image"
        elif block_type == "equation":
            element_type = "equation"
        elif block_type == "ref_text":
            element_type = "references"
        elif block_type == "page_footnote":
            element_type = "footnote"
        elif block_type == "aside_text":
            element_type = "text"
        else:
            element_type = "unknown"
            warnings.append(_warning(
                "unknown_block_type",
                f"block_index={block_index} 保留了未知类型 {block_type!r}",
            ))

        prefix = _block_prefix(element_type)
        block_id = f"{prefix}_{page_id}_{block_index}"
        element: dict[str, Any] = {
            "block_id": block_id,
            "page_id": page_id,
            "block_index": block_index,
            "element_type": element_type,
            "bbox": bbox_value,
            "alignment_status": alignment_status,
        }

        if element_type in {"text", "title", "references", "footnote"}:
            text = str(block.get("text") or "")
            merged_source_ids: list[str] = []
            for v2_block_index, v2_block in v2_matches:
                rendered, inline_indexes = _render_v2_text(v2_block)
                if inline_indexes and rendered:
                    text = rendered
                    merged_source_ids.append(block_id)
                    merged_source_ids.extend(
                        f"V2_{page_id}_{v2_block_index}_{child_index}"
                        for child_index in inline_indexes
                    )
                    break
            element["text"] = text
            if merged_source_ids:
                element["merged_source_block_ids"] = merged_source_ids
            if block_type == "aside_text":
                element["element_subtype"] = "aside"
            if element_type == "title":
                level = int(block.get("text_level") or 1)
                element["title_level"] = level
                if level == 1 and not document_title_seen:
                    element["section"] = "DocumentTitle"
                    document_title_seen = True
                else:
                    element["section"] = text or current_section
                current_section = text or current_section
            else:
                element["section"] = current_section

        elif element_type == "table":
            raw_body = str(block.get("table_body") or "")
            table_body = raw_body
            matched_table_index = next(
                (
                    index for index, candidate in enumerate(markdown_tables)
                    if index not in used_markdown_tables
                    and _normalized_html(candidate) == _normalized_html(raw_body)
                ),
                None,
            )
            if matched_table_index is None:
                matched_table_index = next(
                    (
                        index for index in range(len(markdown_tables))
                        if index not in used_markdown_tables
                    ),
                    None,
                )
                alignment_status = "unresolved"
            if matched_table_index is not None:
                used_markdown_tables.add(matched_table_index)
                table_body = markdown_tables[matched_table_index]
            else:
                warnings.append(_warning(
                    "table_markdown_unmatched",
                    f"{block_id} 未在 Markdown 中找到表格，已保留 v1 table_body",
                ))
            element.update({
                "caption": _caption_text(block),
                "table_body": table_body,
                "image_path": _organized_image_path(
                    organized_root,
                    ref_no,
                    organized_blocks,
                    block_index,
                    str(block.get("img_path") or "") or None,
                ),
                "alignment_status": alignment_status,
            })
            if alignment_status == "unresolved":
                warnings.append(_warning(
                    "table_alignment_unresolved",
                    f"{block_id} 的 Markdown 表格仅按顺序对齐",
                ))

        elif element_type == "image":
            element.update({
                "image_kind": block_type,
                "caption": _caption_text(block),
                "image_path": _organized_image_path(
                    organized_root,
                    ref_no,
                    organized_blocks,
                    block_index,
                    str(block.get("img_path") or "") or None,
                ),
            })
            content = block.get("content")
            if content not in (None, ""):
                element["content"] = content

        elif element_type == "equation":
            has_display_match = any(
                str(v2_block.get("type")) == "equation_interline"
                for _, v2_block in v2_matches
            )
            element.update({
                "text": str(block.get("text") or ""),
                "equation_kind": "display" if has_display_match else "unresolved",
                "alignment_status": "matched" if has_display_match else "unresolved",
                "section": current_section,
            })
            if not has_display_match:
                warnings.append(_warning(
                    "equation_alignment_unresolved",
                    f"{block_id} 无法与 v2 独立公式可靠对齐，已保留原文",
                ))
        else:
            element["raw"] = block

        elements.append(element)

    unused_table_count = len(markdown_tables) - len(used_markdown_tables)
    if unused_table_count:
        warnings.append(_warning(
            "unused_markdown_tables",
            f"Markdown 中有 {unused_table_count} 个表格未与 v1 block 对齐",
        ))
    return elements, warnings


def _meta_source(blocks: list[dict[str, Any]]) -> tuple[str, list[int]]:
    candidates: list[tuple[int, int, str]] = []
    allowed_types = {"text", "header", "footer", "page_header", "page_footer"}
    for block_index, block in enumerate(blocks):
        page_idx = int(block.get("page_idx", 0))
        if page_idx > 1 or str(block.get("type") or "") not in allowed_types:
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            candidates.append((page_idx, block_index, text.strip()))
    candidates.sort(key=lambda item: (item[0], item[1]))
    pages = sorted({item[0] for item in candidates})
    source_text = "\n".join(item[2] for item in candidates)
    return source_text, pages


def validate_meta_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != META_OUTPUT_FIELDS:
        raise ValueError("元数据响应必须且只能包含固定 5 个字段和 confidence")

    for field in ("doi", "title", "journal"):
        value = payload[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} 必须是字符串或 null")
    authors = payload["authors"]
    if authors is not None and (
        not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, str) or not author.strip() for author in authors)
    ):
        raise ValueError("authors 必须是非空字符串数组或 null")
    year = payload["year"]
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise ValueError("year 必须是整数或 null")
    return {
        "doi": payload["doi"],
        "title": payload["title"],
        "authors": payload["authors"],
        "journal": payload["journal"],
        "year": payload["year"],
    }


def validate_meta_confidence(payload: Any) -> dict[str, Any]:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from extraction.schema.polymer_schema import MetadataConfidence

    confidence = MetadataConfidence.model_validate(payload)
    referenced = {
        field_path.split(".", 1)[0]
        for field_path in (
            *confidence.uncertain_fields,
            *confidence.field_scores,
        )
    }
    unknown = sorted(referenced - META_FIELDS)
    if unknown:
        raise ValueError(f"元数据 confidence 引用了未知字段：{unknown}")
    return confidence.model_dump(mode="json")


def _empty_meta() -> dict[str, Any]:
    return {
        "doi": None,
        "title": None,
        "authors": None,
        "journal": None,
        "year": None,
    }


def extract_paper_meta(
    blocks: list[dict[str, Any]],
    extractor: MetaExtractor | None,
) -> tuple[dict[str, Any], str, dict[str, Any], list[dict[str, str]]]:
    prompt = META_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    source_text, source_pages = _meta_source(blocks)
    extraction: dict[str, Any] = {
        "method": "llm",
        "provider": None,
        "model": None,
        "prompt_file": "prompts/meta_extract.md",
        "prompt_sha256": prompt_sha256,
        "source_pages": source_pages,
    }
    if extractor is None:
        extraction.update({
            "status": "failed",
            "error_type": "MetaExtractorUnavailable",
        })
        return (
            _empty_meta(),
            "failed",
            extraction,
            [_warning(
                "metadata_extraction_failed",
                "未配置统一 LLM 元数据提取器；document JSON 已继续生成",
            )],
        )

    try:
        response = extractor.extract(prompt, source_text)
        meta = validate_meta_payload(response.data)
        confidence = validate_meta_confidence(
            response.data.get("confidence")
        )
    except Exception as exc:
        extraction.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "usage": getattr(extractor, "last_usage", None),
            "cost": getattr(extractor, "last_cost", None),
        })
        return (
            _empty_meta(),
            "failed",
            extraction,
            [_warning(
                "metadata_extraction_failed",
                "元数据 LLM 调用或固定字段校验失败；document JSON 已继续生成",
            )],
        )

    extraction.update({
        "provider": response.provider,
        "model": response.model,
        "confidence": confidence,
        "usage": response.usage,
        "cost": response.cost,
        "status": "success",
    })
    status = "complete" if all(value is not None for value in meta.values()) else "partial"
    return meta, status, extraction, []


def _source_path(root_name: str, ref_no: str, filename: str) -> str:
    return str(PurePosixPath(root_name) / ref_no / filename)


def _build_paper(
    files: MineruFiles,
    organized_root: Path,
    blocks: list[dict[str, Any]],
    existing_document: dict[str, Any] | None,
    force_meta: bool,
    meta_extractor: MetaExtractor | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    cached_paper = (
        existing_document.get("paper")
        if isinstance(existing_document, dict)
        and isinstance(existing_document.get("paper"), dict)
        else None
    )
    if (
        cached_paper
        and cached_paper.get("metadata_status") == "complete"
        and not force_meta
    ):
        return dict(cached_paper), []

    meta, status, extraction, warnings = extract_paper_meta(blocks, meta_extractor)
    paper = {
        "ref_no": files.ref_no,
        "pdf_filename": files.pdf.name,
        "source_pdf_path": _source_path(
            files.paper_dir.parent.name,
            files.ref_no,
            files.pdf.name,
        ),
        "organized_pdf_path": _source_path(
            organized_root.name,
            files.ref_no,
            "origin.pdf",
        ),
        **meta,
        "metadata_status": status,
        "metadata_extraction": extraction,
    }
    return paper, warnings


def _source_files(
    files: MineruFiles,
    organized_root: Path,
) -> dict[str, str | None]:
    organized_dir = organized_root / files.ref_no
    return {
        "markdown": (
            f"{files.ref_no}.md"
            if (organized_dir / f"{files.ref_no}.md").is_file()
            else files.markdown.name
        ),
        "content_v1": (
            "content.json"
            if (organized_dir / "content.json").is_file()
            else files.content_v1.name
        ),
        "content_v2": (
            "content_v2.json"
            if (organized_dir / "content_v2.json").is_file()
            else files.content_v2.name if files.content_v2 else None
        ),
        "pdf": (
            "origin.pdf"
            if (organized_dir / "origin.pdf").is_file()
            else files.pdf.name
        ),
    }


def _ocr_record(
    context: BatchContext,
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    option_fields = (
        "ocr_enabled",
        "language",
        "page_ranges",
        "enable_formula",
        "enable_table",
        "extra_formats",
    )
    missing_fields = [field for field in option_fields if field not in context.manifest]
    if missing_fields:
        warnings.append(_warning(
            "legacy_manifest_missing_options",
            "旧 manifest 缺少 OCR 选项，已按 null 保留：" + ", ".join(missing_fields),
        ))
    return {
        "engine": "mineru",
        "batch_id": context.batch_id,
        "model_version": context.manifest.get("model_version"),
        "ocr_enabled": context.manifest.get("ocr_enabled"),
        "language": context.manifest.get("language"),
        "page_ranges": context.manifest.get("page_ranges"),
        "enable_formula": context.manifest.get("enable_formula"),
        "enable_table": context.manifest.get("enable_table"),
        "extra_formats": context.manifest.get("extra_formats"),
        "status": context.state,
        "manifest_file": context.manifest_path.name,
        "status_file": context.status_path.name,
    }


def transform_paper(
    mineru_output: Path,
    organized_root: Path,
    processed_output: Path,
    ref_no: str,
    *,
    force_meta: bool = False,
    meta_extractor: MetaExtractor | None = None,
) -> Path:
    files = discover_mineru_files(mineru_output, ref_no)
    context = find_batch_context(mineru_output, ref_no)
    validate_organized_files(organized_root, ref_no)
    blocks = read_json(files.content_v1)
    if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
        raise DocumentGateError(f"content_list v1 必须是对象数组：{files.content_v1}")
    markdown_text = files.markdown.read_text(encoding="utf-8-sig")
    validate_image_references(files, blocks, markdown_text)

    warnings: list[dict[str, str]] = []
    if files.content_v2:
        content_v2 = read_json(files.content_v2)
    else:
        content_v2 = []
        warnings.append(_warning(
            "content_v2_missing",
            "缺少 content_list_v2.json；公式和阅读顺序对齐将标记 unresolved",
        ))

    elements, element_warnings = standardize_elements(
        blocks,
        content_v2,
        markdown_text,
        organized_root,
        ref_no,
    )
    validate_organized_image_paths(elements, organized_root, ref_no)
    warnings.extend(element_warnings)

    output_path = processed_output / "documents" / f"{ref_no}_document.json"
    existing_document = read_json(output_path) if output_path.is_file() else None
    paper, meta_warnings = _build_paper(
        files,
        organized_root,
        blocks,
        existing_document,
        force_meta,
        meta_extractor,
    )
    warnings.extend(meta_warnings)

    document = {
        "schema_version": "1.0",
        "document_id": ref_no,
        "paper": paper,
        "source_files": _source_files(files, organized_root),
        "ocr": _ocr_record(context, warnings),
        "elements": elements,
        "warnings": warnings,
    }
    write_json_atomic(output_path, document)
    return output_path


def transform_batch(
    mineru_output: Path,
    organized_root: Path,
    processed_output: Path,
    *,
    ref_no: str | None = None,
    force_meta: bool = False,
    meta_extractor: MetaExtractor | None = None,
) -> tuple[list[Path], list[tuple[str, str]]]:
    if not mineru_output.is_dir():
        raise FileNotFoundError(f"MinerU 输出目录不存在：{mineru_output}")
    if ref_no:
        ref_nos = [ref_no]
    else:
        ref_nos = sorted(
            path.name for path in mineru_output.iterdir()
            if path.is_dir() and path.name.startswith("reference_no_")
        )

    outputs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for current_ref_no in ref_nos:
        try:
            output = transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                current_ref_no,
                force_meta=force_meta,
                meta_extractor=meta_extractor,
            )
            outputs.append(output)
            print(f"[done] {current_ref_no} -> {output}")
        except Exception as exc:
            failures.append((current_ref_no, str(exc)))
            print(f"[failed] {current_ref_no}: {exc}", file=sys.stderr)
    return outputs, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 MinerU 输出转换为标准化 document JSON",
    )
    parser.add_argument("--mineru-output", type=Path, default=DEFAULT_MINERU_OUTPUT)
    parser.add_argument("--organized-root", type=Path, default=DEFAULT_ORGANIZED_ROOT)
    parser.add_argument("--processed-output", type=Path, default=DEFAULT_PROCESSED_OUTPUT)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
    )
    parser.add_argument("--ref-no")
    parser.add_argument(
        "--force-meta",
        action="store_true",
        help="忽略 complete 元数据缓存并重新提取",
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="不调用 LLM；Paper 元数据按 failed 状态落盘",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    meta_extractor: MetaExtractor | None = None
    if not args.skip_meta:
        try:
            meta_extractor = ConfiguredMetaExtractor(
                args.pipeline_config.expanduser().resolve()
            )
        except Exception as exc:
            print(
                f"[warning] 元数据 LLM 未启用：{type(exc).__name__}",
                file=sys.stderr,
            )
    outputs, failures = transform_batch(
        args.mineru_output.expanduser().resolve(),
        args.organized_root.expanduser().resolve(),
        args.processed_output.expanduser().resolve(),
        ref_no=args.ref_no,
        force_meta=args.force_meta,
        meta_extractor=meta_extractor,
    )
    print(f"完成：成功 {len(outputs)} 篇，失败 {len(failures)} 篇")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
