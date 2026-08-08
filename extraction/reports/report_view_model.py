"""为 HTML 报告构建只读展示数据，不修改抽取结果 Schema。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[3]

_FIGURE_LABEL_RE = re.compile(
    r"\b(?:fig(?:ure)?\.?)\s*([A-Za-z]?\d+[A-Za-z]?)",
    re.IGNORECASE,
)
_OBJECT_COLLECTIONS = (
    "material_mentions",
    "polymer_entities",
    "samples",
    "process_steps",
    "property_observations",
    "measurement_conditions",
    "characterizations",
)
_OBJECT_ID_FIELDS = (
    "mention_id",
    "entity_id",
    "sample_id",
    "step_id",
    "property_id",
    "condition_id",
    "characterization_id",
)


def _nonempty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _object_id(item: dict[str, Any]) -> str | None:
    for field in _OBJECT_ID_FIELDS:
        value = _nonempty(item.get(field))
        if value:
            return value
    return None


def _safe_image_url(
    image_path: str | None,
    *,
    project_root: Path,
    report_dir: Path,
) -> tuple[str | None, bool]:
    """返回报告可用的本地 URL；拒绝项目目录之外的路径。"""

    if not image_path:
        return None, False
    stored_path = Path(image_path)
    candidate = (
        stored_path
        if stored_path.is_absolute()
        else project_root / stored_path
    ).resolve()
    root = project_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, False
    try:
        relative = Path(
            os.path.relpath(candidate, report_dir.resolve())
        ).as_posix()
    except ValueError:
        return candidate.as_uri(), candidate.is_file()
    return quote(relative, safe="/:."), candidate.is_file()


def _figure_label(caption: str | None, fallback_number: int) -> str:
    if caption:
        match = _FIGURE_LABEL_RE.search(caption)
        if match:
            return f"Fig. {match.group(1)}"
    return f"图组 {fallback_number}"


def build_figure_groups(
    stage0_data: dict[str, Any] | None,
    *,
    project_root: Path = PROJECT_ROOT,
    report_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按同页、连续 image block 和随附图注保守构建 Figure 分组。"""

    if not stage0_data:
        return [], []

    figures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    previous_source_index: int | None = None
    previous_page: int | None = None

    def flush() -> None:
        nonlocal pending, previous_source_index, previous_page
        if not pending:
            return

        figure_number = len(figures) + 1
        figure_id = f"fig{figure_number:03d}"
        caption_block = next(
            (
                block
                for block in reversed(pending)
                if _nonempty(block.get("caption"))
            ),
            None,
        )
        caption = (
            _nonempty(caption_block.get("caption"))
            if caption_block is not None
            else None
        )
        images: list[dict[str, Any]] = []
        for block in pending:
            image_path = _nonempty(block.get("image_path"))
            image_url, image_exists = _safe_image_url(
                image_path,
                project_root=project_root,
                report_dir=report_dir,
            )
            images.append({
                "block_id": block.get("block_id"),
                "page": block.get("page"),
                "bbox": block.get("bbox"),
                "source_block_index": block.get("source_block_index"),
                "image_kind": block.get("image_kind"),
                "image_path": image_path,
                "image_url": image_url,
                "image_exists": image_exists,
            })

        if caption:
            association_status = (
                "explicit_caption"
                if len(pending) == 1
                else "grouped_by_adjacency"
            )
        else:
            association_status = "caption_missing"

        figure = {
            "figure_id": figure_id,
            "label": _figure_label(caption, figure_number),
            "page": pending[0].get("page"),
            "image_block_ids": [
                block.get("block_id")
                for block in pending
                if block.get("block_id")
            ],
            "caption_source_block_id": (
                caption_block.get("block_id")
                if caption_block is not None
                else None
            ),
            "caption": caption,
            "association_status": association_status,
            "images": images,
            "linked_evidence_ids": [],
            "linked_object_ids": [],
            "link_basis": [],
        }
        figures.append(figure)
        if caption is None:
            warnings.append({
                "code": "figure_group_caption_missing",
                "message": (
                    f"{figure_id} 的 {len(pending)} 个连续图片块没有可用图注"
                ),
                "object_id": figure_id,
            })

        pending = []
        previous_source_index = None
        previous_page = None

    for element in stage0_data.get("elements") or []:
        if not isinstance(element, dict) or element.get("type") != "image":
            flush()
            continue

        page = element.get("page")
        source_index = element.get("source_block_index")
        remains_adjacent = (
            bool(pending)
            and page == previous_page
            and isinstance(source_index, int)
            and isinstance(previous_source_index, int)
            and source_index == previous_source_index + 1
        )
        if pending and not remains_adjacent:
            flush()

        pending.append(element)
        previous_page = page
        previous_source_index = (
            source_index if isinstance(source_index, int) else None
        )
        if _nonempty(element.get("caption")):
            flush()

    flush()
    return figures, warnings


def build_report_context(
    final_data: dict[str, Any],
    stage0_data: dict[str, Any] | None,
    *,
    project_root: Path = PROJECT_ROOT,
    report_dir: Path,
) -> dict[str, Any]:
    """生成 Figure 展示数据和基于明确 block/evidence 引用的关系。"""

    figures, figure_warnings = build_figure_groups(
        stage0_data,
        project_root=project_root,
        report_dir=report_dir,
    )
    table_sources = {
        str(element["block_id"]): str(element["table_body"])
        for element in (stage0_data or {}).get("elements") or []
        if isinstance(element, dict)
        and element.get("type") == "table"
        and _nonempty(element.get("block_id"))
        and _nonempty(element.get("table_body"))
    }
    figure_by_block: dict[str, dict[str, Any]] = {}
    for figure in figures:
        for block_id in figure["image_block_ids"]:
            figure_by_block[block_id] = figure

    evidence_to_figure: dict[str, str] = {}
    for evidence in final_data.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        evidence_id = _nonempty(evidence.get("evidence_id"))
        block_id = _nonempty(evidence.get("block_id"))
        figure = figure_by_block.get(block_id or "")
        if not evidence_id or figure is None:
            continue
        evidence_to_figure[evidence_id] = figure["figure_id"]
        figure["linked_evidence_ids"].append(evidence_id)
        figure["link_basis"].append("image_evidence")

    for collection_name in _OBJECT_COLLECTIONS:
        for item in final_data.get(collection_name) or []:
            if not isinstance(item, dict):
                continue
            object_id = _object_id(item)
            if not object_id:
                continue
            linked_figures: set[str] = set()
            for evidence_id in item.get("evidence_ids") or []:
                figure_id = evidence_to_figure.get(str(evidence_id))
                if figure_id:
                    linked_figures.add(figure_id)
            for image_ref in item.get("source_image_refs") or []:
                if not isinstance(image_ref, dict):
                    continue
                figure = figure_by_block.get(
                    str(image_ref.get("block_id") or "")
                )
                if figure is not None:
                    linked_figures.add(figure["figure_id"])
                    figure["link_basis"].append("source_image_ref")
            for figure_id in linked_figures:
                figure = next(
                    item
                    for item in figures
                    if item["figure_id"] == figure_id
                )
                figure["linked_object_ids"].append(object_id)

    for figure in figures:
        for field in (
            "linked_evidence_ids",
            "linked_object_ids",
            "link_basis",
        ):
            figure[field] = sorted(set(figure[field]))

    return {
        "figures": figures,
        "figure_warnings": figure_warnings,
        "evidence_to_figure": evidence_to_figure,
        "table_sources": table_sources,
        "stage0_available": stage0_data is not None,
    }
