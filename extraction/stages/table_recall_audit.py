"""Stage 4 表格单元格召回离线审计。

本模块只读取 Stage 0/Stage 4 JSON，不修改抽取结果，也不调用 LLM。
它从 Stage 0 的稳定 table cell 网格建立候选单元格，并区分：
property value / coordinate / condition / unknown。Stage 4 覆盖仅由真实
property value evidence 计算，避免把 coordinate 或 condition 误算成性质召回。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from schema.polymer_schema import Stage0Document, Stage0Element, Stage0TableCell
from stages.table_grid import table_cells_for


AUDIT_VERSION = "1.0.0"
DEFAULT_VOCABULARY_PATH = EXTRACTION_ROOT / "config" / "polymer_schema.yaml"
DEFAULT_THRESHOLD = 0.8

ROLE_PROPERTY = "property_value_candidate"
ROLE_COORDINATE = "coordinate_candidate"
ROLE_CONDITION = "condition_candidate"
ROLE_UNKNOWN = "unknown"

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])"
    r"[<>≤≥~≈]?\s*[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)"
    r"(?:\s*(?:±|\+/-)\s*\d+(?:[.,]\d*)?)?"
    r"(?:\s*(?:[x×·]\s*)?10\s*(?:\^|\*\*)?\s*[−-]?\d+|[eE][+-]?\d+)?"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z]{1,8}[-_]?\d+(?:[-_]\d+)*$")

# 这里不建立第二份97项Schema，只补充论文表头中常见、无法由normalized_name
# 直接恢复的缩写。正式范围仍来自 config/polymer_schema.yaml。
_PROPERTY_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:electrical?|ionic|dc|ac)?\s*conductivit(?:y|ies)\b|(?:^|\W)(?:\\sigma|σ)(?:\W|$)|电导率", "electric_conductivity"),
    (r"\bvolume\s+resistivit(?:y|ies)\b|体积电阻率", "volume_resistivity"),
    (r"\bsurface\s+resistivit(?:y|ies)\b|表面电阻率", "surface_resistivity"),
    (r"\bdielectric\s+(?:constant|permittivity)\b|介电常数", "dielectric_constant_dc"),
    (r"\bdielectric\s+(?:breakdown|strength)\b|击穿", "dielectric_breakdown_voltage"),
    (r"(?:^|\W)t\s*[_-]?g(?:\W|$)|glass\s+transition|玻璃化转变", "glass_transition_temperature"),
    (r"(?:^|\W)t\s*[_-]?m(?:\W|$)|(?:^|\W)m\.?\s*p\.?(?:\W|$)|(?:^|\W)pmt(?:\W|$)|melting\s+(?:point|temperature)|熔融温度|熔点", "melting_temperature"),
    (r"(?:^|\W)t\s*[_-]?c(?:\W|$)|crystallization\s+temperature|结晶温度", "crystallization_temperature"),
    (r"(?:^|\W)t\s*[_-]?d(?:\s*\d+\s*%?)?(?:\W|$)|decomposition\s+temperature|thermal\s+degradation|热分解|"
     r"(?:^|\W)t\s*_?\s*\{?\s*\d+(?:\.\d+)?\s*\\?%", "thermal_decomposition_temperature"),
    (r"\b(?:surface\s+tension|surface\s+energy)\b|表面张力|表面能", "surface_tension"),
    (r"\binterfacial\s+tension\b|界面张力", "interfacial_tension"),
    (r"\btensile\s+(?:stress|strength)\s+(?:at\s+)?break\b|断裂拉伸强度|拉伸断裂强度", "tensile_stress_at_break"),
    (r"\btensile\s+(?:stress|strength)\s+(?:at\s+)?yield\b|屈服拉伸强度", "tensile_stress_at_yield"),
    (r"\btensile\s+modulus\b|拉伸模量", "tensile_modulus"),
    (r"\b(?:dynamic\s+)?storage\s+modulus\b|(?:^|\W)e\s*['′](?:\W|$)|储能模量", "dynamic_tensile_properties"),
    (r"\belongation\s+(?:at\s+)?break\b|断裂伸长率", "elongation_at_break"),
    (r"\bwater\s+(?:absorption|uptake)\b|吸水率", "water_absorption"),
    (r"\bthermal\s+conductivit(?:y|ies)\b|导热率|热导率", "thermal_conductivity"),
    (r"\bthermal\s+diffusivit(?:y|ies)\b|热扩散率", "thermal_diffusivity"),
    (r"\brefractive\s+index\b|折射率", "refractive_index"),
    (r"\bintrinsic\s+viscosity\b|\binherent\s+viscosity\b|特性黏度|特性粘度|(?:^|\W)(?:\\eta|eta|η)\s*(?:_?\s*\{?\s*(?:inh|int|sp|red)\b)?", "intrinsic_viscosity"),
    (r"\bdensit(?:y|ies)\b|密度", "density"),
    (r"\bspecific\s+volume\b|比容", "specific_volume"),
    (r"\boxygen\s+index\b|\bloi\b|氧指数", "oxygen_index"),
    (r"\bvicat\b|维卡", "vicat_softening_temperature"),
    (r"\bsoftening\s+temperature\b|软化温度", "softening_temperature"),
    (r"\bizod\b|悬臂梁冲击", "izod_impact"),
    (r"\bcharpy\b|简支梁冲击", "charpy_impact"),
    # 裸写 "Impact strength (J/m)" 是论文表头最常见写法；08_名称映射 中
    # pipeline 侧同样归到 izod_impact（词表无 impact_strength 键）。
    (r"\bimpact\s+(?:strength|resistance)\b|\bimpact\s+energy\b|冲击强度", "izod_impact"),
    # 裸写 "Tensile strength (MPa)"，08_名称映射 归到 tensile_stress_at_break。
    # 带 at yield 的已由上面的专用别名先行命中，这里只兜底裸写形式。
    (r"\btensile\s+(?:strength|stress)\b|\bts\b|拉伸强度|抗张强度", "tensile_stress_at_break"),
    (r"\bflexural\s+(?:strength|stress)\b|弯曲强度", "flexural_stress_at_break"),
    (r"\bflexural\s+modulus\b|弯曲模量", "flexural_modulus"),
    (r"\byoung'?[’']?s\s+modul(?:us|i)\b|杨氏模量", "tensile_modulus"),
    (r"\b(?:loss\s+modulus)\b|损耗模量", "dynamic_tensile_properties"),
    (r"\bshore\s+hardness\b|邵氏硬度", "shore_hardness"),
    (r"\brockwell\s+hardness\b|洛氏硬度", "rockwell_hardness"),
    # 焓：论文表头基本不写英文全称，全批 5 处 ΔH 列没有一处是 "heat of fusion"。
    # 下标决定归属，所以带下标的两条必须排在裸写 ΔH 之前：
    #   ΔH_c / ΔH_cryst -> 结晶焓        ΔH_m / ΔH_f(usion) -> 熔融焓
    #   ΔH_exo（放热反应焓）、ΔH_s（混合/溶解焓）不在 97 项内，一律不认 ——
    #   所以裸写分支要求 ΔH 后面紧跟单位或行尾，不能吃掉任意下标。
    (r"\bheat\s+of\s+crystallization\b|\bcrystallization\s+enthalpy\b|结晶焓|"
     r"(?:^|\W)(?:\\delta|Δ)\s*h\s*[_-]?\s*(?:c|cryst\w*)(?:\W|$)", "heat_of_crystallization"),
    (r"\bheat\s+of\s+fusion\b|\benthalpy\s+of\s+fusion\b|熔融焓|"
     r"(?:^|\W)(?:\\delta|Δ)\s*h\s*[_-]?\s*(?:m|f|fus\w*)(?:\W|$)|"
     r"(?:^|\W)(?:\\delta|Δ)\s*h\s*(?=\s*[/(,]?\s*(?:k?j|k?cal)\b)", "heat_of_fusion"),
    (r"\b(?:residual\s+mass|residue|char\s+yield)\b|残炭率|残余质量", "thermal_decomposition_temperature"),
)

_COORDINATE_RE = re.compile(
    r"(?:^|\b)(?:time|temperature|temp\.?|frequency|freq\.?|pressure|strain|"
    r"concentration|composition|content|loading|wavelength|wave\s*number|"
    r"shear\s+rate|heating\s+rate|cooling\s+rate|dose|humidity|rh|ph|"
    r"sample|specimen|run|cycle|entry|no\.?)(?:\b|$)|"
    r"时间|温度|频率|压力|应变|浓度|含量|波长|波数|剪切速率|升温速率|湿度|样品|编号",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"(?:test|measurement|anneal(?:ing)?|dry(?:ing)?|cure|curing|aging|"
    r"atmosphere|solvent|medium|method|instrument|rate|duration|"
    r"temperature|pressure|frequency|humidity|ph)\b|"
    r"测试|测量|退火|干燥|固化|老化|气氛|溶剂|介质|方法|仪器|速率|时长|温度|压力|频率|湿度",
    re.IGNORECASE,
)


def _normalized_text(value: Any) -> str:
    text = html.unescape(_TAG_RE.sub(" ", str(value or "")))
    text = text.replace("$", " ").replace("\\text", " ")
    text = text.replace("{", " ").replace("}", " ").replace("_", " ")
    text = text.replace("−", "-").replace("–", "-")
    return _SPACE_RE.sub(" ", text).strip().casefold()


def _contains_number(value: str) -> bool:
    text = _normalized_text(value)
    if not text or not any(char.isdigit() for char in text):
        return False
    return _NUMBER_RE.search(text) is not None


def _looks_like_identifier(value: str) -> bool:
    return _IDENTIFIER_RE.fullmatch(_normalized_text(value).replace(" ", "")) is not None


_ALLOWED_VALUE_TOKENS = {
    "c", "k", "pa", "kpa", "mpa", "gpa", "hz", "khz", "mhz", "ghz",
    "s", "sec", "min", "h", "m", "mm", "cm", "nm", "um", "kg", "g",
    "mg", "mol", "mmol", "l", "ml", "dl", "j", "kj", "w", "kw",
    "wt", "vol", "ppm", "ppb", "x", "times", "e", "d",
}

# 纯排版/单位类 LaTeX 命令：出现在数据值里是正常的（误差项、度数、乘号），
# 不应据此判定"这格含未知字母、不是数值"。
# 注意只剥这些；\theta \chi \varepsilon 这类是变量名，剥掉会把表头误当数据值。
_LATEX_FORMAT_RE = re.compile(
    r"\\(?:pm|mp|circ|degree|times|cdot|sim|approx|leq|geq|ll|gg|"
    r"mathrm|mathbf|mathit|text|rm|it|bf|left|right|,|;|:|!|\s)",
    re.IGNORECASE,
)


def _strip_format_latex(text: str) -> str:
    return _SPACE_RE.sub(" ", _LATEX_FORMAT_RE.sub(" ", text)).strip()


def _is_value_like_numeric(value: str) -> bool:
    """识别表格数据值，排除带数字的表头、样品名和配比。"""

    text = _normalized_text(value)
    if not _contains_number(text):
        return False
    compact = text.replace(" ", "")
    if ":" in text and len(_NUMBER_RE.findall(text)) >= 2:
        return False
    if _looks_like_identifier(text):
        return False
    # 先剥排版类 LaTeX，"0.0032 \pm 6 \times 10^{-6}"、"56^{\circ}" 都是数据值。
    text_for_tokens = _strip_format_latex(text)
    # 诸如 25-75% AP-PCL、100% Amylopectin 是样品/配方，不是性质值。
    tokens = re.findall(r"[a-zA-Z]+", text_for_tokens)
    unknown_tokens = [token for token in tokens if token not in _ALLOWED_VALUE_TOKENS]
    if unknown_tokens:
        return False
    # 变量型表头（Tg、Tm、Td10%、1/Tmax）含公式字母和数字，但不是数据。
    if ("_" in str(value) or "\\" in str(value)) and any(
        token not in {"times", "e", "d"} for token in tokens
    ):
        return False
    return bool(_NUMBER_RE.search(compact))


def _infer_header_rows(cells: Sequence[Stage0TableCell]) -> set[int]:
    rows: dict[int, list[Stage0TableCell]] = {}
    for cell in cells:
        rows.setdefault(cell.row_index, []).append(cell)
    header_rows: set[int] = set()
    for row_index in sorted(rows):
        row = [cell for cell in rows[row_index] if cell.text.strip()]
        if not row:
            header_rows.add(row_index)
            continue
        value_count = sum(_is_value_like_numeric(cell.text) for cell in row)
        if value_count >= 1 and value_count / len(row) >= 0.4:
            break
        header_rows.add(row_index)
    return header_rows


def _load_property_patterns(path: Path = DEFAULT_VOCABULARY_PATH) -> list[tuple[re.Pattern[str], str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    vocabulary = payload.get("property_vocabulary") or {}
    # 别名表的归一名必须存在于词表，否则 Stage4R 会写出词表外的性质名，
    # 下游按 97 项过滤时会被静默丢弃（历史上 impact_strength/hardness 就是这样丢的）。
    unknown = sorted({
        normalized for _, normalized in _PROPERTY_ALIAS_PATTERNS
        if vocabulary and normalized not in vocabulary
    })
    if unknown:
        raise ValueError(f"别名表归一名不在 property_vocabulary 中: {unknown}")
    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(pattern, re.IGNORECASE), normalized)
        for pattern, normalized in _PROPERTY_ALIAS_PATTERNS
    ]
    for normalized_name in vocabulary:
        phrase = str(normalized_name).replace("_", " ")
        # 单个过宽词（如 solvent）容易把条件列误判为性质；这些只依赖上面的
        # 专用别名，或留给 unknown 人工复核。
        if len(phrase) < 6 or phrase in {"solvent", "good solvent", "poor solvent", "non solvent"}:
            continue
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
        patterns.append((re.compile(pattern, re.IGNORECASE), str(normalized_name)))
    return patterns


def _cell_covers_column(cell: Stage0TableCell, column_index: int) -> bool:
    return cell.column_index <= column_index < cell.column_index + cell.column_span


def _cell_covers_row(cell: Stage0TableCell, row_index: int) -> bool:
    return cell.row_index <= row_index < cell.row_index + cell.row_span


def _header_context(
    cells: Sequence[Stage0TableCell],
    target: Stage0TableCell,
    *,
    header_rows: set[int],
) -> tuple[list[str], list[str]]:
    column_headers = [
        cell
        for cell in cells
        if cell.cell_id != target.cell_id
        and cell.row_index in header_rows
        and cell.row_index < target.row_index
        and _cell_covers_column(cell, target.column_index)
        and cell.text.strip()
    ]
    row_headers = [
        cell
        for cell in cells
        if cell.cell_id != target.cell_id
        and cell.column_index < target.column_index
        and _cell_covers_row(cell, target.row_index)
        and cell.text.strip()
        and not _is_value_like_numeric(cell.text)
    ]
    column_headers.sort(key=lambda item: (item.row_index, item.column_index))
    row_headers.sort(key=lambda item: (item.column_index, item.row_index))
    return (
        [item.text for item in column_headers],
        # 取最靠近数据格的 3 层行表头。原来截到 2 层，多级行表头
        # （样品组 / 子组 / 性质）会把带性质名的那一层截掉。
        [item.text for item in row_headers[-3:]],
    )


def _property_match(
    context: str,
    patterns: Sequence[tuple[re.Pattern[str], str]],
) -> str | None:
    for pattern, normalized_name in patterns:
        if pattern.search(context):
            return normalized_name
    return None


def _classify_cell(
    cell: Stage0TableCell,
    *,
    column_headers: Sequence[str],
    row_headers: Sequence[str],
    caption: str,
    property_patterns: Sequence[tuple[re.Pattern[str], str]],
) -> tuple[str, str, str | None]:
    if cell.is_header:
        return ROLE_UNKNOWN, "stage0_header_cell", None
    context_parts = [*column_headers, *row_headers]
    local_context = _normalized_text(" | ".join(context_parts))
    property_name = _property_match(local_context, property_patterns)
    if property_name is not None:
        return ROLE_PROPERTY, f"property_header:{property_name}", property_name
    if _looks_like_identifier(cell.text):
        return ROLE_UNKNOWN, "sample_or_entry_identifier", None
    if _COORDINATE_RE.search(local_context):
        return ROLE_COORDINATE, "coordinate_header", None
    if _CONDITION_RE.search(local_context):
        return ROLE_CONDITION, "condition_header", None
    # caption 只作提示、不升格为候选：caption 覆盖整张表，而一张表通常混着
    # 多种性质。实测 5 张有 caption 命中的表里，4 张的格子并不是 caption 说的
    # 那个性质（如 "…Limiting Oxygen Index…" 表里多数格是 Contact Angle 和
    # Br%）。据此升格会把这些格子贴上错误的性质名。
    # 正确做法是让局部表头能被认出（见 _PROPERTY_ALIAS_PATTERNS 中
    # ηinh、T_{3%} 等），而不是拿 caption 去覆盖列头。
    property_name = _property_match(_normalized_text(caption), property_patterns)
    if property_name is not None:
        return ROLE_UNKNOWN, f"property_caption_hint:{property_name}", property_name
    return ROLE_UNKNOWN, "numeric_role_unresolved", None


def _iter_locators(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        locator = value.get("table_locator")
        if isinstance(locator, Mapping):
            yield locator
        for child in value.values():
            yield from _iter_locators(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_locators(child)


def _locator_cell_ids(
    values: Iterable[Any],
    *,
    valid_cells: Mapping[str, Stage0TableCell],
    numeric_only: bool = False,
) -> set[str]:
    result: set[str] = set()
    for value in values:
        for locator in _iter_locators(value):
            cell_id = locator.get("cell_id")
            if not isinstance(cell_id, str):
                continue
            cell = valid_cells.get(cell_id)
            if cell is None:
                continue
            if numeric_only and not _is_value_like_numeric(cell.text):
                continue
            result.add(cell_id)
    return result


def _stage4_cell_indexes(
    stage4: Mapping[str, Any],
    *,
    valid_cells: Mapping[str, Stage0TableCell],
) -> dict[str, set[str]]:
    property_values: list[Any] = []
    property_values.extend(stage4.get("properties") or [])
    property_values.extend(stage4.get("unresolved_properties") or [])
    series_points: list[Any] = []
    coordinates: list[Any] = []
    for series in stage4.get("property_series") or []:
        if not isinstance(series, Mapping):
            continue
        for point in series.get("points") or []:
            if not isinstance(point, Mapping):
                continue
            series_points.append({"evidence": point.get("evidence") or []})
            coordinates.extend(point.get("coordinates") or [])
    property_values.extend(series_points)

    conditions: list[Any] = list(stage4.get("measurement_conditions") or [])
    for item in [
        *(stage4.get("properties") or []),
        *(stage4.get("unresolved_properties") or []),
        *(stage4.get("property_series") or []),
    ]:
        if isinstance(item, Mapping) and item.get("measurement_context") is not None:
            conditions.append(item.get("measurement_context"))

    value_ids = _locator_cell_ids(
        property_values,
        valid_cells=valid_cells,
        numeric_only=True,
    )
    coordinate_ids = _locator_cell_ids(
        coordinates,
        valid_cells=valid_cells,
        numeric_only=True,
    )
    condition_ids = _locator_cell_ids(
        conditions,
        valid_cells=valid_cells,
        numeric_only=True,
    )
    any_ids = _locator_cell_ids([stage4], valid_cells=valid_cells, numeric_only=True)
    return {
        "property_value": value_ids,
        "coordinate": coordinate_ids,
        "condition": condition_ids,
        "any": any_ids,
    }


def _ratio(covered: int, expected: int) -> float | None:
    if expected <= 0:
        return None
    return round(covered / expected, 6)


def audit_documents(
    stage0: Stage0Document,
    stage4: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    vocabulary_path: Path = DEFAULT_VOCABULARY_PATH,
) -> dict[str, Any]:
    if stage4.get("document_id") != stage0.document_id:
        raise ValueError("Stage 0 与 Stage 4 document_id 不一致")
    property_patterns = _load_property_patterns(vocabulary_path)
    tables = [element for element in stage0.elements if element.type == "table"]
    table_cells = {table.block_id: table_cells_for(table) for table in tables}
    valid_cells = {
        cell.cell_id: cell
        for cells in table_cells.values()
        for cell in cells
    }
    indexes = _stage4_cell_indexes(stage4, valid_cells=valid_cells)

    table_reports: list[dict[str, Any]] = []
    summary_roles: Counter[str] = Counter()
    summary_covered_roles: Counter[str] = Counter()
    for table in tables:
        cells = table_cells[table.block_id]
        header_rows = _infer_header_rows(cells)
        numeric_cells = [cell for cell in cells if _is_value_like_numeric(cell.text)]
        cell_reports: list[dict[str, Any]] = []
        for cell in numeric_cells:
            column_headers, row_headers = _header_context(
                cells, cell, header_rows=header_rows
            )
            role, reason, property_name = _classify_cell(
                cell,
                column_headers=column_headers,
                row_headers=row_headers,
                caption=table.caption or "",
                property_patterns=property_patterns,
            )
            covered_as = [
                name
                for name in ("property_value", "coordinate", "condition", "any")
                if cell.cell_id in indexes[name]
            ]
            summary_roles[role] += 1
            if "property_value" in covered_as:
                summary_covered_roles[role] += 1
            cell_reports.append({
                "cell_id": cell.cell_id,
                "row_index": cell.row_index,
                "column_index": cell.column_index,
                "text": cell.text,
                "is_header": cell.is_header,
                "column_headers": list(column_headers),
                "row_headers": list(row_headers),
                "role": role,
                "role_reason": reason,
                "property_name_normalized": property_name,
                "covered_as": covered_as,
            })

        role_counts = Counter(item["role"] for item in cell_reports)
        property_candidates = [
            item for item in cell_reports if item["role"] == ROLE_PROPERTY
        ]
        missing_property = [
            item for item in property_candidates
            if "property_value" not in item["covered_as"]
        ]
        represented = [item for item in cell_reports if "any" in item["covered_as"]]
        covered_properties = [
            item for item in property_candidates
            if "property_value" in item["covered_as"]
        ]
        property_ratio = _ratio(len(covered_properties), len(property_candidates))
        representation_ratio = _ratio(len(represented), len(cell_reports))
        table_reports.append({
            "table_id": table.block_id,
            "page": table.page,
            "section": table.section,
            "caption": table.caption,
            "numeric_cell_count": len(cell_reports),
            "role_counts": dict(sorted(role_counts.items())),
            "represented_numeric_cell_count": len(represented),
            "property_value_candidate_count": len(property_candidates),
            "covered_property_value_candidate_count": len(covered_properties),
            "missing_property_value_candidate_count": len(missing_property),
            "representation_ratio": representation_ratio,
            "property_value_ratio": property_ratio,
            "numeric_zero_coverage": bool(cell_reports) and not represented,
            "property_zero_coverage": bool(property_candidates) and not covered_properties,
            "needs_recovery": bool(
                property_candidates
                and property_ratio is not None
                and property_ratio < threshold
            ),
            "missing_property_cells": missing_property,
            "cells": cell_reports,
        })

    property_expected = summary_roles[ROLE_PROPERTY]
    property_covered = summary_covered_roles[ROLE_PROPERTY]
    total_numeric = sum(summary_roles.values())
    total_represented = len(indexes["any"] & set(valid_cells))
    return {
        "audit_schema_version": "1.0",
        "audit_version": AUDIT_VERSION,
        "document_id": stage0.document_id,
        "threshold": threshold,
        "summary": {
            "table_count": len(tables),
            "tables_with_numeric_cells": sum(
                report["numeric_cell_count"] > 0 for report in table_reports
            ),
            "numeric_cell_count": total_numeric,
            "represented_numeric_cell_count": total_represented,
            "representation_ratio": _ratio(total_represented, total_numeric),
            "role_counts": dict(sorted(summary_roles.items())),
            "property_value_candidate_count": property_expected,
            "covered_property_value_candidate_count": property_covered,
            "missing_property_value_candidate_count": property_expected - property_covered,
            "property_value_ratio": _ratio(property_covered, property_expected),
            "gap_table_count": sum(report["needs_recovery"] for report in table_reports),
            "numeric_zero_coverage_table_count": sum(
                report["numeric_zero_coverage"] for report in table_reports
            ),
            "property_zero_coverage_table_count": sum(
                report["property_zero_coverage"] for report in table_reports
            ),
        },
        "tables": table_reports,
    }


def audit_files(
    stage0_path: Path,
    stage4_path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    vocabulary_path: Path = DEFAULT_VOCABULARY_PATH,
) -> dict[str, Any]:
    stage0 = Stage0Document.model_validate_json(
        stage0_path.read_text(encoding="utf-8-sig")
    )
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8-sig"))
    if not isinstance(stage4, dict):
        raise ValueError("Stage 4 JSON 顶层必须是对象")
    return audit_documents(
        stage0,
        stage4,
        threshold=threshold,
        vocabulary_path=vocabulary_path,
    )


def audit_batch(
    batch_root: Path,
    report_root: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    vocabulary_path: Path = DEFAULT_VOCABULARY_PATH,
) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    reference_dirs = sorted(
        path for path in batch_root.iterdir()
        if path.is_dir() and path.name.startswith("reference_no_")
    )
    for reference_dir in reference_dirs:
        stage0_path = reference_dir / "stage0_blocks.json"
        stage4_path = reference_dir / "stage4_properties.json"
        if not stage0_path.is_file() or not stage4_path.is_file():
            failures.append({
                "reference": reference_dir.name,
                "error": "missing_stage0_or_stage4",
            })
            continue
        try:
            report = audit_files(
                stage0_path,
                stage4_path,
                threshold=threshold,
                vocabulary_path=vocabulary_path,
            )
        except Exception as exc:  # batch审计要保留其他文献结果
            failures.append({
                "reference": reference_dir.name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        target = report_root / reference_dir.name / "stage4r_table_recall_audit.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports.append(report)

    totals = Counter()
    weighted_property_expected = 0
    weighted_property_covered = 0
    weighted_numeric = 0
    weighted_represented = 0
    for report in reports:
        summary = report["summary"]
        for key in (
            "table_count",
            "tables_with_numeric_cells",
            "gap_table_count",
            "numeric_zero_coverage_table_count",
            "property_zero_coverage_table_count",
        ):
            totals[key] += int(summary[key])
        weighted_property_expected += int(summary["property_value_candidate_count"])
        weighted_property_covered += int(summary["covered_property_value_candidate_count"])
        weighted_numeric += int(summary["numeric_cell_count"])
        weighted_represented += int(summary["represented_numeric_cell_count"])

    batch_summary = {
        "audit_schema_version": "1.0",
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_root": str(batch_root.resolve()),
        "threshold": threshold,
        "document_count": len(reports),
        "failure_count": len(failures),
        "failures": failures,
        "summary": {
            **dict(totals),
            "numeric_cell_count": weighted_numeric,
            "represented_numeric_cell_count": weighted_represented,
            "representation_ratio": _ratio(weighted_represented, weighted_numeric),
            "property_value_candidate_count": weighted_property_expected,
            "covered_property_value_candidate_count": weighted_property_covered,
            "missing_property_value_candidate_count": (
                weighted_property_expected - weighted_property_covered
            ),
            "property_value_ratio": _ratio(
                weighted_property_covered,
                weighted_property_expected,
            ),
        },
        "documents": [
            {"document_id": report["document_id"], **report["summary"]}
            for report in reports
        ],
    }
    (report_root / "batch_table_recall_audit.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return batch_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线审计 Stage 4 表格单元格召回")
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=DEFAULT_VOCABULARY_PATH,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold 必须位于0到1之间")
    summary = audit_batch(
        args.batch_root,
        args.report_root,
        threshold=args.threshold,
        vocabulary_path=args.vocabulary,
    )
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    if summary["failures"]:
        print(json.dumps(summary["failures"], ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
