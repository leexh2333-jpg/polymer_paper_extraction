"""Preview-only Stage 4R：从明确缺失的表格格子确定性补入性质值。

默认只写 stage4_properties.recovery_preview.json，不覆盖 Stage 4。使用 --apply
时会先备份原文件，再写回 stage4_properties.json。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from schema.polymer_schema import Stage0Document, Stage4Document
from stages.table_grid import table_cells_for
from stages.table_recall_audit import _infer_header_rows, audit_documents

STAGE_ID = "stage4r_table_recovery"
IMPLEMENTATION_VERSION = "0.1.0"
_ID_RE = re.compile(r"^uprop(\d+)$")


def _norm(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").casefold())


def _cell_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        locator = value.get("table_locator")
        if isinstance(locator, Mapping) and isinstance(locator.get("cell_id"), str):
            result.add(locator["cell_id"])
        for child in value.values():
            result.update(_cell_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_cell_ids(child))
    return result


def next_unresolved_number(stage4: Mapping[str, Any]) -> int:
    numbers = []
    for item in stage4.get("unresolved_properties") or []:
        match = _ID_RE.fullmatch(str(item.get("unresolved_id") or ""))
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _sample_entity_map(stage3: Mapping[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    samples = [item for item in stage3.get("samples") or [] if isinstance(item, Mapping)]
    mapping = {
        str(item["sample_id"]): str(item["refers_to_entity"])
        for item in samples
        if item.get("sample_id") and item.get("refers_to_entity")
    }
    return mapping, [dict(item) for item in samples]


def _entity_aliases(stage2: Mapping[str, Any]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for item in stage2.get("polymer_entities") or []:
        if not isinstance(item, Mapping) or not item.get("entity_id"):
            continue
        entity_id = str(item["entity_id"])
        for key in ("canonical_name", "normalized_name", "display_name"):
            if item.get(key):
                aliases[entity_id].add(str(item[key]))
        for value in item.get("source_names") or []:
            if value:
                aliases[entity_id].add(str(value))
    return aliases


def _table_entities(stage4: Mapping[str, Any], sample_entities: Mapping[str, str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in [
        *(stage4.get("properties") or []),
        *(stage4.get("unresolved_properties") or []),
        *(stage4.get("property_series") or []),
    ]:
        if not isinstance(item, Mapping):
            continue
        entity = item.get("entity_id") or sample_entities.get(str(item.get("sample_id") or ""))
        if not entity:
            continue
        for cell_id in _cell_ids(item):
            result[cell_id.split(":", 1)[0]].add(str(entity))
    return result



def _row_label_candidates(
    cells: Sequence[Any],
    row_index: int,
    column_index: int,
) -> list[str]:
    """取该行位于目标列左侧的所有单元格文本，作为样品标签候选。

    审计器的 row_headers 会把"看起来像数值"的格子剔掉，而不少论文的样品编码
    恰好是纯数字加连字符（0-2、0-2-0-6、0-4-0-10），于是真正的样品标签被当成
    数值滤掉，infer_entity_id 拿到空的 row_headers 只能报 entity_ambiguous。
    这里绕过该过滤，直接按网格取左侧标签列。

    另外补一路**向上填充**：这类表把同一个聚合物的多条工艺写成续行，编号只印在
    该组第一行，续行的编号列是空白：

        2a | 0-2-0-9 | 2 |        | 9 | HTS |     | 265
        b  |         |   | DiEt-2 | 9 | HTS | 295 |
        c  |         | 2 |        | 9 | HTS | 320 |     <- 只剩 ['c', 'HTS']

    续行拿不到 `0-2-0-9`，实体必然推不出来。空白在这类表里的含义就是"同上"，
    所以逐列向上找最近的非空格子补进候选。补进来的只走 infer_entity_id 的
    **精确相等**分支，且要求全局唯一命中；补错的词配不上任何实体名，最多让命中
    变成两个而退回 entity_ambiguous（照旧跳过），不会造成错挂。
    """
    covering = [
        cell
        for cell in cells
        if cell.row_index <= row_index < cell.row_index + cell.row_span
        and cell.column_index < column_index
    ]
    out = [cell.text.strip() for cell in sorted(covering, key=lambda item: item.column_index)
           if cell.text.strip()]

    filled = {cell.column_index for cell in covering if cell.text.strip()}
    for column in range(column_index):
        if column in filled:
            continue
        above = [
            cell
            for cell in cells
            if cell.column_index == column
            and cell.row_index + cell.row_span <= row_index
            and cell.text.strip()
        ]
        if above:
            out.append(max(above, key=lambda item: item.row_index).text.strip())
    return out


def infer_entity_id(
    *,
    row_headers: Sequence[str],
    row_index: int | None = None,
    table_id: str,
    valid_entity_ids: set[str],
    samples: Sequence[Mapping[str, Any]],
    entity_aliases: Mapping[str, set[str]] | None = None,
    table_entities: Mapping[str, set[str]],
    row_label_candidates: Sequence[str] | None = None,
) -> tuple[str | None, str]:
    row_parts = [_norm(value) for value in row_headers if _norm(value)]
    aliases_by_entity: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        entity = str(sample.get("refers_to_entity") or "")
        if entity not in valid_entity_ids:
            continue
        for label in (sample.get("sample_label_raw"), sample.get("polymer_name")):
            if label_norm := _norm(label):
                aliases_by_entity[entity].add(label_norm)
    for entity, aliases in (entity_aliases or {}).items():
        if entity not in valid_entity_ids:
            continue
        aliases_by_entity[entity].update(
            alias_norm for alias in aliases if (alias_norm := _norm(alias))
        )

    exact_matches = {
        entity
        for entity, aliases in aliases_by_entity.items()
        if any(row_part == alias for row_part in row_parts for alias in aliases)
    }
    if len(exact_matches) == 1:
        return next(iter(exact_matches)), "row_label_exact_entity_alias"

    # 仅保留最长的包含匹配，避免 0-2 与 0-2-0-I 同时命中造成假歧义。
    best_length = 0
    best_matches: set[str] = set()
    for entity, aliases in aliases_by_entity.items():
        for row_part in row_parts:
            for alias in aliases:
                if len(alias) < 3 or not (alias in row_part or row_part in alias):
                    continue
                score = min(len(alias), len(row_part))
                if score > best_length:
                    best_length = score
                    best_matches = {entity}
                elif score == best_length:
                    best_matches.add(entity)
    row_text = _norm(" ".join(row_headers))
    mixture_markers = ("blend", "mixture", "composite", "共混", "混合", "复合")
    if len(best_matches) == 1 and not any(marker in row_text for marker in mixture_markers):
        return next(iter(best_matches)), "row_label_longest_entity_alias"

    sample_entity_ids = {
        str(sample.get("refers_to_entity"))
        for sample in samples
        if str(sample.get("refers_to_entity") or "") in valid_entity_ids
    }
    if not row_parts and len(sample_entity_ids) == 1:
        return next(iter(sample_entity_ids)), "document_samples_single_entity"

    existing = set(table_entities.get(table_id) or set()) & valid_entity_ids
    if not row_parts and len(existing) == 1:
        return next(iter(existing)), "table_existing_entity"
    if len(valid_entity_ids) == 1:
        return next(iter(valid_entity_ids)), "document_single_entity"

    # 兜底：审计器把纯数字样品编码（0-2-0-6）当数值滤掉了，用未过滤的
    # 左侧标签列重试一次严格匹配。只认精确相等，不做包含匹配 —— 包含匹配在
    # 这类编码上极易假阳（0-2 是 0-2-0-6 的子串）。
    extra_parts = [_norm(value) for value in (row_label_candidates or []) if _norm(value)]
    extra_parts = [part for part in extra_parts if part not in set(row_parts)]
    if extra_parts:
        extra_exact = {
            entity
            for entity, aliases in aliases_by_entity.items()
            if any(part == alias for part in extra_parts for alias in aliases)
        }
        if len(extra_exact) == 1:
            return next(iter(extra_exact)), "row_cell_exact_entity_alias"

    return None, "entity_ambiguous" if valid_entity_ids else "entity_not_found"


def _row_sentence(cells: Sequence[Any], row_index: int) -> str:
    row = sorted(
        (cell for cell in cells if cell.row_index == row_index and cell.text.strip()),
        key=lambda cell: cell.column_index,
    )
    return " | ".join(cell.text.strip() for cell in row)


# 表头里出现的单位写法 -> 规范单位。只认白名单里的写法：表头层级里混着数据值
# （'290'、'-25'、'1.3'）和样品名（'1AQA-PPDI'），靠"括号里的东西就是单位"
# 这类结构规则会把它们当成单位。
_UNIT_ALIASES: dict[str, str] = {
    # 温度
    "°c": "°C", "oc": "°C", "℃": "°C", "\\circ c": "°C", "deg c": "°C", "k": "K",
    # 力学
    "mpa": "MPa", "gpa": "GPa", "kpa": "kPa", "pa": "Pa",
    # 冲击
    "j/m": "J/m", "kj/m": "kJ/m", "j/m2": "J/m²", "kj/m2": "kJ/m²",
    "j/m^2": "J/m²", "kj/m^2": "kJ/m²",
    # 电学
    "s/cm": "S/cm", "s/m": "S/m",
    "ohm*cm": "Ω·cm", "ohm cm": "Ω·cm", "ω·cm": "Ω·cm", "ωcm": "Ω·cm",
    "ohm*m": "Ω·m", "ω·m": "Ω·m",
    "1/(ohm*cm)": "S/cm", "s cm-1": "S/cm", "scm-1": "S/cm",
    # 黏度
    "dl/g": "dL/g", "dlg-1": "dL/g", "dl g-1": "dL/g", "dl/g-1": "dL/g",
    "ml/g": "mL/g",
    # 密度 / 表面
    "g/cm3": "g/cm³", "g/cm^3": "g/cm³", "gcm-3": "g/cm³",
    "mj/m2": "mJ/m²", "mj/m^2": "mJ/m²", "mn/m": "mN/m", "dyn/cm": "dyn/cm",
    # 热
    "j/g": "J/g", "kj/mol": "kJ/mol", "kcal/mol": "kcal/mol", "kcal/g": "kcal/g",
    "kj/g": "kJ/g", "cal/g": "cal/g", "j/mol": "J/mol",
    "j/(g k)": "J/(g·K)",
    # 分子量 / 其他
    "g/mol": "g/mol", "gmol-1": "g/mol", "kg/mol": "kg/mol",
    "%": "%", "wt%": "wt%", "vol%": "vol%", "mol%": "mol%",
    "nm": "nm", "μm": "μm", "um": "μm", "mm": "mm", "cm": "cm", "å": "Å",
    "hz": "Hz", "mhz": "MHz", "khz": "kHz",
}

# LaTeX 包装 —— 必须先整串剥掉再切片段，否则 `$(^{\circ}\text{C})$` 会被
# 括号规则从 `\text{` 中间截断。
_UNIT_STRIP_RE = re.compile(r"\\(?:text|mathrm|mathit|rm|it|bf)\s*\{([^{}]*)\}")
_BRACKET_GROUP_RE = re.compile(r"[（(\[]([^）)\]]*)[）)\]]")
_TRAILING_NUMBER_RE = re.compile(r"\d\s*$")
# `T_g°C` 里的 °C 是单位；`Modulus at 160 °C` 里的 °C 是测试条件。差别就是
# ° 前面是不是数字。
_DEGREE_TAIL_RE = re.compile(r"(?<![\d])\s*(°\s*[a-z]?\.?)\s*$")
# `kJ mol^{-1}`、`S cm⁻¹`、`dL g-1` —— 倒数指数就是分母。不折成 `/` 的话
# 白名单里的 `kj/mol`、`s/cm`、`dl/g` 全都对不上。
_INVERSE_EXP_RE = re.compile(r"\s*([a-zμω]+)\s*(?:\^\s*)?(?:-\s*1|⁻¹)(?![\d.])",
                             re.IGNORECASE)


def _normalize_unit_text(text: str) -> str:
    """整串归一：剥 LaTeX、统一度数与欧姆写法、折算倒数指数，转小写。"""
    value = str(text or "")
    for _ in range(3):                       # \text{\circ C} 可能嵌套
        value = _UNIT_STRIP_RE.sub(r"\1", value)
    value = value.replace("\\%", "%").replace("\\,", " ").replace("\\;", " ")
    value = value.replace("^\\circ", "°").replace("\\circ", "°").replace("^o", "°")
    value = re.sub(r"°\s*([a-z])", r"°\1", value, flags=re.IGNORECASE)
    value = value.replace("\\Omega", "ohm").replace("Ω", "ω")
    value = value.replace("·", "*").replace("⋅", "*")
    value = value.replace("$", "").replace("{", "").replace("}", "")
    # 只在前面确实还有一个单位词时才折算，避免把孤零零的 `^-1` 当分母。
    if _INVERSE_EXP_RE.search(value) and re.search(r"[a-zμω]", value):
        value = _INVERSE_EXP_RE.sub(r"/\1", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _clean_unit_token(text: str) -> str:
    """把候选片段两端的括号与标点去掉。"""
    return str(text or "").strip().strip("（）()[]").strip().strip(",.;:·*").strip()


def _unit_candidates(header: str) -> list[str]:
    """从一级表头里列出所有可能是单位的片段，按可信度从高到低。"""
    norm = _normalize_unit_text(header)
    if not norm:
        return []
    # 括号内多为单位，但也可能是测试条件（`(160 °C)`）；两种都留作候选，
    # 靠白名单区分。同时准备一份去掉括号的正文，供逗号/斜杠切分用。
    out: list[str] = list(_BRACKET_GROUP_RE.findall(norm))
    bare = re.sub(r"[（(\[][^）)\]]*[）)\]]", " ", norm).strip()
    out.append(norm)
    out.append(bare)
    out.extend(re.split(r"[,，;；]", bare))
    for src in (bare, norm, *list(out)):
        if not src:
            continue
        if "/" in src:                       # `T_g / °C`、`η_inh^c/dL/g`
            out.append(src.split("/", 1)[1])
        if "^" in src:                       # `T_g^oC`、`(^°C)^b`
            out.append(src.rsplit("^", 1)[1])
            out.append(src.split("^", 1)[1])
        # 末尾的度数符号（`T_g°C`、`M.P. °C.`）。前面是数字时是测试条件不是单位。
        if match := _DEGREE_TAIL_RE.search(src):
            out.append(match.group(1))
        tokens = src.split()
        if len(tokens) > 1 and not _TRAILING_NUMBER_RE.search(tokens[-2]):
            out.append(tokens[-1])
    return out


def infer_unit_from_headers(column_headers: Sequence[str]) -> str | None:
    """从列头里认单位。认不出就返回 None —— 宁可留空，不要猜错。

    单位在表头里的位置没有规律：括号内 `Tensile strength (MPa)`、斜杠后
    `$T_g / ^\\circ C$`、逗号后 `PMT, °C.`、上标 `$T_g^oC$`，也可能自己
    单占一级 `['$T_g$', '°C']`。所以扫描所有层级的所有候选片段，但**只接受
    _UNIT_ALIASES 里的写法** —— 表头层级里混着数据值（'290'、'-25'）和样品名
    （'1AQA-PPDI'），"括号里的就是单位"这类纯结构规则会把它们写成单位。
    """
    for header in reversed([str(item) for item in column_headers if str(item).strip()]):
        for candidate in _unit_candidates(header):
            token = _clean_unit_token(candidate)
            if token and (unit := _UNIT_ALIASES.get(token)):
                return unit
    return None


def build_unresolved_property(
    *,
    unresolved_id: str,
    entity_id: str,
    cell_report: Mapping[str, Any],
    table: Any,
    cells: Sequence[Any],
) -> dict[str, Any]:
    column_headers = [str(item) for item in cell_report.get("column_headers") or [] if str(item).strip()]
    row_headers = [str(item) for item in cell_report.get("row_headers") or [] if str(item).strip()]
    # Preview 首版使用审计器已经确认的规范性质名，避免多层表头中的前序
    # 数据值或占位符被误写成 property_name_raw。
    property_name = str(cell_report["property_name_normalized"]).strip()
    row_index = int(cell_report["row_index"])
    column_index = int(cell_report["column_index"])
    value_raw = str(cell_report["text"]).strip()
    source_sentence = _row_sentence(cells, row_index) or value_raw
    bbox = list(table.bbox) if table.bbox is not None else None
    item = {
        "unresolved_id": unresolved_id,
        "entity_id": entity_id,
        "property_name_raw": property_name,
        "value_raw": value_raw,
        "reason": "sample_ambiguous",
        "evidence": [{
            "block_id": table.block_id,
            "page": table.page,
            "bbox": bbox,
            "source_type": "table",
            "source_sentence": source_sentence,
            "table_locator": {
                "table_id": table.block_id,
                "cell_id": cell_report["cell_id"],
                "row_index": row_index,
                "column_index": column_index,
                "row_label": " / ".join(row_headers) or None,
                "column_label": " / ".join(column_headers) or None,
                "cell_value": value_raw,
            },
        }],
    }
    # 单位就印在列头里，此前被整个丢掉，导致恢复层的记录全部无单位、
    # 无法与 PolyInfo 的统一单位对齐。认不出时不写该字段。
    if unit := infer_unit_from_headers(column_headers):
        item["unit_raw"] = unit
    return item


_HEADER_TEXT_RE = re.compile(r"[A-Za-z一-鿿]{2,}")
_HEADER_NUMBER_RE = re.compile(r"^[\s~<>=±(\[]*[-+]?\d")


def misaligned_header_columns(
    cells: Sequence[Any],
    header_rows: set[int],
) -> set[int]:
    """找出列头与数据错位的列，返回不可信的列号集合。

    这类表的 HTML 会漏写表头的 colspan。0073324 Table II 的
    `Method of polymerization` 实际占 HTS / LTS 两个子列，表头却只给了 1 列，
    而数据行仍按 2 列写，于是从该列往右整条表头左移一格：

        表头  ... | Method of polymerizationb | PMT, °C. | η_inh (DMSO) | Comments
        数据  ... | HTS |    (空)    |     342     |    1.43     |
                              ^col7        ^col8         ^col9
        -> col7 挂着 "PMT" 却装的是 HTS/LTS 文字，col8 的 342（真的 PMT）
           被贴上 η_inh 的标签，写进恢复层就是错的性质名。

    判据必须**只认这个错位**，不能顺手把正常表也判进去。样品名列（`PDLA`、
    `COC`）同样是"整列文字"，但它的表头就写着 `Sample`，本来就不该有数值 ——
    只看"整列文字"会把几乎每张表的首列都误判。所以要求两个条件同时成立：

      1) 该列**自己独占**的表头（不跨列、因此不会串到邻列去）声明了一个数值量，
         如 `PMT, °C.`、`T_g (°C)`；而该列数据全是文字、一个数都没有。
      2) 它右侧存在**表头不像数值量、数据却全是数值**的列 —— 那就是被挤过去的
         真数据列。正常表不会出现这种"数值跑到非数值表头下面"的组合。

    两条都满足才认定该表表头整体左移，标记该列及其右侧所有列。
    """
    own_header: dict[int, list[str]] = defaultdict(list)
    for cell in cells:
        if cell.row_index in header_rows and (getattr(cell, "column_span", 1) or 1) == 1:
            if text := cell.text.strip():
                own_header[cell.column_index].append(text)

    by_column: dict[int, list[str]] = defaultdict(list)
    for cell in cells:
        if cell.row_index in header_rows:
            continue
        if text := cell.text.strip():
            by_column[cell.column_index].append(text)

    def profile(column: int) -> tuple[int, int]:
        values = by_column.get(column) or []
        numeric = sum(1 for text in values if _HEADER_NUMBER_RE.match(text))
        wordy = sum(
            1 for text in values
            if _HEADER_TEXT_RE.search(text) and not _HEADER_NUMBER_RE.match(text)
        )
        return numeric, wordy

    for column in sorted(by_column):
        numeric, wordy = profile(column)
        if numeric or not wordy:
            continue
        header_text = " ".join(own_header.get(column) or [])
        if not header_text or not _NUMERIC_HEADER_HINT_RE.search(header_text):
            continue
        # 条件 2：右侧要有"表头不像数值量、数据却全是数值"的列 —— 那正是被
        # 挤过去的真数据列（`342` 落在 `η_inh (DMSO)` 名下）。缺了这条就只是
        # 一个空的数值列（论文没测），不是错位，不该整片放弃。
        shifted = any(
            profile(later)[0] and not profile(later)[1]
            and not _NUMERIC_HEADER_HINT_RE.search(" ".join(own_header.get(later) or []))
            for later in by_column
            if later > column
        )
        if shifted:
            return set(range(column, max(by_column) + 1))
    return set()


# 表头里出现这些就说明该列本该是数值：温度/模量/强度等带单位或符号的写法。
# 注意别写成会嵌进普通词里的片段 —— `M\.?P\.?` 会match "Sa(mp)le"、"co(mp)ound"，
# 把每张表的样品名列都判成错位。所以带字母的项一律加词边界并要求真的有点号。
_NUMERIC_HEADER_HINT_RE = re.compile(
    r"°\s*[cCkK]|\bK\b|MPa|GPa|kPa|S\s*/\s*cm|Ω|ohm|J\s*/\s*[gm]|dL\s*/\s*g|"
    r"\bPMT\b|\bM\.P\.|\bT_?[gmdc]\b|\bη",
    re.IGNORECASE,
)


def recover_document(
    stage0: Stage0Document,
    stage2: Mapping[str, Any],
    stage3: Mapping[str, Any],
    stage4: Mapping[str, Any],
    *,
    threshold: float = 0.8,
) -> tuple[Stage4Document, dict[str, Any]]:
    original = Stage4Document.model_validate(stage4)
    audit = audit_documents(stage0, stage4, threshold=threshold)
    valid_entities = {
        str(item.get("entity_id"))
        for item in stage2.get("polymer_entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }
    sample_entities, samples = _sample_entity_map(stage3)
    entity_aliases = _entity_aliases(stage2)
    table_entities = _table_entities(stage4, sample_entities)
    tables = {item.block_id: item for item in stage0.elements if item.type == "table"}
    existing_cells = _cell_ids(stage4)
    next_number = next_unresolved_number(stage4)
    additions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for table_report in audit.get("tables") or []:
        table_id = str(table_report["table_id"])
        table = tables.get(table_id)
        if table is None:
            continue
        cells = table_cells_for(table)
        bad_columns = misaligned_header_columns(cells, _infer_header_rows(cells))
        for cell in table_report.get("missing_property_cells") or []:
            cell_id = str(cell["cell_id"])
            if cell_id in existing_cells:
                continue
            if int(cell["column_index"]) in bad_columns:
                skipped.append({
                    "cell_id": cell_id,
                    "table_id": table_id,
                    "property_name_normalized": cell.get("property_name_normalized"),
                    "value_raw": cell.get("text"),
                    "reason": "misaligned_table_header",
                })
                continue
            if not str(cell.get("role_reason") or "").startswith("property_header:"):
                skipped.append({
                    "cell_id": cell_id,
                    "table_id": table_id,
                    "property_name_normalized": cell.get("property_name_normalized"),
                    "value_raw": cell.get("text"),
                    "reason": "unsafe_non_header_property_hint",
                })
                continue
            entity_id, basis = infer_entity_id(
                row_headers=cell.get("row_headers") or [],
                row_index=cell.get("row_index"),
                table_id=table_id,
                valid_entity_ids=valid_entities,
                samples=samples,
                entity_aliases=entity_aliases,
                table_entities=table_entities,
                row_label_candidates=_row_label_candidates(
                    cells,
                    int(cell["row_index"]),
                    int(cell["column_index"]),
                ),
            )
            if entity_id is None:
                skipped.append({
                    "cell_id": cell_id,
                    "table_id": table_id,
                    "property_name_normalized": cell.get("property_name_normalized"),
                    "value_raw": cell.get("text"),
                    "reason": basis,
                })
                continue
            item = build_unresolved_property(
                unresolved_id=f"uprop{next_number:03d}",
                entity_id=entity_id,
                cell_report=cell,
                table=table,
                cells=cells,
            )
            next_number += 1
            additions.append(item)
            item["_entity_resolution_basis"] = basis
            existing_cells.add(cell_id)

    payload = original.model_dump(mode="json")
    payload["unresolved_properties"].extend(
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in additions
    )
    if additions:
        payload["warnings"].append({
            "code": "stage4r_preview_table_recovery",
            "message": f"Stage 4R Preview 确定性补入 {len(additions)} 个表格性质格；样品归属保留 unresolved。",
            "recovered_cell_count": len(additions),
            "skipped_ambiguous_count": len(skipped),
        })
    merged = Stage4Document.model_validate(payload)
    report = {
        "stage": STAGE_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "document_id": stage0.document_id,
        "audit_summary": audit.get("summary"),
        "recovered_count": len(additions),
        "skipped_ambiguous_count": len(skipped),
        "recovered": [
            {
                "unresolved_id": item["unresolved_id"],
                "entity_id": item["entity_id"],
                "property_name_raw": item["property_name_raw"],
                "value_raw": item["value_raw"],
                "cell_id": item["evidence"][0]["table_locator"]["cell_id"],
                "entity_resolution_basis": item["_entity_resolution_basis"],
            }
            for item in additions
        ],
        "skipped_ambiguous": skipped,
    }
    return merged, report


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _same_file_content(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()


def _prepare_stage4_input(
    *,
    stage4_path: Path,
    preview_path: Path,
    report_path: Path,
    backup_path: Path,
    apply: bool,
    force: bool,
    in_place: bool,
) -> tuple[Path, bool]:
    already_applied = apply and in_place and _same_file_content(stage4_path, preview_path)
    if not force and report_path.is_file() and preview_path.is_file():
        if not apply or already_applied:
            return stage4_path, True
    if not apply or not in_place:
        return stage4_path, False
    if already_applied and backup_path.is_file():
        return backup_path, False
    shutil.copy2(stage4_path, backup_path)
    return stage4_path, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ref-no", required=True)
    parser.add_argument("--config", type=Path, help="兼容批处理器；Stage 4R 不读取模型配置")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = (args.output_root or input_root).expanduser().resolve()
    ref_dir = args.ref_no if args.ref_no.startswith("reference_no_") else f"reference_no_{args.ref_no}"
    source = input_root / ref_dir
    target = output_root / ref_dir
    target.mkdir(parents=True, exist_ok=True)
    stage4_path = source / "stage4_properties.json"
    preview_path = target / "stage4_properties.recovery_preview.json"
    report_path = target / "stage4r_recovery.json"
    backup_path = target / "stage4_properties.pre_recovery.json"
    stage4_input_path, cached = _prepare_stage4_input(
        stage4_path=stage4_path,
        preview_path=preview_path,
        report_path=report_path,
        backup_path=backup_path,
        apply=bool(args.apply),
        force=bool(args.force),
        in_place=source.resolve() == target.resolve(),
    )
    if cached:
        print(f"[cached] {ref_dir}: Stage 4R Preview 已存在")
        return 0
    stage0 = Stage0Document.model_validate_json((source / "stage0_blocks.json").read_text(encoding="utf-8"))
    stage2 = _load_json(source / "stage2_entities.json")
    stage3 = _load_json(source / "stage3_process.json")
    stage4 = _load_json(stage4_input_path)
    merged, report = recover_document(stage0, stage2, stage3, stage4, threshold=args.threshold)
    _write_json(report_path, report)
    _write_json(preview_path, merged.model_dump(mode="json"))
    if args.apply:
        destination = target / "stage4_properties.json"
        _write_json(destination, merged.model_dump(mode="json"))
    print(json.dumps({
        "document_id": merged.document_id,
        "recovered_count": report["recovered_count"],
        "skipped_ambiguous_count": report["skipped_ambiguous_count"],
        "preview_path": str(preview_path),
        "applied": bool(args.apply),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
