"""把 Stage 0 表格转换为稳定网格，并解析单元格 locator。"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any, Mapping

from schema.polymer_schema import Stage0Element, Stage0TableCell


_SPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _positive_span(attrs: dict[str, str | None], name: str) -> int:
    try:
        return max(1, int(attrs.get(name) or 1))
    except (TypeError, ValueError):
        return 1


class _TableHTMLParser(HTMLParser):
    def __init__(self, block_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.block_id = block_id
        self.cells: list[Stage0TableCell] = []
        self._row_index = -1
        self._column_cursor = 0
        self._occupied: set[tuple[int, int]] = set()
        self._cell: dict[str, Any] | None = None
        self._cell_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        if tag == "tr":
            self._row_index += 1
            self._column_cursor = 0
            return
        if tag not in {"td", "th"}:
            return
        if self._row_index < 0:
            self._row_index = 0
        while (self._row_index, self._column_cursor) in self._occupied:
            self._column_cursor += 1
        attributes = dict(attrs)
        row_span = _positive_span(attributes, "rowspan")
        column_span = _positive_span(attributes, "colspan")
        self._cell = {
            "row_index": self._row_index,
            "column_index": self._column_cursor,
            "row_span": row_span,
            "column_span": column_span,
            "is_header": tag == "th",
        }
        self._cell_text = []
        for row in range(self._row_index, self._row_index + row_span):
            for column in range(
                self._column_cursor,
                self._column_cursor + column_span,
            ):
                self._occupied.add((row, column))
        self._column_cursor += column_span

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() not in {"td", "th"} or self._cell is None:
            return
        row_index = self._cell["row_index"]
        column_index = self._cell["column_index"]
        value = _SPACE_RE.sub(" ", "".join(self._cell_text)).strip()
        self.cells.append(Stage0TableCell(
            cell_id=_cell_id(self.block_id, row_index, column_index),
            text=value,
            **self._cell,
        ))
        self._cell = None
        self._cell_text = []


def _cell_id(block_id: str, row_index: int, column_index: int) -> str:
    return f"{block_id}:r{row_index:04d}:c{column_index:04d}"


def _markdown_cells(table_body: str, block_id: str) -> list[Stage0TableCell]:
    rows: list[list[str]] = []
    for raw_line in table_body.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        values = [value.strip() for value in line.strip("|").split("|")]
        if values and all(
            _MARKDOWN_SEPARATOR_RE.fullmatch(value.replace(" ", ""))
            for value in values
        ):
            continue
        rows.append(values)
    return [
        Stage0TableCell(
            cell_id=_cell_id(block_id, row_index, column_index),
            row_index=row_index,
            column_index=column_index,
            text=value,
            is_header=row_index == 0,
        )
        for row_index, row in enumerate(rows)
        for column_index, value in enumerate(row)
    ]


def parse_table_cells(table_body: str, block_id: str) -> list[Stage0TableCell]:
    """解析 HTML 或 Markdown 表格；只为源单元格生成一个稳定锚点。"""

    if re.search(r"<(?:table|tr|td|th)\b", table_body, re.IGNORECASE):
        parser = _TableHTMLParser(block_id)
        parser.feed(table_body)
        parser.close()
        if parser.cells:
            return parser.cells
    return _markdown_cells(table_body, block_id)


def table_cells_for(block: Stage0Element) -> list[Stage0TableCell]:
    if block.type != "table" or block.table_body is None:
        return []
    if block.table_cells is not None:
        return block.table_cells
    return parse_table_cells(block.table_body, block.block_id)


def _normalize(value: str) -> str:
    value = html.unescape(_TAG_RE.sub(" ", value))
    value = value.replace("$", "")
    value = _SPACE_RE.sub(" ", value).strip().casefold()
    return value


def _matches(cell_text: str, locator_text: str) -> bool:
    left = _normalize(cell_text)
    right = _normalize(locator_text)
    if not left or not right:
        return False
    if left == right:
        return True
    if _NUMERIC_RE.fullmatch(left) or _NUMERIC_RE.fullmatch(right):
        return False
    if min(
        sum(char.isalnum() for char in left),
        sum(char.isalnum() for char in right),
    ) < 3:
        return False
    return right in left or left in right


def _covers_row(cell: Stage0TableCell, row_index: int) -> bool:
    return cell.row_index <= row_index < cell.row_index + cell.row_span


def _covers_column(cell: Stage0TableCell, column_index: int) -> bool:
    return (
        cell.column_index
        <= column_index
        < cell.column_index + cell.column_span
    )


def resolve_table_locator(
    block: Stage0Element,
    locator: Mapping[str, Any],
) -> dict[str, Any] | None:
    """用行标签、列标签和值把 locator 解析到唯一稳定单元格。"""

    cells = table_cells_for(block)
    if not cells:
        return None
    stable_fields = ("cell_id", "row_index", "column_index")
    if all(locator.get(field) is not None for field in stable_fields):
        expected = next(
            (
                cell
                for cell in cells
                if cell.cell_id == locator["cell_id"]
                and cell.row_index == locator["row_index"]
                and cell.column_index == locator["column_index"]
            ),
            None,
        )
        cell_value = locator.get("cell_value")
        value_matches = (
            expected is not None
            and (
                not expected.text.strip()
                if cell_value is None
                else _matches(expected.text, str(cell_value))
            )
        )
        if expected is None or not value_matches:
            return None
        return {
            **dict(locator),
            "cell_id": expected.cell_id,
            "row_index": expected.row_index,
            "column_index": expected.column_index,
        }

    row_labels = [
        cell
        for cell in cells
        if _matches(cell.text, str(locator.get("row_label") or ""))
    ]
    column_labels = [
        cell
        for cell in cells
        if _matches(cell.text, str(locator.get("column_label") or ""))
    ]
    if locator.get("cell_value") is None:
        intersections = [
            cell
            for cell in cells
            if not cell.text.strip()
            and any(
                _covers_row(label, cell.row_index)
                for label in row_labels
            )
            and any(
                _covers_column(label, cell.column_index)
                for label in column_labels
            )
        ]
        if len(intersections) != 1:
            return None
        cell = intersections[0]
        return {
            **dict(locator),
            "cell_id": cell.cell_id,
            "row_index": cell.row_index,
            "column_index": cell.column_index,
        }
    values = [
        cell
        for cell in cells
        if _matches(cell.text, str(locator.get("cell_value") or ""))
    ]
    if not values:
        return None
    scored: list[tuple[int, Stage0TableCell]] = []
    for value in values:
        row_score = 2 if any(
            _covers_row(label, value.row_index) for label in row_labels
        ) else 0
        column_score = 2 if any(
            _covers_column(label, value.column_index)
            for label in column_labels
        ) else 0
        scored.append((row_score + column_score, value))
    best_score = max(score for score, _ in scored)
    best = [cell for score, cell in scored if score == best_score]
    if len(best) != 1:
        return None
    cell = best[0]
    return {
        **dict(locator),
        "cell_id": cell.cell_id,
        "row_index": cell.row_index,
        "column_index": cell.column_index,
    }
