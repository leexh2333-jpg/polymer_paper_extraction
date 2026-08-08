from __future__ import annotations

from schema.polymer_schema import Stage0Element
from stages.table_grid import parse_table_cells, resolve_table_locator


def _table(block_id: str, body: str) -> Stage0Element:
    return Stage0Element(
        block_id=block_id,
        type="table",
        page=0,
        source_block_index=0,
        table_body=body,
        table_cells=parse_table_cells(body, block_id),
    )


def test_resolve_locator_uses_row_and_spanning_column_headers() -> None:
    block = _table(
        "T_6_81",
        """
        <table>
          <tr><td rowspan="2"></td><td rowspan="2">$[\\eta]$</td>
            <td rowspan="2">$[Q]$</td><td colspan="2">Intercept</td></tr>
          <tr><td>$[\\eta_{\\text{max}}]$</td><td>$[Q_{max}]$</td></tr>
          <tr><td>$\\delta_p$</td><td>8.55</td><td>8.60</td>
            <td>8.60</td><td>8.65</td></tr>
        </table>
        """,
    )

    resolved = resolve_table_locator(block, {
        "table_id": "T_6_81",
        "row_label": r"$\delta_p$",
        "column_label": "$[Q]$",
        "cell_value": "8.60",
    })

    assert resolved is not None
    assert resolved["cell_id"] == "T_6_81:r0002:c0002"
    assert resolved["row_index"] == 2
    assert resolved["column_index"] == 2


def test_markdown_table_has_stable_cells() -> None:
    cells = parse_table_cells(
        "| Sample | Value |\n| --- | --- |\n| S1 | 3.2 |",
        "T_0_1",
    )

    assert [(cell.text, cell.row_index, cell.column_index) for cell in cells] == [
        ("Sample", 0, 0),
        ("Value", 0, 1),
        ("S1", 1, 0),
        ("3.2", 1, 1),
    ]
    assert cells[-1].cell_id == "T_0_1:r0001:c0001"


def test_invalid_existing_stable_coordinates_are_rejected() -> None:
    block = _table(
        "T_0_2",
        "<table><tr><td>Sample</td><td>Value</td></tr>"
        "<tr><td>S1</td><td>3.2</td></tr></table>",
    )

    assert resolve_table_locator(block, {
        "table_id": "T_0_2",
        "row_label": "S1",
        "column_label": "Value",
        "cell_value": "3.2",
        "cell_id": "T_0_2:r0000:c0000",
        "row_index": 0,
        "column_index": 0,
    }) is None


def test_numeric_and_hyphen_substrings_do_not_create_false_matches() -> None:
    block = _table(
        "T_0_3",
        "<table><tr><td>No.</td><td>Solvent</td><td>Value</td></tr>"
        "<tr><td>2</td><td>Cyclohexane</td><td>8.20</td></tr>"
        "<tr><td>7</td><td>p-Cymene</td><td>8.20</td></tr>"
        "<tr><td>20</td><td>Bromobenzene</td><td>-</td></tr></table>",
    )

    resolved = resolve_table_locator(block, {
        "table_id": "T_0_3",
        "row_label": "p-Cymene",
        "column_label": "corrupt header",
        "cell_value": "8.20",
    })

    assert resolved is not None
    assert resolved["cell_id"] == "T_0_3:r0002:c0002"


def test_empty_cell_is_resolved_by_unique_row_and_column() -> None:
    block = _table(
        "T_0_4",
        "<table><tr><td>T (K)</td><td>PTS</td><td>TS</td></tr>"
        "<tr><td>360</td><td>1.156</td><td></td></tr></table>",
    )

    resolved = resolve_table_locator(block, {
        "table_id": "T_0_4",
        "row_label": "360",
        "column_label": "TS",
        "cell_value": None,
    })

    assert resolved is not None
    assert resolved["cell_id"] == "T_0_4:r0001:c0002"


def test_existing_stable_coordinates_accept_only_an_actually_empty_cell() -> None:
    block = _table(
        "T_0_5",
        "<table><tr><td>Sample</td><td>Value</td></tr>"
        "<tr><td>S1</td><td></td></tr>"
        "<tr><td>S2</td><td>3.2</td></tr></table>",
    )
    locator = {
        "table_id": "T_0_5",
        "row_label": "S1",
        "column_label": "Value",
        "cell_value": None,
        "cell_id": "T_0_5:r0001:c0001",
        "row_index": 1,
        "column_index": 1,
    }

    resolved = resolve_table_locator(block, locator)

    assert resolved is not None
    assert resolved["cell_id"] == "T_0_5:r0001:c0001"
    assert resolve_table_locator(block, {
        **locator,
        "row_label": "S2",
        "cell_id": "T_0_5:r0002:c0001",
        "row_index": 2,
    }) is None
