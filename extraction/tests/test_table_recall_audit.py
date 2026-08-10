from __future__ import annotations

from schema.polymer_schema import Stage0Document, Stage0Element
from stages.table_grid import parse_table_cells
from stages.table_recall_audit import (
    ROLE_COORDINATE,
    ROLE_PROPERTY,
    ROLE_UNKNOWN,
    audit_documents,
)


def _stage0(*tables: tuple[str, str, str]) -> Stage0Document:
    elements = []
    for index, (block_id, caption, body) in enumerate(tables):
        elements.append(Stage0Element(
            block_id=block_id,
            type="table",
            section="Results",
            page=index,
            source_block_index=index,
            caption=caption,
            table_body=body,
            table_cells=parse_table_cells(body, block_id),
        ))
    return Stage0Document.model_validate({
        "schema_version": "1.1",
        "source_document_schema_version": "1.0",
        "document_id": "reference_no_test",
        "paper": {
            "ref_no": "reference_no_test",
            "pdf_filename": "test.pdf",
            "source_pdf_path": "test.pdf",
            "organized_pdf_path": "test.pdf",
            "metadata_status": "failed",
            "metadata_extraction": {},
        },
        "source_files": {},
        "ocr": {},
        "elements": [item.model_dump(mode="json") for item in elements],
        "warnings": [],
    })


def _stage4(**updates):
    result = {
        "document_id": "reference_no_test",
        "measurement_conditions": [],
        "properties": [],
        "unresolved_properties": [],
        "property_series": [],
    }
    result.update(updates)
    return result


def _evidence(cell_id: str, value: str):
    table_id, coordinates = cell_id.split(":", 1)
    row_text, column_text = coordinates.split(":")
    return {
        "block_id": table_id,
        "source_type": "table",
        "source_sentence": value,
        "table_locator": {
            "table_id": table_id,
            "cell_id": cell_id,
            "row_index": int(row_text[1:]),
            "column_index": int(column_text[1:]),
            "cell_value": value,
        },
    }


def test_zero_coverage_property_table_is_recovery_gap() -> None:
    stage0 = _stage0((
        "T_1_1",
        "Electrical properties",
        "<table><tr><th>Sample</th><th>Conductivity (S/cm)</th></tr>"
        "<tr><td>A</td><td>1.0e-3</td></tr>"
        "<tr><td>B</td><td>2.0e-3</td></tr></table>",
    ))

    report = audit_documents(stage0, _stage4())
    table = report["tables"][0]

    assert table["property_value_candidate_count"] == 2
    assert table["covered_property_value_candidate_count"] == 0
    assert table["property_zero_coverage"] is True
    assert table["needs_recovery"] is True
    assert {item["cell_id"] for item in table["missing_property_cells"]} == {
        "T_1_1:r0001:c0001",
        "T_1_1:r0002:c0001",
    }


def test_coordinate_cell_does_not_count_as_property_value() -> None:
    stage0 = _stage0((
        "T_1_2",
        "Conductivity over time",
        "<table><tr><th>Time (s)</th><th>Conductivity (S/cm)</th></tr>"
        "<tr><td>10</td><td>1.0e-3</td></tr></table>",
    ))
    stage4 = _stage4(property_series=[{
        "points": [{
            "evidence": [_evidence("T_1_2:r0001:c0001", "1.0e-3")],
            "coordinates": [{
                "evidence": _evidence("T_1_2:r0001:c0000", "10"),
            }],
        }],
    }])

    report = audit_documents(stage0, stage4)
    cells = {item["cell_id"]: item for item in report["tables"][0]["cells"]}

    assert cells["T_1_2:r0001:c0000"]["role"] == ROLE_COORDINATE
    assert "coordinate" in cells["T_1_2:r0001:c0000"]["covered_as"]
    assert "property_value" not in cells["T_1_2:r0001:c0000"]["covered_as"]
    assert cells["T_1_2:r0001:c0001"]["role"] == ROLE_PROPERTY
    assert "property_value" in cells["T_1_2:r0001:c0001"]["covered_as"]
    assert report["tables"][0]["property_value_ratio"] == 1.0


def test_condition_evidence_does_not_hide_missing_property_value() -> None:
    stage0 = _stage0((
        "T_1_3",
        "Thermal properties",
        "<table><tr><th>Sample</th><th>Tg (°C)</th></tr>"
        "<tr><td>A</td><td>125</td></tr></table>",
    ))
    stage4 = _stage4(measurement_conditions=[{
        "evidence": _evidence("T_1_3:r0001:c0001", "125"),
    }])

    report = audit_documents(stage0, stage4)
    cell = report["tables"][0]["cells"][0]

    assert cell["role"] == ROLE_PROPERTY
    assert "condition" in cell["covered_as"]
    assert "property_value" not in cell["covered_as"]
    assert report["tables"][0]["missing_property_value_candidate_count"] == 1


def test_duplicate_property_evidence_counts_cell_once() -> None:
    stage0 = _stage0((
        "T_1_4",
        "Mechanical properties",
        "<table><tr><th>Sample</th><th>Tensile strength at break (MPa)</th></tr>"
        "<tr><td>A</td><td>42</td></tr></table>",
    ))
    evidence = _evidence("T_1_4:r0001:c0001", "42")
    stage4 = _stage4(properties=[
        {"evidence": [evidence]},
        {"evidence": [evidence]},
    ])

    report = audit_documents(stage0, stage4)

    assert report["summary"]["property_value_candidate_count"] == 1
    assert report["summary"]["covered_property_value_candidate_count"] == 1


def test_recipe_numbers_remain_unknown_instead_of_forced_property() -> None:
    stage0 = _stage0((
        "T_1_5",
        "Feed composition",
        "<table><tr><th>Run</th><th>Monomer amount</th></tr>"
        "<tr><td>1</td><td>25</td></tr></table>",
    ))

    report = audit_documents(stage0, _stage4())
    roles = {item["role"] for item in report["tables"][0]["cells"]}

    assert ROLE_PROPERTY not in roles
    assert ROLE_UNKNOWN in roles
    assert report["tables"][0]["needs_recovery"] is False



def test_parameters_text_does_not_match_eta_alias() -> None:
    stage0 = _stage0((
        "T_1_6",
        "Fuoss-Kirkwood Parameters",
        "<table><tr><th>Sample</th><th>Parameter m</th></tr>"
        "<tr><td>A</td><td>0.42</td></tr></table>",
    ))

    report = audit_documents(stage0, _stage4())

    assert report["summary"]["property_value_candidate_count"] == 0
    assert report["tables"][0]["cells"][0]["role"] == ROLE_UNKNOWN


def test_td_only_header_rows_are_not_replaced_by_previous_data_values() -> None:
    stage0 = _stage0((
        "T_2_1",
        "Thermal and surface properties",
        "<table>"
        "<tr><td>Sample</td><td>Tg (°C)</td><td>Tm (°C)</td><td>Surface Energy (160 °C)</td></tr>"
        "<tr><td>A</td><td>105</td><td>166</td><td>16.0</td></tr>"
        "<tr><td>B</td><td>119</td><td>172</td><td>48.9</td></tr>"
        "</table>",
    ))

    report = audit_documents(stage0, _stage4())
    cells = {item["cell_id"]: item for item in report["tables"][0]["cells"]}

    assert cells["T_2_1:r0001:c0001"]["property_name_normalized"] == "glass_transition_temperature"
    assert cells["T_2_1:r0002:c0001"]["property_name_normalized"] == "glass_transition_temperature"
    assert cells["T_2_1:r0001:c0002"]["property_name_normalized"] == "melting_temperature"
    assert cells["T_2_1:r0002:c0002"]["property_name_normalized"] == "melting_temperature"
    assert cells["T_2_1:r0001:c0003"]["property_name_normalized"] == "surface_tension"
    assert cells["T_2_1:r0002:c0003"]["property_name_normalized"] == "surface_tension"


def test_numeric_formula_headers_are_not_counted_as_data_cells() -> None:
    stage0 = _stage0((
        "T_2_2",
        "Thermal properties",
        "<table>"
        "<tr><td>Sample</td><td>$T_d^{10\\%}$ (°C)</td><td>Surface Energy (160 °C)</td></tr>"
        "<tr><td>A</td><td>403</td><td>26.1</td></tr>"
        "</table>",
    ))

    report = audit_documents(stage0, _stage4())
    table = report["tables"][0]
    cells = {item["cell_id"]: item for item in table["cells"]}

    assert table["numeric_cell_count"] == 2
    assert cells["T_2_2:r0001:c0001"]["property_name_normalized"] == "thermal_decomposition_temperature"
    assert cells["T_2_2:r0001:c0002"]["property_name_normalized"] == "surface_tension"


def test_storage_modulus_symbol_and_legacy_melting_headers_are_properties() -> None:
    stage0 = _stage0((
        "T_2_3",
        "Reported polymer properties",
        "<table>"
        "<tr><td>Sample</td><td>E' / MPa</td><td>M.P. °C</td><td>PMT °C</td></tr>"
        "<tr><td>A</td><td>1.3</td><td>285c</td><td>290</td></tr>"
        "</table>",
    ))

    report = audit_documents(stage0, _stage4())
    cells = {item["cell_id"]: item for item in report["tables"][0]["cells"]}

    assert cells["T_2_3:r0001:c0001"]["property_name_normalized"] == "dynamic_tensile_properties"
    assert cells["T_2_3:r0001:c0002"]["property_name_normalized"] == "melting_temperature"
    assert cells["T_2_3:r0001:c0003"]["property_name_normalized"] == "melting_temperature"
