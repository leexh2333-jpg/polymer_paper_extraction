from pathlib import Path

from stages.stage4r_table_recovery import (
    _prepare_stage4_input,
    build_parser,
    infer_entity_id,
    next_unresolved_number,
)


def test_unresolved_ids_only_append_after_existing_ids() -> None:
    stage4 = {"unresolved_properties": [
        {"unresolved_id": "uprop002"},
        {"unresolved_id": "uprop010"},
    ]}
    assert next_unresolved_number(stage4) == 11


def test_unique_row_label_resolves_to_sample_entity() -> None:
    entity, basis = infer_entity_id(
        row_headers=["Sample-A"],
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[
            {"sample_label_raw": "Sample-A", "polymer_name": "Poly A", "refers_to_entity": "pe001"},
            {"sample_label_raw": "Sample-B", "polymer_name": "Poly B", "refers_to_entity": "pe002"},
        ],
        table_entities={},
    )
    assert entity == "pe001"
    assert basis == "row_label_exact_entity_alias"


def test_ambiguous_multi_entity_document_does_not_choose_first() -> None:
    entity, basis = infer_entity_id(
        row_headers=["unknown row"],
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[],
        table_entities={},
    )
    assert entity is None
    assert basis == "entity_ambiguous"


def test_single_existing_table_entity_is_safe_fallback() -> None:
    entity, basis = infer_entity_id(
        row_headers=[],
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[],
        table_entities={"T_1_1": {"pe002"}},
    )
    assert entity == "pe002"
    assert basis == "table_existing_entity"

from types import SimpleNamespace
from stages.stage4r_table_recovery import build_unresolved_property


def test_build_uses_normalized_property_name_not_trailing_data_header() -> None:
    table = SimpleNamespace(block_id="T_1_1", page=1, bbox=None)
    cell = SimpleNamespace(row_index=2, column_index=1, text="2.8")
    target = SimpleNamespace(row_index=3, column_index=1, text="1.5")
    item = build_unresolved_property(
        unresolved_id="uprop001",
        entity_id="pe001",
        cell_report={
            "cell_id": "T_1_1:r0003:c0001",
            "row_index": 3,
            "column_index": 1,
            "text": "1.5",
            "column_headers": ["E' / MPa", "1.3", "2.8"],
            "row_headers": ["Sample A"],
            "property_name_normalized": "dynamic_tensile_properties",
        },
        table=table,
        cells=[cell, target],
    )
    assert item["property_name_raw"] == "dynamic_tensile_properties"
    assert item["property_name_raw"] not in {"2.8", "-"}




def test_exact_row_label_beats_shorter_nested_sample_label() -> None:
    entity, basis = infer_entity_id(
        row_headers=["0-2-0-I"],
        row_index=3,
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[
            {"sample_label_raw": "0-2", "refers_to_entity": "pe001"},
            {"sample_label_raw": "0-2-0-I", "refers_to_entity": "pe002"},
        ],
        table_entities={},
    )
    assert entity == "pe002"
    assert basis == "row_label_exact_entity_alias"


def test_unique_stage3_sample_entity_is_safe_fallback() -> None:
    entity, basis = infer_entity_id(
        row_headers=[],
        row_index=2,
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[
            {"sample_label_raw": "A", "refers_to_entity": "pe001"},
            {"sample_label_raw": "B", "refers_to_entity": "pe001"},
        ],
        table_entities={},
    )
    assert entity == "pe001"
    assert basis == "document_samples_single_entity"


def test_longest_alias_does_not_bind_blend_to_one_component() -> None:
    entity, basis = infer_entity_id(
        row_headers=["50/50 Blend", "PM (EO/PO) polymer"],
        row_index=4,
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[],
        entity_aliases={"pe002": {"PM (EO/PO)"}},
        table_entities={},
    )
    assert entity is None
    assert basis == "entity_ambiguous"


def test_single_sample_entity_fallback_requires_empty_row_label() -> None:
    entity, basis = infer_entity_id(
        row_headers=["Electrical conductivity (S/cm)"],
        row_index=1,
        table_id="T_1_1",
        valid_entity_ids={"pe001", "pe002"},
        samples=[{"sample_label_raw": "COC", "refers_to_entity": "pe001"}],
        table_entities={"T_1_1": {"pe001"}},
    )
    assert entity is None
    assert basis == "entity_ambiguous"


def test_parser_accepts_batch_runner_compatibility_arguments(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "--input-root", str(tmp_path),
        "--output-root", str(tmp_path),
        "--ref-no", "reference_no_0000001",
        "--config", str(tmp_path / "pipeline.yaml"),
        "--apply",
        "--force",
    ])
    assert args.apply is True
    assert args.force is True


def test_prepare_stage4_input_uses_backup_when_forcing_applied_result(tmp_path: Path) -> None:
    stage4 = tmp_path / "stage4_properties.json"
    preview = tmp_path / "stage4_properties.recovery_preview.json"
    report = tmp_path / "stage4r_recovery.json"
    backup = tmp_path / "stage4_properties.pre_recovery.json"
    stage4.write_text("recovered", encoding="utf-8")
    preview.write_text("recovered", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    backup.write_text("original", encoding="utf-8")

    source, cached = _prepare_stage4_input(
        stage4_path=stage4,
        preview_path=preview,
        report_path=report,
        backup_path=backup,
        apply=True,
        force=True,
        in_place=True,
    )

    assert cached is False
    assert source == backup
    assert backup.read_text(encoding="utf-8") == "original"


def test_prepare_stage4_input_refreshes_backup_for_new_stage4(tmp_path: Path) -> None:
    stage4 = tmp_path / "stage4_properties.json"
    preview = tmp_path / "stage4_properties.recovery_preview.json"
    report = tmp_path / "stage4r_recovery.json"
    backup = tmp_path / "stage4_properties.pre_recovery.json"
    stage4.write_text("new-stage4", encoding="utf-8")
    preview.write_text("old-recovery", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    backup.write_text("old-stage4", encoding="utf-8")

    source, cached = _prepare_stage4_input(
        stage4_path=stage4,
        preview_path=preview,
        report_path=report,
        backup_path=backup,
        apply=True,
        force=False,
        in_place=True,
    )

    assert cached is False
    assert source == stage4
    assert backup.read_text(encoding="utf-8") == "new-stage4"
