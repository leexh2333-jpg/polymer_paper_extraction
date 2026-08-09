import json
import sys
from pathlib import Path

import pytest


PREVIEW_ROOT = Path(__file__).resolve().parents[1]
if str(PREVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(PREVIEW_ROOT))

from publish_candidate import (  # noqa: E402
    CandidatePublishError,
    build_candidate_payload,
    publish_candidate,
)


def _stages(ref_no: str = "reference_no_0000001") -> dict:
    evidence = {
        "block_id": "P_0_1",
        "page": 0,
        "source_type": "paragraph",
        "source_sentence": "Polymer A was prepared.",
    }
    return {
        "stage0": {
            "schema_version": "1.0",
            "document_id": ref_no,
            "paper": {"title": "Candidate paper", "doi": None},
            "ocr": {"status": "done"},
            "elements": [{"block_id": "P_0_1", "type": "text", "page": 0, "text": evidence["source_sentence"]}],
            "warnings": [],
        },
        "stage1": {
            "schema_version": "1.0",
            "document_id": ref_no,
            "material_mentions": [{"mention_id": "m001", "text": "Polymer A", "evidence": evidence}],
            "warnings": [],
        },
        "stage2": {
            "schema_version": "1.0",
            "document_id": ref_no,
            "polymer_entities": [{"entity_id": "pe001", "polymer_name": "Polymer A", "evidence": evidence}],
            "unresolved_mention_ids": [],
            "warnings": [],
        },
        "stage3": {
            "schema_version": "1.0",
            "document_id": ref_no,
            "samples": [],
            "process_steps": [],
            "unresolved_entity_ids": [],
            "warnings": [],
        },
        "stage4": {
            "schema_version": "1.0",
            "document_id": ref_no,
            "measurement_conditions": [],
            "properties": [],
            "unresolved_properties": [],
            "property_series": [],
            "warnings": [],
        },
        "stage5": {
            "schema_version": "1.0",
            "document_id": ref_no,
            "characterizations": [],
            "properties": [],
            "warnings": [],
        },
    }


def test_build_candidate_flattens_stages_and_registers_evidence() -> None:
    candidate = build_candidate_payload("reference_no_0000001", _stages())

    assert candidate["publication"]["status"] == "complete"
    assert candidate["publication"]["validation_status"] == "not_validated"
    assert candidate["material_mentions"][0]["evidence_ids"] == ["ev00001"]
    assert candidate["polymer_entities"][0]["evidence_ids"] == ["ev00001"]
    assert len(candidate["evidence"]) == 1


def test_publish_candidate_writes_json_and_html(tmp_path: Path) -> None:
    ref_no = "reference_no_0000001"
    input_dir = tmp_path / "input" / ref_no
    input_dir.mkdir(parents=True)
    for stage_name, payload in _stages(ref_no).items():
        filename = {
            "stage0": "stage0_blocks.json",
            "stage1": "stage1_mentions.json",
            "stage2": "stage2_entities.json",
            "stage3": "stage3_process.json",
            "stage4": "stage4_properties.json",
            "stage5": "stage5_characterizations.json",
        }[stage_name]
        (input_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    candidate_path, report_path = publish_candidate(
        ref_no,
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
    )

    assert candidate_path.is_file()
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "候选结果 · Stage 0-5 已完成" in report
    assert "未经完整科学语义校验" in report


def test_publish_candidate_recovers_partial_stage_from_failure(
    tmp_path: Path,
) -> None:
    ref_no = "reference_no_0000001"
    input_dir = tmp_path / "input" / ref_no
    input_dir.mkdir(parents=True)
    stages = _stages(ref_no)
    for stage_name in ("stage0", "stage1"):
        filename = {
            "stage0": "stage0_blocks.json",
            "stage1": "stage1_mentions.json",
        }[stage_name]
        (input_dir / filename).write_text(
            json.dumps(stages[stage_name]),
            encoding="utf-8",
        )
    failure = {
        "status": "failed",
        "stage": "stage2_polymer_entity",
        "document_id": ref_no,
        "error_type": "Stage2Error",
        "error": "nested mentions split",
        "raw_response": {
            "content": json.dumps({
                "entities": [{
                    "entity_id": "pe001",
                    "polymer_name": "Polymer A",
                    "evidence": {
                        "block_id": "P_0_1",
                        "page": 0,
                        "source_type": "paragraph",
                        "source_sentence": "Polymer A was prepared.",
                    },
                }]
            })
        },
    }
    (input_dir / "stage2_failure.json").write_text(
        json.dumps(failure),
        encoding="utf-8",
    )

    candidate_path, report_path = publish_candidate(
        ref_no,
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert candidate["publication"]["status"] == "partial"
    assert candidate["publication"]["candidate_stages"] == ["stage2"]
    assert candidate["polymer_entities"][0]["entity_id"] == "pe001"
    assert candidate["stage_failures"][0]["stage"] == "stage2"
    assert "部分抽取结果" in report_path.read_text(encoding="utf-8")


def test_candidate_rejects_sensitive_fields() -> None:
    stages = _stages()
    stages["stage1"]["provenance"] = {"api_key": "not-for-publication"}

    with pytest.raises(CandidatePublishError, match="敏感"):
        build_candidate_payload("reference_no_0000001", stages)



def test_candidate_keeps_stage4_scalar_series_unresolved_and_stage5_properties() -> None:
    stages = _stages()
    evidence = {
        "block_id": "P_0_1",
        "page": 0,
        "source_type": "paragraph",
        "source_sentence": "Polymer A was prepared.",
    }
    stages["stage4"].update({
        "measurement_conditions": [{
            "condition_id": "mc001",
            "condition_status": "not_reported",
            "evidence": evidence,
        }],
        "properties": [{
            "property_id": "prop001",
            "sample_id": "s001",
            "property_name_raw": "Tg",
            "value_raw": "100",
            "measurement_condition_id": "mc001",
            "evidence": [evidence],
        }],
        "unresolved_properties": [{
            "unresolved_id": "up001",
            "property_name_raw": "modulus",
            "value_raw": "2.0",
            "evidence": [evidence],
        }],
        "property_series": [{
            "series_id": "series001",
            "property_name_raw": "Tensile strength",
            "points": [{
                "point_id": "pt001",
                "value_raw": "50",
                "evidence": [evidence],
            }],
            "evidence": [evidence],
        }],
    })
    stages["stage5"]["properties"] = [{
        "property_id": "stage5prop001",
        "property_name_raw": "crystallinity",
        "value_raw": "35",
        "evidence": [evidence],
    }]

    candidate = build_candidate_payload("reference_no_0000001", stages)

    assert [item["property_id"] for item in candidate["property_observations"]] == [
        "prop001",
        "stage5prop001",
    ]
    assert candidate["measurement_conditions"][0]["condition_id"] == "mc001"
    assert candidate["unresolved_property_observations"][0]["unresolved_id"] == "up001"
    assert candidate["property_series"][0]["points"][0]["value_raw"] == "50"
    assert candidate["property_series"][0]["evidence_ids"]
