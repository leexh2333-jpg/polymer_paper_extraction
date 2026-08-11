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


def _write_stage_files(input_dir: Path, ref_no: str) -> None:
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


def test_publish_candidate_rejects_missing_input_dir(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "output"

    with pytest.raises(CandidatePublishError) as excinfo:
        publish_candidate(
            "reference_no_9999999",
            input_root=input_root,
            output_root=output_root,
        )

    assert "输入目录不存在" in str(excinfo.value)
    # 关键：失败时不能留下任何输出。
    assert not output_root.exists()


def test_publish_candidate_rejects_ref_no_without_prefix(tmp_path: Path) -> None:
    """传短号时必须报错，而不是静默产出 0 条 observation 的 candidate。"""
    ref_no = "reference_no_0000001"
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_stage_files(input_root / ref_no, ref_no)

    with pytest.raises(CandidatePublishError) as excinfo:
        publish_candidate(
            "0000001",
            input_root=input_root,
            output_root=output_root,
        )

    message = str(excinfo.value)
    assert "输入目录不存在" in message
    # 目录确实存在、只是少了前缀时，要给出可直接照做的提示。
    assert "reference_no_0000001" in message
    assert not output_root.exists()
    assert not (input_root / "0000001").exists()


def test_publish_candidate_still_accepts_full_reference_no(tmp_path: Path) -> None:
    """守卫不能影响正常路径：完整目录名照常发布。"""
    ref_no = "reference_no_0000001"
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_stage_files(input_root / ref_no, ref_no)

    candidate_path, report_path = publish_candidate(
        ref_no,
        input_root=input_root,
        output_root=output_root,
    )

    assert candidate_path.is_file()
    assert report_path.is_file()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["document_id"] == ref_no
    # _stages() 的 stage4/stage5 性质列表本来就是空的，这里要断言的是
    # Stage 全部读到了（区别于目录不存在时的静默空结果）。
    assert candidate["publication"]["status"] == "complete"
    assert candidate["material_mentions"]


def test_publish_candidate_missing_stage_files_still_publishes(tmp_path: Path) -> None:
    """目录在、只缺 Stage 文件时仍要发布 candidate（candidate_partial 行为不变）。"""
    ref_no = "reference_no_0000001"
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_dir = input_root / ref_no
    input_dir.mkdir(parents=True)
    stage0 = _stages(ref_no)["stage0"]
    (input_dir / "stage0_blocks.json").write_text(
        json.dumps(stage0), encoding="utf-8"
    )

    candidate_path, report_path = publish_candidate(
        ref_no,
        input_root=input_root,
        output_root=output_root,
    )

    assert candidate_path.is_file()
    assert report_path.is_file()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["publication"]["status"] != "complete"


def test_publish_candidate_cli_returns_nonzero_on_missing_ref(
    tmp_path: Path,
) -> None:
    import subprocess

    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "output"
    script = PREVIEW_ROOT / "publish_candidate.py"

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--ref-no", "reference_no_9999999",
            "--input-root", str(input_root),
            "--output-root", str(output_root),
        ],
        capture_output=True, text=True, encoding="utf-8",
    )

    assert result.returncode != 0
    assert "输入目录不存在" in result.stderr
    assert not output_root.exists()
