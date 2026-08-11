"""把已有 Stage 0-5 输出合并为 candidate.json 和候选 HTML。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable


TESTCODE_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_ROOT = TESTCODE_ROOT / "extraction"
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from reports.render_extraction_html import render_extraction_html  # noqa: E402
from stages.stage4_property import write_json_atomic  # noqa: E402
from stages.stage6_validate_merge import _scan_sensitive  # noqa: E402


STAGE_FILES = {
    "stage0": "stage0_blocks.json",
    "stage1": "stage1_mentions.json",
    "stage2": "stage2_entities.json",
    "stage3": "stage3_process.json",
    "stage4": "stage4_properties.json",
    "stage5": "stage5_characterizations.json",
}
STAGE_FAILURE_FILES = {
    "stage0": "stage0_failure.json",
    "stage1": "stage1_failure.json",
    "stage2": "stage2_failure.json",
    "stage3": "stage3_failure.json",
    "stage4": "stage4_failure.json",
    "stage5": "stage5_failure.json",
}
STAGE_COLLECTION_ALIASES = {
    "stage1": {"material_mentions": ("material_mentions", "mentions")},
    "stage2": {"polymer_entities": ("polymer_entities", "entities")},
    "stage3": {
        "samples": ("samples",),
        "process_steps": ("process_steps", "steps"),
    },
    "stage4": {
        "measurement_conditions": ("measurement_conditions", "conditions"),
        "properties": ("properties",),
        "unresolved_properties": ("unresolved_properties",),
        "property_series": ("property_series", "series"),
    },
    "stage5": {
        "characterizations": ("characterizations",),
        "properties": ("properties",),
    },
}
ID_FIELDS = (
    "mention_id",
    "entity_id",
    "sample_id",
    "step_id",
    "condition_id",
    "property_id",
    "unresolved_id",
    "series_id",
    "point_id",
    "characterization_id",
)


class CandidatePublishError(RuntimeError):
    """候选结果缺少必要输入、无法读取或不适合发布。"""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise CandidatePublishError(f"无法读取 {label}：{path}") from exc
    except json.JSONDecodeError as exc:
        raise CandidatePublishError(f"{label} 不是有效 JSON：{path.name}") from exc
    if not isinstance(payload, dict):
        raise CandidatePublishError(f"{label} 顶层必须是 JSON object：{path.name}")
    return payload


def _parse_json_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return copy.deepcopy(content)
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _candidate_stage_from_failure(
    stage_name: str,
    failure: dict[str, Any],
) -> dict[str, Any] | None:
    raw_response = failure.get("raw_response")
    content = raw_response.get("content") if isinstance(raw_response, dict) else None
    payload = _parse_json_content(content)
    if payload is None:
        return None
    normalized = copy.deepcopy(payload)
    for target, aliases in STAGE_COLLECTION_ALIASES.get(stage_name, {}).items():
        for alias in aliases:
            if isinstance(payload.get(alias), list):
                normalized.setdefault(target, copy.deepcopy(payload[alias]))
                break
    normalized.setdefault("schema_version", "candidate.raw")
    normalized.setdefault("document_id", failure.get("document_id"))
    normalized.setdefault("warnings", [])
    return normalized


def load_candidate_sources(
    ref_no: str,
    input_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    stages: dict[str, dict[str, Any]] = {}
    stage_states: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    for stage_name, filename in STAGE_FILES.items():
        stage_path = input_dir / filename
        if stage_path.is_file():
            try:
                stages[stage_name] = _load_json_object(stage_path, stage_name)
                stage_states[stage_name] = "completed"
                continue
            except CandidatePublishError as exc:
                failures.append({
                    "stage": stage_name,
                    "error_type": "InvalidStageOutput",
                    "error": str(exc),
                    "raw_candidate_preserved": False,
                })

        failure_path = input_dir / STAGE_FAILURE_FILES[stage_name]
        failure: dict[str, Any] = {}
        if failure_path.is_file():
            try:
                failure = _load_json_object(failure_path, f"{stage_name} failure")
            except CandidatePublishError as exc:
                failure = {
                    "error_type": "InvalidFailureOutput",
                    "error": str(exc),
                }
        candidate_stage = _candidate_stage_from_failure(stage_name, failure)
        if candidate_stage is not None:
            stages[stage_name] = candidate_stage
            stage_states[stage_name] = "candidate_from_failure"
        else:
            stage_states[stage_name] = "missing"
        failures.append({
            "stage": stage_name,
            "error_type": str(failure.get("error_type") or "StageOutputMissing"),
            "error": str(
                failure.get("error")
                or f"{stage_name} 没有可用的成功输出或可解析候选响应"
            ),
            "raw_candidate_preserved": candidate_stage is not None,
        })
    return stages, stage_states, failures


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value if isinstance(item, dict)]


def _object_id(value: dict[str, Any], fallback: str | None = None) -> str | None:
    for field in ID_FIELDS:
        candidate = value.get(field)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return fallback


class EvidenceRegistry:
    """为候选展示补充 evidence_id，不改变原始 inline evidence。"""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._ids_by_payload: dict[str, str] = {}

    def add_many(
        self,
        raw: Any,
        *,
        source_stage: str,
        object_id: str | None,
    ) -> list[str]:
        if isinstance(raw, dict):
            values: Iterable[Any] = (raw,)
        elif isinstance(raw, list):
            values = raw
        else:
            return []

        evidence_ids: list[str] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            canonical = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            evidence_id = self._ids_by_payload.get(canonical)
            if evidence_id is None:
                evidence_id = f"ev{len(self.items) + 1:05d}"
                item = copy.deepcopy(value)
                item["evidence_id"] = evidence_id
                item.setdefault("source_stage", source_stage)
                if object_id:
                    item.setdefault("object_id", object_id)
                self.items.append(item)
                self._ids_by_payload[canonical] = evidence_id
            evidence_ids.append(evidence_id)
        return list(dict.fromkeys(evidence_ids))


def _attach_evidence_ids(
    value: Any,
    registry: EvidenceRegistry,
    *,
    source_stage: str,
    parent_id: str | None = None,
) -> None:
    if isinstance(value, list):
        for child in value:
            _attach_evidence_ids(
                child,
                registry,
                source_stage=source_stage,
                parent_id=parent_id,
            )
        return
    if not isinstance(value, dict):
        return

    object_id = _object_id(value, parent_id)
    evidence_ids = registry.add_many(
        value.get("evidence"),
        source_stage=source_stage,
        object_id=object_id,
    )
    if evidence_ids:
        existing = value.get("evidence_ids")
        existing_ids = existing if isinstance(existing, list) else []
        value["evidence_ids"] = list(dict.fromkeys([*existing_ids, *evidence_ids]))
    for key, child in value.items():
        if key not in {"evidence", "evidence_ids"}:
            _attach_evidence_ids(
                child,
                registry,
                source_stage=source_stage,
                parent_id=object_id,
            )


def _normalized_warnings(stages: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for stage_name, payload in stages.items():
        for warning in payload.get("warnings") or []:
            if isinstance(warning, dict):
                item = copy.deepcopy(warning)
                item.setdefault("stage", stage_name)
                item.setdefault("code", "upstream_warning")
                item.setdefault("message", json.dumps(warning, ensure_ascii=False))
            else:
                item = {
                    "stage": stage_name,
                    "code": "upstream_warning",
                    "message": str(warning),
                }
            warnings.append(item)
    return warnings


def build_candidate_payload(
    ref_no: str,
    stages: dict[str, dict[str, Any]],
    *,
    stage_states: dict[str, str] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stage_states = stage_states or {
        stage_name: "completed" for stage_name in STAGE_FILES
    }
    failures = failures or []
    stage0 = stages.get("stage0") or {}
    stage1 = stages.get("stage1") or {}
    stage2 = stages.get("stage2") or {}
    stage3 = stages.get("stage3") or {}
    stage4 = stages.get("stage4") or {}
    stage5 = stages.get("stage5") or {}
    registry = EvidenceRegistry()

    collections: list[tuple[str, list[dict[str, Any]], str]] = [
        ("material_mentions", _as_dict_list(stage1.get("material_mentions")), "stage1_material_mention"),
        ("polymer_entities", _as_dict_list(stage2.get("polymer_entities")), "stage2_polymer_entity"),
        ("samples", _as_dict_list(stage3.get("samples")), "stage3_sample_process"),
        ("process_steps", _as_dict_list(stage3.get("process_steps")), "stage3_sample_process"),
        ("measurement_conditions", _as_dict_list(stage4.get("measurement_conditions")), "stage4_property"),
        ("stage4_properties", _as_dict_list(stage4.get("properties")), "stage4_property"),
        ("unresolved_property_observations", _as_dict_list(stage4.get("unresolved_properties")), "stage4_property"),
        ("property_series", _as_dict_list(stage4.get("property_series")), "stage4_property"),
        ("characterizations", _as_dict_list(stage5.get("characterizations")), "stage5_characterization"),
        ("stage5_properties", _as_dict_list(stage5.get("properties")), "stage5_characterization"),
    ]
    values: dict[str, list[dict[str, Any]]] = {}
    for name, items, source_stage in collections:
        _attach_evidence_ids(items, registry, source_stage=source_stage)
        values[name] = items

    stage_document_ids = {
        str(payload.get("document_id"))
        for payload in stages.values()
        if payload.get("document_id")
    }
    warnings = _normalized_warnings(stages)
    for failure in failures:
        warnings.append({
            "stage": failure["stage"],
            "code": "stage_failed_candidate_preserved",
            "message": failure["error"],
        })
    if stage_document_ids and stage_document_ids != {ref_no}:
        warnings.append({
            "stage": "candidate_publish",
            "code": "document_id_mismatch",
            "message": (
                f"候选发布编号 {ref_no} 与 Stage 0-5 编号不完全一致："
                f"{sorted(stage_document_ids)}"
            ),
        })

    provenance: list[dict[str, Any]] = []
    ocr = stage0.get("ocr")
    if isinstance(ocr, dict):
        provenance.append({"stage": "ocr", **copy.deepcopy(ocr)})
    for stage_name in ("stage1", "stage2", "stage3", "stage4", "stage5"):
        item = (stages.get(stage_name) or {}).get("provenance")
        if isinstance(item, dict):
            provenance.append(copy.deepcopy(item))

    completed_stages = [
        stage_name
        for stage_name in STAGE_FILES
        if stage_states.get(stage_name) == "completed"
    ]
    candidate_stages = [
        stage_name
        for stage_name in STAGE_FILES
        if stage_states.get(stage_name) == "candidate_from_failure"
    ]
    failed_stages = [
        stage_name
        for stage_name in STAGE_FILES
        if stage_states.get(stage_name) != "completed"
    ]
    publication_status = "complete" if not failed_stages else "partial"
    publication_message = (
        "Stage 0-5 已完成；未经完整科学语义校验，仅用于预览模型抽取结果。"
        if publication_status == "complete"
        else (
            "部分抽取结果；未经完整科学语义校验。"
            f"未完成阶段：{', '.join(failed_stages)}。"
        )
    )

    payload = {
        "schema_version": "candidate.v1",
        "document_id": ref_no,
        "publication": {
            "kind": "candidate",
            "status": publication_status,
            "validation_status": "not_validated",
            "message": publication_message,
            "completed_stages": completed_stages,
            "candidate_stages": candidate_stages,
            "failed_stages": failed_stages,
        },
        "paper": copy.deepcopy(stage0.get("paper") or {}),
        "material_mentions": values["material_mentions"],
        "polymer_entities": values["polymer_entities"],
        "unresolved_mention_ids": copy.deepcopy(stage2.get("unresolved_mention_ids") or []),
        "samples": values["samples"],
        "process_steps": values["process_steps"],
        "unresolved_entity_ids": copy.deepcopy(stage3.get("unresolved_entity_ids") or []),
        "property_observations": [
            *values["stage4_properties"],
            *values["stage5_properties"],
        ],
        "measurement_conditions": values["measurement_conditions"],
        "unresolved_property_observations": values["unresolved_property_observations"],
        "property_series": values["property_series"],
        "characterizations": values["characterizations"],
        "evidence": registry.items,
        "provenance": provenance,
        "warnings": warnings,
        "stage_failures": copy.deepcopy(failures),
        "raw_stage_candidates": {
            stage_name: copy.deepcopy(stages[stage_name])
            for stage_name in candidate_stages
            if stage_name in stages
        },
        "validation_summary": {
            "status": "not_validated",
            "error_count": 0,
            "warning_count": len(warnings),
        },
        "source_stage_versions": {
            stage_name: (stages.get(stage_name) or {}).get("schema_version")
            for stage_name in STAGE_FILES
        },
    }
    sensitive_issues: list[Any] = []
    _scan_sensitive(payload, sensitive_issues)
    if sensitive_issues:
        raise CandidatePublishError("候选结果包含敏感字段或疑似密钥，拒绝发布")
    return payload


def publish_candidate(
    ref_no: str,
    *,
    input_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    input_dir = input_root / ref_no
    # 输入目录不存在时必须直接失败，不能建输出目录、也不能写空 candidate。
    # 缺**某几个 Stage 文件**是正常的（走 candidate_partial），但整个目录都
    # 不在，说明 --ref-no 传错了——最常见的是漏了 reference_no_ 前缀。
    # 这种情况以前会静默产出 0 条 observation 的 candidate，比报错更危险。
    if not input_dir.is_dir():
        hint = ""
        if not ref_no.startswith("reference_no_"):
            prefixed = input_root / f"reference_no_{ref_no}"
            if prefixed.is_dir():
                hint = f"；--ref-no 需要完整目录名，请改用 reference_no_{ref_no}"
        raise CandidatePublishError(
            f"输入目录不存在：{input_dir}{hint}"
        )
    stages, stage_states, failures = load_candidate_sources(ref_no, input_dir)
    candidate = build_candidate_payload(
        ref_no,
        stages,
        stage_states=stage_states,
        failures=failures,
    )
    output_dir = output_root / ref_no
    candidate_path = output_dir / "candidate.json"
    report_path = output_dir / "report_candidate.html"
    write_json_atomic(candidate_path, candidate)
    render_extraction_html(
        candidate,
        report_path,
        stage0_data=stages.get("stage0"),
        project_root=TESTCODE_ROOT.parent,
    )
    return candidate_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布未经完整语义校验的候选抽取结果")
    parser.add_argument("--ref-no", required=True)
    parser.add_argument("--config", type=Path, help="与其他 Stage 保持一致；候选发布不读取模型配置")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="兼容批处理器；候选文件始终原子覆盖")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_path, report_path = publish_candidate(
        args.ref_no,
        input_root=args.input_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(f"[done] {args.ref_no} -> {candidate_path}")
    print(f"[done] {args.ref_no} -> {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
