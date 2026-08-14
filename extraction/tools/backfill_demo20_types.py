"""Backfill demo20 type fields without changing entity/sample granularity.

This recovery path preserves the successful Stage 0-5 extraction graph, applies
the current deterministic Stage 2/3 type policies, and records source/output
hashes in a manifest.  It does not claim that the LLM stages were rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


EXTRACTION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTRACTION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_ROOT))

from schema.polymer_schema import Sample, Stage2Document, Stage3Document
from stages.stage2_polymer_entity import (
    IMPLEMENTATION_VERSION as STAGE2_IMPLEMENTATION_VERSION,
    _apply_polymer_type_policy,
    _stage2_output_payload,
)
from stages.stage3_sample_process import (
    IMPLEMENTATION_VERSION as STAGE3_IMPLEMENTATION_VERSION,
    _apply_material_type_policy,
    _apply_process_polymer_type_policy,
    _stage3_output_payload,
)
from tools.evaluate_demo20_types import DEFAULT_REF_LIST, load_ref_nos


class BackfillError(RuntimeError):
    """The requested backfill cannot be performed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def align_sample_polymer_types(
    samples: list[Sample],
    entities: Stage2Document,
) -> tuple[list[Sample], list[dict[str, Any]]]:
    entity_map = {entity.entity_id: entity for entity in entities.polymer_entities}
    aligned: list[Sample] = []
    items: list[dict[str, Any]] = []
    for sample in samples:
        entity = entity_map.get(sample.refers_to_entity or "")
        if entity is None or entity.polymer_type is None:
            aligned.append(sample)
            continue
        polymer_type = entity.polymer_type
        copolymer_type = (
            entity.copolymer_type if polymer_type == "copolymer" else None
        )
        if (
            sample.polymer_type != polymer_type
            or sample.copolymer_type != copolymer_type
        ):
            items.append({
                "sample_id": sample.sample_id,
                "entity_id": entity.entity_id,
                "fields": ["polymer_type", "copolymer_type"],
                "polymer_type": polymer_type,
                "copolymer_type": copolymer_type,
            })
            sample = sample.model_copy(update={
                "polymer_type": polymer_type,
                "copolymer_type": copolymer_type,
            })
        aligned.append(sample)
    return aligned, items


def _append_warning(
    warnings: list[dict[str, Any]],
    *,
    stage: str,
    code: str,
    message: str,
    items: list[dict[str, Any]],
) -> None:
    if items:
        warnings.append({
            "stage": stage,
            "code": code,
            "message": message,
            "items": items,
            "backfill": True,
        })


def backfill_document(source: Path, destination: Path) -> dict[str, Any]:
    stage2_path = source / "stage2_entities.json"
    stage3_path = source / "stage3_process.json"
    if not stage2_path.is_file() or not stage3_path.is_file():
        raise BackfillError(f"缺少 Stage2/3 成功产物：{source}")

    shutil.copytree(source, destination)
    stage2 = Stage2Document.model_validate_json(
        stage2_path.read_text(encoding="utf-8-sig")
    )
    entities, polymer_defaults, polymer_repairs = _apply_polymer_type_policy(
        stage2.polymer_entities
    )
    stage2_warnings = list(stage2.warnings)
    _append_warning(
        stage2_warnings,
        stage="stage2_polymer_entity",
        code="polymer_type_default_inferred",
        message="Backfill：无共聚或共混反证的实体已推断为 homopolymer",
        items=polymer_defaults,
    )
    _append_warning(
        stage2_warnings,
        stage="stage2_polymer_entity",
        code="polymer_type_negative_rule_applied",
        message="Backfill：已修复区域规整性/交替结构或材料配方类型误判",
        items=polymer_repairs,
    )
    stage2 = stage2.model_copy(update={
        "polymer_entities": entities,
        "warnings": stage2_warnings,
    })
    destination_stage2 = destination / "stage2_entities.json"
    _write_json_atomic(destination_stage2, _stage2_output_payload(stage2))

    stage3 = Stage3Document.model_validate_json(
        stage3_path.read_text(encoding="utf-8-sig")
    )
    aligned_samples, aligned_items = align_sample_polymer_types(
        stage3.samples, stage2
    )
    aligned_samples, process_polymer_items = _apply_process_polymer_type_policy(
        aligned_samples, stage3.process_steps
    )
    (
        samples,
        evidence_items,
        inheritance_items,
        material_defaults,
    ) = _apply_material_type_policy(aligned_samples, stage3.process_steps)
    stage3_warnings = list(stage3.warnings)
    _append_warning(
        stage3_warnings,
        stage="stage3_sample_process",
        code="sample_polymer_type_backfilled_from_entity",
        message="Backfill：Sample 已采用关联 PolymerEntity 的类型",
        items=aligned_items,
    )
    _append_warning(
        stage3_warnings,
        stage="stage3_sample_process",
        code="sample_polymer_type_process_inferred",
        message="Backfill：已按配方输入或成分保持工艺补全 polymer_type",
        items=process_polymer_items,
    )
    _append_warning(
        stage3_warnings,
        stage="stage3_sample_process",
        code="material_type_evidence_inferred",
        message="Backfill：已按明确配方或增强组分证据补全 material_type",
        items=evidence_items,
    )
    _append_warning(
        stage3_warnings,
        stage="stage3_sample_process",
        code="material_type_process_inherited",
        message="Backfill：成分保持工艺输出已继承一致输入 material_type",
        items=inheritance_items,
    )
    _append_warning(
        stage3_warnings,
        stage="stage3_sample_process",
        code="material_type_default_inferred",
        message="Backfill：无第二组分反证的样品已推断为 neat_resin",
        items=material_defaults,
    )
    stage3 = stage3.model_copy(update={
        "samples": samples,
        "warnings": stage3_warnings,
    })
    destination_stage3 = destination / "stage3_process.json"
    _write_json_atomic(destination_stage3, _stage3_output_payload(stage3))

    return {
        "document_id": stage2.document_id,
        "source": {
            "stage2_path": str(stage2_path.resolve()),
            "stage2_sha256": _sha256(stage2_path),
            "stage3_path": str(stage3_path.resolve()),
            "stage3_sha256": _sha256(stage3_path),
        },
        "output": {
            "stage2_path": str(destination_stage2.resolve()),
            "stage2_sha256": _sha256(destination_stage2),
            "stage3_path": str(destination_stage3.resolve()),
            "stage3_sha256": _sha256(destination_stage3),
        },
        "changes": {
            "polymer_type_defaults": len(polymer_defaults),
            "polymer_type_negative_repairs": len(polymer_repairs),
            "sample_polymer_type_alignments": len(aligned_items),
            "sample_polymer_type_process_inferences": len(process_polymer_items),
            "material_type_evidence_inferences": len(evidence_items),
            "material_type_process_inheritances": len(inheritance_items),
            "material_type_defaults": len(material_defaults),
        },
    }


def run_backfill(
    source_dir: Path,
    output_dir: Path,
    ref_list: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise BackfillError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    refs = load_ref_nos(ref_list)
    output_dir.mkdir(parents=True)
    documents = []
    for ref_no in refs:
        source = source_dir / ref_no
        if not source.is_dir():
            raise BackfillError(f"缺少源文档目录：{source}")
        documents.append(backfill_document(source, output_dir / ref_no))
    manifest = {
        "schema_version": "demo20_type_backfill.v1",
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "ref_list_path": str(ref_list.resolve()),
        "ref_list_sha256": _sha256(ref_list),
        "policy_versions": {
            "stage2": STAGE2_IMPLEMENTATION_VERSION,
            "stage3": STAGE3_IMPLEMENTATION_VERSION,
        },
        "llm_rerun": False,
        "documents": documents,
    }
    _write_json_atomic(output_dir / "type_backfill_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 demo20 稳定产物回填类型字段")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref-list", type=Path, default=DEFAULT_REF_LIST)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_backfill(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.ref_list.resolve(),
    )
    print(
        f"已回填 {len(manifest['documents'])} 篇文档："
        f"{manifest['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
