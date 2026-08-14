"""Validate immutable published batch-result collections before Git upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


BATCH_NAME_RE = re.compile(r"^[a-z0-9._-]+_(\d{8})$")
REF_NO_RE = re.compile(r"^reference_no_\d{7}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/[^/]+/)")
SECRET_RE = re.compile(
    r"(?:github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+|sk-[A-Za-z0-9]{16,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)
PROHIBITED_SUFFIXES = {".log", ".sqlite", ".sqlite3", ".db", ".pem", ".key"}
PROHIBITED_NAMES = {".env", "progress_state.json", "run_manifest.json"}
MAX_FILE_BYTES = 95 * 1024 * 1024
REQUIRED_CANDIDATE_KEYS = {
    "paper",
    "polymer_entities",
    "samples",
    "process_steps",
    "property_observations",
    "evidence",
    "publication",
}


class PublishedBatchError(RuntimeError):
    """A published collection violates the public batch contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishedBatchError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise PublishedBatchError(f"JSON 顶层不是 object：{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(items: Any, field: str, label: str) -> set[str]:
    values = items if isinstance(items, list) else []
    result = {
        str(item[field])
        for item in values
        if isinstance(item, dict) and item.get(field)
    }
    if len(result) != len(values):
        raise PublishedBatchError(f"{label} ID 缺失或重复")
    return result


def _require_subset(values: Iterable[Any], allowed: set[str], label: str) -> None:
    invalid = sorted({str(value) for value in values if str(value) not in allowed})
    if invalid:
        raise PublishedBatchError(f"{label} 包含悬空引用：{invalid[:5]}")


def _walk_evidence_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        evidence_ids = value.get("evidence_ids")
        if isinstance(evidence_ids, list):
            yield from (str(item) for item in evidence_ids)
        for key, child in value.items():
            if key != "evidence_ids":
                yield from _walk_evidence_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_evidence_ids(child)


def validate_candidate(candidate: dict[str, Any], ref_no: str) -> None:
    if candidate.get("document_id") != ref_no:
        raise PublishedBatchError(f"{ref_no} candidate.document_id 不一致")
    missing = sorted(REQUIRED_CANDIDATE_KEYS - set(candidate))
    if missing:
        raise PublishedBatchError(f"{ref_no} candidate 缺字段：{missing}")
    publication = candidate.get("publication") or {}
    if publication.get("status") != "complete":
        raise PublishedBatchError(f"{ref_no} publication.status 不是 complete")

    entity_ids = _ids(candidate.get("polymer_entities"), "entity_id", "PolymerEntity")
    sample_ids = _ids(candidate.get("samples"), "sample_id", "Sample")
    evidence_ids = _ids(candidate.get("evidence"), "evidence_id", "Evidence")

    for sample in candidate.get("samples") or []:
        entity_id = sample.get("refers_to_entity")
        if entity_id is not None:
            _require_subset([entity_id], entity_ids, "Sample.refers_to_entity")
    for step in candidate.get("process_steps") or []:
        _require_subset(
            [*(step.get("input_sample_ids") or []), *(step.get("output_sample_ids") or [])],
            sample_ids,
            "ProcessStep sample",
        )
    for item in candidate.get("property_observations") or []:
        sample_id = item.get("sample_id")
        if sample_id is not None:
            _require_subset([sample_id], sample_ids, "Property.sample_id")
    for item in candidate.get("characterizations") or []:
        _require_subset(item.get("sample_ids") or [], sample_ids, "Characterization.sample_ids")
        _require_subset(item.get("entity_ids") or [], entity_ids, "Characterization.entity_ids")
    _require_subset(_walk_evidence_ids(candidate), evidence_ids, "evidence_ids")


def validate_collection(collection_dir: Path) -> dict[str, Any]:
    collection_dir = collection_dir.resolve()
    match = BATCH_NAME_RE.fullmatch(collection_dir.name)
    if match is None:
        raise PublishedBatchError(f"批次目录名不合规：{collection_dir.name}")
    index = _load_json(collection_dir / "RESULT_INDEX.json")
    if index.get("schema_version") != "polymerlit-batch/2.0":
        raise PublishedBatchError("RESULT_INDEX schema_version 必须为 polymerlit-batch/2.0")
    expected_date = f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
    if index.get("result_date") != expected_date:
        raise PublishedBatchError("result_date 与目录日期不一致")
    commit = str((index.get("pipeline") or {}).get("git_commit") or "")
    if re.fullmatch(r"[a-f0-9]{40}", commit) is None:
        raise PublishedBatchError("pipeline.git_commit 必须为 40 位 SHA")

    documents = index.get("documents")
    if not isinstance(documents, list) or not documents:
        raise PublishedBatchError("RESULT_INDEX.documents 不能为空")
    refs: set[str] = set()
    file_count = 0
    for document in documents:
        ref_no = str(document.get("reference_no") or "")
        if REF_NO_RE.fullmatch(ref_no) is None or ref_no in refs:
            raise PublishedBatchError(f"无效或重复 reference_no：{ref_no}")
        refs.add(ref_no)
        if document.get("result_dir") != ref_no:
            raise PublishedBatchError(f"{ref_no} result_dir 不一致")
        document_dir = collection_dir / ref_no
        candidate_path = document_dir / "candidate.json"
        report_path = document_dir / "report_candidate.html"
        if not candidate_path.is_file() or not report_path.is_file() or not report_path.stat().st_size:
            raise PublishedBatchError(f"{ref_no} 缺 candidate 或候选报告")
        validate_candidate(_load_json(candidate_path), ref_no)

        manifest_files = document.get("files")
        if not isinstance(manifest_files, list):
            raise PublishedBatchError(f"{ref_no} files 不是列表")
        manifest_names: set[str] = set()
        for item in manifest_files:
            name = str(item.get("name") or "")
            path = document_dir / name
            if not name or Path(name).name != name or name in manifest_names:
                raise PublishedBatchError(f"{ref_no} 文件名无效或重复：{name}")
            manifest_names.add(name)
            if not path.is_file():
                raise PublishedBatchError(f"{ref_no} 清单文件不存在：{name}")
            if path.stat().st_size != item.get("size_bytes"):
                raise PublishedBatchError(f"{ref_no}/{name} size 不一致")
            checksum = str(item.get("sha256") or "")
            if SHA256_RE.fullmatch(checksum) is None or _sha256(path) != checksum:
                raise PublishedBatchError(f"{ref_no}/{name} sha256 不一致")
            file_count += 1
        actual_names = {path.name for path in document_dir.iterdir() if path.is_file()}
        if actual_names != manifest_names:
            raise PublishedBatchError(f"{ref_no} 文件清单与目录不一致")

    for path in collection_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.stat().st_size >= MAX_FILE_BYTES:
            raise PublishedBatchError(f"文件达到 95 MB 发布上限：{path}")
        if path.name in PROHIBITED_NAMES or path.suffix.lower() in PROHIBITED_SUFFIXES:
            raise PublishedBatchError(f"批次包含禁止文件：{path}")
        if path.suffix.lower() in {".json", ".html", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if ABSOLUTE_PATH_RE.search(text):
                raise PublishedBatchError(f"批次包含本机绝对路径：{path}")
            if SECRET_RE.search(text):
                raise PublishedBatchError(f"批次包含疑似密钥：{path}")
    return {"collection": collection_dir.name, "documents": len(refs), "files": file_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验 polymerlit-batch/2.0 发布批次")
    parser.add_argument("collection", type=Path)
    return parser


def main() -> int:
    result = validate_collection(build_parser().parse_args().collection)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
