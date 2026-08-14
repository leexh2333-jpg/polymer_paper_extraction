import hashlib
import json
import sys
from pathlib import Path

import pytest


PREVIEW_ROOT = Path(__file__).resolve().parents[1]
if str(PREVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(PREVIEW_ROOT))

from validate_published_batches import PublishedBatchError, validate_collection


def _write_collection(root: Path) -> Path:
    collection = root / "demo_types_20260814"
    document = collection / "reference_no_0000001"
    document.mkdir(parents=True)
    candidate = {
        "document_id": "reference_no_0000001",
        "publication": {"status": "complete"},
        "paper": {},
        "polymer_entities": [{"entity_id": "pe001", "evidence_ids": ["ev001"]}],
        "samples": [{"sample_id": "s001", "refers_to_entity": "pe001"}],
        "process_steps": [],
        "property_observations": [],
        "characterizations": [],
        "evidence": [{"evidence_id": "ev001"}],
    }
    candidate_path = document / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    report_path = document / "report_candidate.html"
    report_path.write_text("<html>ok</html>", encoding="utf-8")

    files = []
    for path in (candidate_path, report_path):
        files.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    index = {
        "schema_version": "polymerlit-batch/2.0",
        "result_date": "2026-08-14",
        "pipeline": {"git_commit": "a" * 40},
        "documents": [{
            "reference_no": "reference_no_0000001",
            "result_dir": "reference_no_0000001",
            "files": files,
        }],
    }
    (collection / "RESULT_INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    return collection


def test_valid_collection_passes(tmp_path: Path) -> None:
    assert validate_collection(_write_collection(tmp_path))["documents"] == 1


def test_dangling_entity_reference_fails(tmp_path: Path) -> None:
    collection = _write_collection(tmp_path)
    candidate_path = collection / "reference_no_0000001" / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["samples"][0]["refers_to_entity"] = "pe999"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    index_path = collection / "RESULT_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    item = index["documents"][0]["files"][0]
    item["size_bytes"] = candidate_path.stat().st_size
    item["sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(PublishedBatchError, match="悬空引用"):
        validate_collection(collection)
