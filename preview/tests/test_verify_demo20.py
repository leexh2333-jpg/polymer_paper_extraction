import json
import sys
from pathlib import Path


PREVIEW_ROOT = Path(__file__).resolve().parents[1]
if str(PREVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(PREVIEW_ROOT))

from verify_demo20 import (  # noqa: E402
    STAGE_FILES,
    load_ref_list,
    main,
    verify_batch,
    verify_document,
)


def _write_complete_candidate(root: Path, ref_no: str) -> None:
    document_dir = root / ref_no
    document_dir.mkdir(parents=True)
    for filename in STAGE_FILES:
        payload = {"document_id": ref_no}
        if filename == "stage1_mentions.json":
            payload["material_mentions"] = [{"mention_id": "m001"}]
        (document_dir / filename).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    candidate = {
        "document_id": ref_no,
        "publication": {"status": "complete"},
        "stage_failures": [],
        "material_mentions": [{"mention_id": "m001"}],
    }
    (document_dir / "candidate.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    (document_dir / "report_candidate.html").write_text(
        "<html>candidate</html>", encoding="utf-8"
    )


def test_load_ref_list_ignores_comments_and_deduplicates(tmp_path: Path) -> None:
    ref_list = tmp_path / "refs.txt"
    ref_list.write_text(
        "# demo\nreference_no_0000001\n\n"
        "reference_no_0000001_document.json\nreference_no_0000002\n",
        encoding="utf-8",
    )

    assert load_ref_list(ref_list) == [
        "reference_no_0000001",
        "reference_no_0000002",
    ]


def test_complete_document_passes_and_stale_failure_is_historical(
    tmp_path: Path,
) -> None:
    ref_no = "reference_no_0000001"
    _write_complete_candidate(tmp_path, ref_no)
    (tmp_path / ref_no / "stage4_failure.json").write_text(
        "{}", encoding="utf-8"
    )

    result = verify_document(ref_no, tmp_path)

    assert result.passed is True
    assert result.issues == []
    assert result.historical_failures == ["stage4_failure.json"]


def test_partial_candidate_fails(tmp_path: Path) -> None:
    ref_no = "reference_no_0000001"
    _write_complete_candidate(tmp_path, ref_no)
    candidate_path = tmp_path / ref_no / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["publication"]["status"] = "partial"
    candidate["stage_failures"] = [{"stage": "stage4"}]
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = verify_document(ref_no, tmp_path)

    assert result.passed is False
    assert any("不是 complete" in issue for issue in result.issues)
    assert any("必须是空列表" in issue for issue in result.issues)


def test_preview_mode_accepts_partial_degraded_candidate(tmp_path: Path) -> None:
    ref_no = "reference_no_0000001"
    _write_complete_candidate(tmp_path, ref_no)
    candidate_path = tmp_path / ref_no / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["publication"]["status"] = "partial"
    candidate["stage_failures"] = [{"stage": "stage4"}]
    candidate["material_mentions"] = []
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = verify_document(ref_no, tmp_path, mode="preview")

    assert result.passed is True
    assert result.issues == []


def test_preview_batch_reports_mode(tmp_path: Path) -> None:
    ref_no = "reference_no_0000001"
    _write_complete_candidate(tmp_path, ref_no)

    report = verify_batch(
        [ref_no],
        tmp_path,
        expected_count=1,
        mode="preview",
    )

    assert report["accepted"] is True
    assert report["mode"] == "preview"


def test_empty_stage5_is_allowed_when_other_results_exist(tmp_path: Path) -> None:
    ref_no = "reference_no_0042246"
    _write_complete_candidate(tmp_path, ref_no)
    stage5_path = tmp_path / ref_no / "stage5_characterizations.json"
    stage5_path.write_text(
        json.dumps({"document_id": ref_no, "characterizations": []}),
        encoding="utf-8",
    )

    assert verify_document(ref_no, tmp_path).passed is True


def test_batch_rejects_wrong_expected_count(tmp_path: Path) -> None:
    ref_no = "reference_no_0000001"
    _write_complete_candidate(tmp_path, ref_no)

    report = verify_batch([ref_no], tmp_path, expected_count=20)

    assert report["accepted"] is False
    assert report["passed_count"] == 1
    assert report["count_issues"]


def test_main_returns_nonzero_for_empty_html(tmp_path: Path) -> None:
    ref_no = "reference_no_0000001"
    _write_complete_candidate(tmp_path, ref_no)
    (tmp_path / ref_no / "report_candidate.html").write_text("", encoding="utf-8")
    ref_list = tmp_path / "refs.txt"
    ref_list.write_text(ref_no + "\n", encoding="utf-8")

    exit_code = main([
        "--ref-list", str(ref_list),
        "--output-dir", str(tmp_path),
        "--expected-count", "1",
    ])

    assert exit_code == 1
