"""严格验收 Preview 候选批次，只有全部文献 complete 时返回成功。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


PREVIEW_ROOT = Path(__file__).resolve().parent
DEFAULT_REF_LIST = PREVIEW_ROOT / "demo_latest_20_refs.txt"
DEFAULT_OUTPUT_DIR = PREVIEW_ROOT.parent / "extraction" / "output_test"
STAGE_FILES = (
    "stage0_blocks.json",
    "stage1_mentions.json",
    "stage2_entities.json",
    "stage3_process.json",
    "stage4_properties.json",
    "stage5_characterizations.json",
)
STAGE_FAILURE_FILES = {
    "stage0_blocks.json": "stage0_failure.json",
    "stage1_mentions.json": "stage1_failure.json",
    "stage2_entities.json": "stage2_failure.json",
    "stage3_process.json": "stage3_failure.json",
    "stage4_properties.json": "stage4_failure.json",
    "stage5_characterizations.json": "stage5_failure.json",
}
VerificationMode = Literal["preview", "strict"]


STRUCTURED_RESULT_FIELDS = (
    "material_mentions",
    "polymer_entities",
    "samples",
    "process_steps",
    "measurement_conditions",
    "properties",
    "unresolved_property_observations",
    "property_series",
    "characterizations",
)


class VerificationError(RuntimeError):
    """验收参数或清单错误。"""


@dataclass(frozen=True)
class DocumentVerification:
    ref_no: str
    passed: bool
    issues: list[str]
    historical_failures: list[str]


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise VerificationError(f"无法读取 {label}：{path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{label} 不是合法 JSON：{path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"{label} 顶层必须是对象：{path}")
    return payload


def load_ref_list(path: Path) -> list[str]:
    if not path.is_file():
        raise VerificationError(f"文献清单不存在：{path}")
    refs: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        if value.endswith("_document.json"):
            value = value.removesuffix("_document.json")
        refs.append(value)
    refs = list(dict.fromkeys(refs))
    if not refs:
        raise VerificationError(f"文献清单为空：{path}")
    return refs


def _document_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("document_id")
    return str(value) if value is not None else None


def verify_document(
    ref_no: str,
    output_root: Path,
    *,
    mode: VerificationMode = "strict",
) -> DocumentVerification:
    document_dir = output_root / ref_no
    issues: list[str] = []
    historical_failures: list[str] = []
    stages: dict[str, dict[str, Any]] = {}

    if not document_dir.is_dir():
        return DocumentVerification(
            ref_no=ref_no,
            passed=False,
            issues=[f"文献输出目录不存在：{document_dir}"],
            historical_failures=[],
        )

    for filename in STAGE_FILES:
        path = document_dir / filename
        if not path.is_file():
            issues.append(f"缺少 {filename}")
            continue
        try:
            payload = _load_json_object(path, filename)
        except VerificationError as exc:
            issues.append(str(exc))
            continue
        stages[filename] = payload
        document_id = _document_id(payload)
        if document_id != ref_no:
            issues.append(
                f"{filename} document_id 不匹配：期望 {ref_no}，实际 {document_id}"
            )
        failure_name = STAGE_FAILURE_FILES[filename]
        if (document_dir / failure_name).is_file():
            historical_failures.append(failure_name)

    candidate_path = document_dir / "candidate.json"
    candidate: dict[str, Any] | None = None
    if not candidate_path.is_file():
        issues.append("缺少 candidate.json")
    else:
        try:
            candidate = _load_json_object(candidate_path, "candidate.json")
        except VerificationError as exc:
            issues.append(str(exc))

    if candidate is not None:
        candidate_document_id = _document_id(candidate)
        if candidate_document_id != ref_no:
            issues.append(
                "candidate.json document_id 不匹配："
                f"期望 {ref_no}，实际 {candidate_document_id}"
            )
        if mode == "strict":
            publication = candidate.get("publication")
            if not isinstance(publication, dict):
                issues.append("candidate.publication 缺失或不是对象")
            elif publication.get("status") != "complete":
                issues.append(
                    "candidate.publication.status 不是 complete："
                    f"{publication.get('status')}"
                )
            stage_failures = candidate.get("stage_failures")
            if stage_failures != []:
                issues.append("candidate.stage_failures 必须是空列表")
            if not any(
                isinstance(candidate.get(field), list) and bool(candidate[field])
                for field in STRUCTURED_RESULT_FIELDS
            ):
                issues.append("candidate 未包含任何非空结构化抽取结果")

    html_path = document_dir / "report_candidate.html"
    if not html_path.is_file():
        issues.append("缺少 report_candidate.html")
    elif html_path.stat().st_size == 0:
        issues.append("report_candidate.html 为空")

    return DocumentVerification(
        ref_no=ref_no,
        passed=not issues,
        issues=issues,
        historical_failures=historical_failures,
    )


def verify_batch(
    ref_nos: Sequence[str],
    output_root: Path,
    *,
    expected_count: int | None = None,
    mode: VerificationMode = "strict",
) -> dict[str, Any]:
    refs = list(dict.fromkeys(ref_nos))
    count_issues: list[str] = []
    if expected_count is not None and len(refs) != expected_count:
        count_issues.append(
            f"文献清单数量不匹配：期望 {expected_count}，实际 {len(refs)}"
        )
    results = [
        verify_document(ref_no, output_root, mode=mode)
        for ref_no in refs
    ]
    passed_count = sum(result.passed for result in results)
    failed_count = len(results) - passed_count
    accepted = not count_issues and failed_count == 0
    return {
        "accepted": accepted,
        "mode": mode,
        "output_root": str(output_root),
        "expected_count": expected_count,
        "selected_count": len(refs),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "count_issues": count_issues,
        "documents": [asdict(result) for result in results],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分层验收固定演示集的 Stage 0-5、candidate 和 HTML",
    )
    parser.add_argument("--ref-list", type=Path, default=DEFAULT_REF_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("preview", "strict"),
        default="strict",
        help="preview 只验流程产物；strict 继续验候选完整合规",
    )
    parser.add_argument("--report-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        refs = load_ref_list(args.ref_list.expanduser().resolve())
        report = verify_batch(
            refs,
            args.output_dir.expanduser().resolve(),
            expected_count=args.expected_count,
            mode=args.mode,
        )
        if args.report_out:
            report_path = args.report_out.expanduser().resolve()
            write_json_atomic(report_path, report)
            print(f"验收报告：{report_path}")
        for document in report["documents"]:
            state = "PASS" if document["passed"] else "FAIL"
            print(f"[{state}] {document['ref_no']}")
            for issue in document["issues"]:
                print(f"  - {issue}")
            if document["historical_failures"]:
                print(
                    "  - 历史 failure："
                    + ", ".join(document["historical_failures"])
                )
        for issue in report["count_issues"]:
            print(f"[FAIL] {issue}")
        print(
            "验收汇总："
            f"mode={report['mode']}，"
            f"selected={report['selected_count']}，"
            f"passed={report['passed_count']}，"
            f"failed={report['failed_count']}，"
            f"accepted={str(report['accepted']).lower()}"
        )
        return 0 if report["accepted"] else 1
    except VerificationError as exc:
        print(f"验收失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
