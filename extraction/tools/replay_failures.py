"""离线回放全部 failure JSON，判定哪些失败在当前代码下依然存活。

只读语义：所有回放写入独立的 scratch 输出根目录，不触碰 output/ 与
output_batch_16/ 的既有产物；不发起任何模型调用（stage 1-5 使用各自的
--replay-failure 通道，从 failure JSON 内保存的 raw_response 重放）。

默认跳过已有本阶段成功产物的历史 failure；使用 --include-resolved 可纳入。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXTRACTION_ROOT = Path(__file__).resolve().parent.parent
REPLAYABLE_STAGES = {
    "stage1": EXTRACTION_ROOT / "stages" / "stage1_material_mention.py",
    "stage2": EXTRACTION_ROOT / "stages" / "stage2_polymer_entity.py",
    "stage3": EXTRACTION_ROOT / "stages" / "stage3_sample_process.py",
    "stage4": EXTRACTION_ROOT / "stages" / "stage4_property.py",
    "stage5": EXTRACTION_ROOT / "stages" / "stage5_characterization.py",
}
SUCCESS_ARTIFACT = {
    "stage1": "stage1_mentions.json",
    "stage2": "stage2_entities.json",
    "stage3": "stage3_process.json",
    "stage4": "stage4_properties.json",
    "stage5": "stage5_characterizations.json",
}


@dataclass
class ReplayCase:
    source_root: str
    ref_no: str
    stage: str
    failure_path: str


@dataclass
class ReplayResult:
    source_root: str
    ref_no: str
    stage: str
    original_error: str
    original_error_type: str
    replayable: bool
    outcome: str          # resolved | survived | unreplayable | harness_error
    new_error: str | None
    exit_code: int | None
    has_raw_content: bool
    needed_unfence: bool = False


def discover_cases(
    roots: list[Path],
    *,
    include_resolved: bool = False,
) -> list[ReplayCase]:
    cases: list[ReplayCase] = []
    for root in roots:
        if not root.is_dir():
            continue
        for failure_path in sorted(root.glob("reference_no_*/stage*_failure.json")):
            stage = failure_path.name.split("_", 1)[0]
            success_name = SUCCESS_ARTIFACT.get(stage)
            if (
                not include_resolved
                and success_name is not None
                and (failure_path.parent / success_name).is_file()
            ):
                continue
            cases.append(ReplayCase(
                source_root=str(root),
                ref_no=failure_path.parent.name,
                stage=stage,
                failure_path=str(failure_path),
            ))
    return cases


def _read_failure(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _strip_code_fence(text: str) -> str:
    """与 llm_client.extract_json_object 一致地剥离 markdown 代码围栏。

    仅用于本工具的静态判定（区分"带围栏"与"真截断"）。
    stage 侧已复用 extract_json_object，回放本身不再依赖本函数预处理。
    """
    stripped = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else stripped


def _parse_raw_content(text: str) -> tuple[dict[str, Any] | None, bool]:
    """返回 (解析结果, 是否需要剥离围栏才能解析)。"""
    try:
        data = json.loads(text)
        return (data if isinstance(data, dict) else None), False
    except json.JSONDecodeError:
        pass
    unfenced = _strip_code_fence(text)
    if unfenced == text.strip():
        return None, False
    try:
        data = json.loads(unfenced)
        return (data if isinstance(data, dict) else None), True
    except json.JSONDecodeError:
        return None, True


def replay_one(
    case: ReplayCase,
    scratch_root: Path,
    config_path: Path,
) -> ReplayResult:
    failure = _read_failure(Path(case.failure_path))
    raw = failure.get("raw_response")
    raw_content = raw.get("content") if isinstance(raw, dict) else None
    has_raw = isinstance(raw_content, str) and bool(raw_content.strip())
    parsed_raw: dict[str, Any] | None = None
    needed_unfence = False
    if has_raw:
        parsed_raw, needed_unfence = _parse_raw_content(raw_content)
    original_error = str(failure.get("error", ""))
    original_error_type = str(failure.get("error_type", ""))

    base = ReplayResult(
        source_root=case.source_root,
        ref_no=case.ref_no,
        stage=case.stage,
        original_error=original_error,
        original_error_type=original_error_type,
        replayable=case.stage in REPLAYABLE_STAGES and parsed_raw is not None,
        outcome="unreplayable",
        new_error=None,
        exit_code=None,
        has_raw_content=has_raw,
        needed_unfence=needed_unfence,
    )

    if case.stage not in REPLAYABLE_STAGES:
        base.new_error = f"{case.stage} 无 --replay-failure 通道"
        return base
    if not has_raw:
        base.new_error = "failure JSON 未保存 raw_response.content"
        return base
    if parsed_raw is None:
        base.new_error = (
            "raw_response.content 剥离围栏后仍不是完整 JSON（疑似真实截断）"
            if needed_unfence
            else "raw_response.content 不是完整 JSON（疑似真实截断）"
        )
        return base

    # 复制到独立 scratch 目录，避免污染既有产物
    tag = Path(case.source_root).name
    work_root = scratch_root / tag
    work_ref = work_root / case.ref_no
    if work_ref.exists():
        shutil.rmtree(work_ref)
    work_ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(case.source_root) / case.ref_no, work_ref)

    # 带围栏的 raw content 现在由 stage 侧的 extract_json_object 处理，
    # 工具不再改写 scratch 副本——否则测不到真实回放路径。
    # needed_unfence 仅作为归因信息保留在报告中。

    # 移除本阶段的既有成功产物，确保回放真正重新校验
    artifact = work_ref / SUCCESS_ARTIFACT[case.stage]
    if artifact.exists():
        artifact.unlink()

    cmd = [
        sys.executable,
        str(REPLAYABLE_STAGES[case.stage]),
        "--ref-no", case.ref_no,
        "--config", str(config_path),
        "--input-root", str(work_root),
        "--output-root", str(work_root),
        "--replay-failure",
        "--force",
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(EXTRACTION_ROOT),
        env=env,
    )
    base.exit_code = proc.returncode

    produced = (work_ref / SUCCESS_ARTIFACT[case.stage]).exists()
    if produced:
        base.outcome = "resolved"
        return base

    # 注意：各 stage 在回放失败时不会重写 *_failure.json（保留原始输入），
    # 且失败时进程仍返回 0。因此新错误只能从 stdout 的 [failed] 行解析，
    # 不能读取 failure JSON——那会读回陈旧的原始错误。
    new_error = ""
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[failed]"):
            new_error = stripped[len("[failed]"):].strip()
            break
    if not new_error:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        new_error = tail[-1] if tail else "(无错误输出)"
    base.new_error = new_error
    base.outcome = "survived"
    return base


def main() -> int:
    # Windows 控制台默认 GBK，回放报错含中文会导致 print 崩溃
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            EXTRACTION_ROOT / "output_batch_16",
            EXTRACTION_ROOT / "output" / "_series_preview",
        ],
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=EXTRACTION_ROOT / "output" / "_replay_scratch",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXTRACTION_ROOT / "config" / "pipeline.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EXTRACTION_ROOT / "output" / "_replay_scratch" / "replay_report.json",
    )
    parser.add_argument(
        "--include-resolved",
        action="store_true",
        help="同时回放已有成功产物的历史 failure",
    )
    args = parser.parse_args()

    cases = discover_cases(
        [r.resolve() for r in args.roots],
        include_resolved=args.include_resolved,
    )
    print(f"发现 {len(cases)} 个 failure 案例\n")

    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    results: list[ReplayResult] = []
    for index, case in enumerate(cases, 1):
        label = f"[{index}/{len(cases)}] {case.ref_no} {case.stage}"
        print(f"{label} ...", flush=True)
        result = replay_one(case, scratch, args.config.resolve())
        results.append(result)
        print(f"    -> {result.outcome}", flush=True)
        if result.outcome == "survived" and result.new_error:
            print(f"       {result.new_error[:160]}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 汇总 =====")
    for outcome in ("resolved", "survived", "unreplayable", "harness_error"):
        subset = [r for r in results if r.outcome == outcome]
        print(f"{outcome:14s}: {len(subset)}")
    print(f"\n报告已写入：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
