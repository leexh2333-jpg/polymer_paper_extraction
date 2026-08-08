"""从 PDF 到结构化报告的统一流水线入口。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
OCR_ROOT = CODE_ROOT / "ocr"
EXTRACTION_ROOT = CODE_ROOT / "extraction"
MINERU_SCRIPT = OCR_ROOT / "mineru_batch_parse.py"
BATCH_SCRIPT = EXTRACTION_ROOT / "batch_runner.py"
DEFAULT_CONFIG = EXTRACTION_ROOT / "config" / "pipeline.yaml"

for module_root in (OCR_ROOT, EXTRACTION_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from mineru_batch_parse import normalize_ref_no, select_pdfs  # noqa: E402
from stage_minus1_reorganize_mineru import reorganize_paper  # noqa: E402
from transform_mineru_to_standard import (  # noqa: E402
    ConfiguredMetaExtractor,
    MetaExtractor,
    transform_paper,
)


class PipelineRunnerError(RuntimeError):
    """统一流水线的参数或阶段执行错误。"""


def run_command(command: Sequence[str], *, label: str) -> None:
    print(f"\n[{label}] " + subprocess.list2cmdline(list(command)))
    completed = subprocess.run(list(command), cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise PipelineRunnerError(f"{label} 失败，退出码 {completed.returncode}")


def _batch_ref_nos(mineru_output: Path, batch_id: str) -> list[str]:
    manifest_path = mineru_output / f"batch_{batch_id}_manifest.json"
    status_path = mineru_output / f"batch_{batch_id}_status.json"
    candidates: list[str] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        candidates = [str(value) for value in payload.get("files") or []]
    elif status_path.is_file():
        payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
        results = ((payload.get("data") or {}).get("extract_result") or [])
        candidates = [str(item.get("file_name") or "") for item in results]
    if not candidates:
        raise PipelineRunnerError(
            f"无法从批次清单或状态文件确定文献：batch_id={batch_id}"
        )
    return list(dict.fromkeys(normalize_ref_no(value) for value in candidates))


def build_mineru_command(
    args: argparse.Namespace,
    *,
    input_dir: Path,
    mineru_output: Path,
    ref_nos: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        str(MINERU_SCRIPT),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(mineru_output),
        "--env-file",
        str(args.env_file.expanduser().resolve()),
        "--model-version",
        args.model_version,
        "--timeout-minutes",
        str(args.timeout_minutes),
    ]
    if args.batch_id:
        command.extend(("--batch-id", args.batch_id))
    else:
        for ref_no in ref_nos:
            command.extend(("--ref-no", ref_no))
    if args.ocr:
        command.append("--ocr")
    if args.language:
        command.extend(("--language", args.language))
    if args.disable_formula:
        command.append("--disable-formula")
    if args.disable_table:
        command.append("--disable-table")
    return command


def build_extraction_command(
    args: argparse.Namespace,
    *,
    processed_output: Path,
    ref_nos: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        str(BATCH_SCRIPT),
        "--config",
        str(args.config.expanduser().resolve()),
        "--input-dir",
        str(processed_output / "documents"),
    ]
    for ref_no in ref_nos:
        command.extend(("--ref-no", ref_no))
    optional_paths = (
        ("--output-dir", args.output_dir),
        ("--state-db", args.state_db),
        ("--summary-out", args.summary_out),
    )
    for option, value in optional_paths:
        if value is not None:
            command.extend((option, str(value.expanduser().resolve())))
    if args.workers is not None:
        command.extend(("--workers", str(args.workers)))
    if args.llm_workers is not None:
        command.extend(("--llm-workers", str(args.llm_workers)))
    for flag in ("retry_failed", "retry_interrupted", "recheck_completed", "force"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    if args.preview:
        command.append("--preview")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="串联 MinerU、标准化转换和 Stage 0-6 抽取",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="PDF 文件夹")
    parser.add_argument("--mineru-output", type=Path)
    parser.add_argument("--organized-root", type=Path, default=PROJECT_ROOT / "wenxian")
    parser.add_argument("--processed-output", type=Path, default=PROJECT_ROOT / "processed_data")
    parser.add_argument("--output-dir", type=Path, help="Stage 0-6 输出目录；默认读取配置")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=OCR_ROOT / ".env")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ref-no", action="append", help="只处理指定文献，可重复")
    selection.add_argument("--ref-list", type=Path, help="每行一个文献编号的清单")
    selection.add_argument("--max-documents", type=int, help="最多处理排序后的前 N 篇")

    parser.add_argument("--batch-id", help="继续已有 MinerU 批次")
    parser.add_argument("--model-version", choices=("pipeline", "vlm"), default="vlm")
    parser.add_argument("--language")
    parser.add_argument("--ocr", action="store_true", help="为扫描型 PDF 启用 OCR")
    parser.add_argument("--disable-formula", action="store_true")
    parser.add_argument("--disable-table", action="store_true")
    parser.add_argument("--timeout-minutes", type=float, default=120.0)

    parser.add_argument("--skip-ocr", action="store_true", help="复用已有 MinerU 结果")
    parser.add_argument("--skip-organize", action="store_true", help="复用已有 Stage -1 结果")
    parser.add_argument("--skip-transform", action="store_true", help="复用已有 document JSON")
    parser.add_argument("--skip-extraction", action="store_true", help="只运行到 document JSON")
    parser.add_argument("--skip-meta", action="store_true", help="标准化时不调用元数据 LLM")
    parser.add_argument("--force-meta", action="store_true")

    parser.add_argument("--workers", type=int)
    parser.add_argument("--llm-workers", type=int)
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-interrupted", action="store_true")
    parser.add_argument("--recheck-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Stage 5 后发布 candidate.json 和 report_candidate.html，不执行 Stage 6 严格校验",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    mineru_output = (
        args.mineru_output.expanduser().resolve()
        if args.mineru_output
        else input_dir / "mineru_output"
    )
    organized_root = args.organized_root.expanduser().resolve()
    processed_output = args.processed_output.expanduser().resolve()

    if args.batch_id and (args.ref_no or args.ref_list or args.max_documents):
        raise PipelineRunnerError("--batch-id 不能与文献选择参数同时使用")
    if args.batch_id:
        ref_nos = []
        if args.skip_ocr:
            ref_nos = _batch_ref_nos(mineru_output, args.batch_id)
    else:
        pdf_files = select_pdfs(
            input_dir,
            ref_nos=args.ref_no,
            ref_list_path=args.ref_list,
            max_documents=args.max_documents,
        )
        ref_nos = [normalize_ref_no(path.name) for path in pdf_files]

    if ref_nos:
        print(f"已选择 {len(ref_nos)} 篇：" + ", ".join(ref_nos))

    mineru_command = build_mineru_command(
        args,
        input_dir=input_dir,
        mineru_output=mineru_output,
        ref_nos=ref_nos,
    )
    if args.dry_run:
        if not ref_nos and args.batch_id:
            print("文献清单将在 MinerU 批次恢复完成后从状态文件读取。")
        if not args.skip_ocr:
            print("[dry-run: MinerU] " + subprocess.list2cmdline(mineru_command))
        if not ref_nos:
            print("[dry-run] 后续命令将在取得批次文献清单后生成。")
            return 0
        extraction_command = build_extraction_command(
            args,
            processed_output=processed_output,
            ref_nos=ref_nos,
        )
        if not args.skip_organize:
            print(f"[dry-run: Stage -1] {len(ref_nos)} 篇 -> {organized_root}")
        if not args.skip_transform:
            print(f"[dry-run: 标准化] {len(ref_nos)} 篇 -> {processed_output / 'documents'}")
        if not args.skip_extraction:
            label = "Stage 0-5 + Candidate" if args.preview else "Stage 0-6"
            print(f"[dry-run: {label}] " + subprocess.list2cmdline(extraction_command))
        return 0

    if not args.skip_ocr:
        run_command(mineru_command, label="MinerU")
    if args.batch_id and not ref_nos:
        ref_nos = _batch_ref_nos(mineru_output, args.batch_id)

    if args.batch_id:
        print(f"批次包含 {len(ref_nos)} 篇：" + ", ".join(ref_nos))

    if not args.skip_organize:
        print("\n[Stage -1] 整理 MinerU 产物")
        for ref_no in ref_nos:
            stats = reorganize_paper(mineru_output, organized_root, ref_no)
            errors = stats.get("errors") or []
            if errors:
                raise PipelineRunnerError(f"Stage -1 失败：{ref_no}：{'；'.join(errors)}")
            print(f"[done] {ref_no}")

    if not args.skip_transform:
        print("\n[标准化] 生成 document JSON")
        meta_extractor: MetaExtractor | None = None
        if not args.skip_meta:
            try:
                meta_extractor = ConfiguredMetaExtractor(args.config.expanduser().resolve())
            except Exception as exc:
                raise PipelineRunnerError(
                    "元数据 LLM 初始化失败；如需明确跳过，请使用 --skip-meta"
                ) from exc
        for ref_no in ref_nos:
            output = transform_paper(
                mineru_output,
                organized_root,
                processed_output,
                ref_no,
                force_meta=args.force_meta,
                meta_extractor=meta_extractor,
            )
            if not args.skip_meta:
                try:
                    document = json.loads(output.read_text(encoding="utf-8-sig"))
                    extraction = document["paper"]["metadata_extraction"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise PipelineRunnerError(
                        f"元数据抽取结果无法审计：{ref_no}"
                    ) from exc
                if extraction.get("status") != "success":
                    error_type = extraction.get("error_type") or "unknown"
                    raise PipelineRunnerError(
                        f"元数据抽取失败：{ref_no}（{error_type}）；"
                        "如需明确跳过，请使用 --skip-meta"
                    )
            print(f"[done] {ref_no} -> {output}")

    if not args.skip_extraction:
        extraction_command = build_extraction_command(
            args,
            processed_output=processed_output,
            ref_nos=ref_nos,
        )
        label = "Stage 0-5 + Candidate" if args.preview else "Stage 0-6"
        run_command(extraction_command, label=label)
    if args.preview:
        print("\nPreview 流水线执行完成。")
    else:
        print("\n完整流水线执行完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中止。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
