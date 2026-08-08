"""批量上传本地 PDF 到 MinerU，并下载、解压解析结果。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://mineru.net/api/v4"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "polyinfo数据" / "sample_exprot_34"
TERMINAL_STATES = {"done", "failed"}


def load_env_file(env_path: Path) -> None:
    """读取简单 KEY=VALUE 格式的 .env，不覆盖已有环境变量。"""
    if not env_path.is_file():
        raise FileNotFoundError(
            f"未找到环境变量文件：{env_path}\n"
            "请复制 .env.example 为 .env，并填写 MINERU_API_KEY。"
        )
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def make_data_id(pdf_path: Path) -> str:
    """生成符合 MinerU 限制、尽量可读且稳定的 data_id。"""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf_path.stem).strip("_.-") or "document"
    digest = hashlib.sha1(pdf_path.name.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:117]}-{digest}"


def api_json(session: requests.Session, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = session.request(method, url, timeout=kwargs.pop("timeout", (30, 120)), **kwargs)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"MinerU 返回了非 JSON 响应：{response.text[:500]}") from exc
    if payload.get("code") != 0:
        trace_id = payload.get("trace_id", "")
        suffix = f"（trace_id={trace_id}）" if trace_id else ""
        raise RuntimeError(f"MinerU API 请求失败：{payload.get('msg', payload)}{suffix}")
    return payload


def collect_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    pdf_files = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )
    if not pdf_files:
        raise FileNotFoundError(f"目录中没有 PDF 文件：{input_dir}")
    return pdf_files


def normalize_ref_no(value: str) -> str:
    """将文献编号、PDF 文件名统一为 reference_no_XXXXXXX。"""
    ref_no = Path(value.strip()).stem
    if ref_no.endswith("_document"):
        ref_no = ref_no.removesuffix("_document")
    suffix = ref_no.removeprefix("reference_no_")
    if not ref_no.startswith("reference_no_") or not suffix.isdigit():
        raise ValueError(f"无效文献编号：{value!r}")
    return ref_no


def select_pdfs(
    input_dir: Path,
    *,
    ref_nos: list[str] | None = None,
    ref_list_path: Path | None = None,
    max_documents: int | None = None,
) -> list[Path]:
    """按显式清单或数量上限选择 PDF，并执行 MinerU 单批上限校验。"""
    pdf_files = collect_pdfs(input_dir)
    selected_ref_nos: list[str] = []
    if ref_nos:
        selected_ref_nos.extend(normalize_ref_no(value) for value in ref_nos)
    elif ref_list_path is not None:
        path = ref_list_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"文献清单不存在：{path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                selected_ref_nos.append(normalize_ref_no(value))

    if selected_ref_nos:
        by_ref_no = {normalize_ref_no(path.name): path for path in pdf_files}
        unique_ref_nos = list(dict.fromkeys(selected_ref_nos))
        missing = [ref_no for ref_no in unique_ref_nos if ref_no not in by_ref_no]
        if missing:
            raise FileNotFoundError("未找到指定 PDF：" + ", ".join(missing))
        pdf_files = [by_ref_no[ref_no] for ref_no in unique_ref_nos]
    elif max_documents is not None:
        if max_documents < 1:
            raise ValueError("max-documents 必须大于 0")
        pdf_files = pdf_files[:max_documents]

    if len(pdf_files) > 200:
        raise ValueError(f"当前选择 {len(pdf_files)} 个 PDF；MinerU 单批上限为 200 个。")
    return pdf_files


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def build_manifest(
    batch_id: str,
    input_dir: Path,
    pdf_files: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """构造不含凭据和临时 URL 的 OCR provenance manifest。"""
    return {
        "batch_id": batch_id,
        "input_dir": str(input_dir),
        "files": [str(path) for path in pdf_files],
        "model_version": args.model_version,
        "ocr_enabled": bool(args.ocr),
        "language": args.language,
        "page_ranges": args.page_ranges,
        "enable_formula": not args.disable_formula,
        "enable_table": not args.disable_table,
        "extra_formats": list(args.extra_formats),
    }


def apply_upload_urls(session: requests.Session, pdf_files: list[Path], args: argparse.Namespace) -> tuple[str, list[str]]:
    files = []
    for pdf_path in pdf_files:
        item: dict[str, Any] = {"name": pdf_path.name, "data_id": make_data_id(pdf_path)}
        if args.ocr:
            item["is_ocr"] = True
        if args.page_ranges:
            item["page_ranges"] = args.page_ranges
        files.append(item)

    body: dict[str, Any] = {
        "files": files,
        "model_version": args.model_version,
        "enable_formula": not args.disable_formula,
        "enable_table": not args.disable_table,
    }
    if args.language:
        body["language"] = args.language
    if args.extra_formats:
        body["extra_formats"] = args.extra_formats

    payload = api_json(session, "POST", f"{BASE_URL}/file-urls/batch", json=body)
    data = payload.get("data") or {}
    batch_id = data.get("batch_id")
    upload_urls = data.get("file_urls") or []
    if not batch_id or len(upload_urls) != len(pdf_files):
        raise RuntimeError(
            "MinerU 返回的 batch_id 或上传链接数量不符合预期："
            f"batch_id={batch_id!r}, urls={len(upload_urls)}, files={len(pdf_files)}"
        )
    return str(batch_id), [str(url) for url in upload_urls]


def upload_files(session: requests.Session, pdf_files: list[Path], upload_urls: list[str]) -> None:
    max_attempts = 4
    for index, (pdf_path, upload_url) in enumerate(zip(pdf_files, upload_urls), start=1):
        print(f"[{index}/{len(pdf_files)}] 上传：{pdf_path.name}")
        for attempt in range(1, max_attempts + 1):
            try:
                # 每次重试都重新打开文件，避免文件指针停在文件末尾。
                with pdf_path.open("rb") as file_obj:
                    # 预签名 URL 上传时不携带 MinerU API 鉴权头。
                    response = session.put(
                        upload_url,
                        data=file_obj,
                        headers={"Authorization": None, "Content-Type": None},
                        timeout=(60, 1200),
                    )
                if 500 <= response.status_code < 600 and attempt < max_attempts:
                    raise requests.HTTPError(f"服务器返回 HTTP {response.status_code}")
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"上传失败：{pdf_path.name}，已重试 {max_attempts} 次：{exc}"
                    ) from exc
                wait_seconds = min(30, 2 ** (attempt - 1))
                print(
                    f"上传暂时失败（第 {attempt}/{max_attempts} 次）：{exc}；"
                    f"{wait_seconds} 秒后重试"
                )
                time.sleep(wait_seconds)


def poll_batch(
    session: requests.Session,
    batch_id: str,
    output_dir: Path,
    poll_interval: float,
    timeout_minutes: float,
    expected_count: int | None,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_minutes * 60
    status_path = output_dir / f"batch_{batch_id}_status.json"
    while True:
        payload = api_json(session, "GET", f"{BASE_URL}/extract-results/batch/{batch_id}")
        write_json(status_path, payload)
        results = (payload.get("data") or {}).get("extract_result") or []
        states = Counter(str(item.get("state", "unknown")) for item in results)
        state_text = ", ".join(f"{k}={v}" for k, v in sorted(states.items()))
        print(f"批次 {batch_id}：已返回 {len(results)} 个结果" + (f"（{state_text}）" if state_text else ""))
        enough = expected_count is None or len(results) >= expected_count
        if results and enough and all(str(item.get("state")) in TERMINAL_STATES for item in results):
            return [dict(item) for item in results]
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"等待 MinerU 解析超时（{timeout_minutes:g} 分钟）。"
                f"可使用 --batch-id {batch_id} 继续查询和下载。"
            )
        time.sleep(poll_interval)


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """解压 ZIP，并阻止成员路径逃逸目标目录。"""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"ZIP 中包含不安全路径：{member.filename}") from exc
        archive.extractall(destination)


def extract_and_delete_zip(zip_path: Path, destination: Path) -> None:
    """成功解压后删除 ZIP；解压失败时保留 ZIP 便于排查。"""
    safe_extract_zip(zip_path, destination)
    zip_path.unlink()


def download_results(
    session: requests.Session,
    results: list[dict[str, Any]],
    output_dir: Path,
    extract_zip: bool,
) -> tuple[int, int]:
    done_count = failed_count = 0
    for item in results:
        file_name = str(item.get("file_name") or item.get("data_id") or "document.pdf")
        state = str(item.get("state", "unknown"))
        if state == "failed":
            failed_count += 1
            print(f"解析失败：{file_name}；原因：{item.get('err_msg') or '未知'}")
            continue
        if state != "done":
            continue

        zip_url = item.get("full_zip_url")
        if not zip_url:
            failed_count += 1
            print(f"结果缺少 full_zip_url：{file_name}")
            continue

        safe_stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(file_name).stem).strip() or "document"
        zip_path = output_dir / f"{safe_stem}.zip"
        extract_dir = output_dir / safe_stem
        print(f"下载结果：{file_name} -> {zip_path}")
        if not (zip_path.is_file() and zipfile.is_zipfile(zip_path)):
            temp_path = zip_path.with_suffix(".zip.part")
            with session.get(
                str(zip_url),
                stream=True,
                headers={"Authorization": None, "Content-Type": None},
                timeout=(30, 600),
            ) as response:
                response.raise_for_status()
                with temp_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)
            if not zipfile.is_zipfile(temp_path):
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(f"下载内容不是有效 ZIP：{file_name}")
            temp_path.replace(zip_path)
        if extract_zip:
            extract_and_delete_zip(zip_path, extract_dir)
            print(f"已解压并删除压缩包：{extract_dir}")
        done_count += 1
    return done_count, failed_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量上传本地 PDF 到 MinerU，并下载解析结果。")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=SCRIPT_DIR / ".env")
    parser.add_argument("--batch-id", help="跳过上传，继续查询并下载已有批次")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ref-no", action="append", help="只处理指定文献，可重复")
    selection.add_argument("--ref-list", type=Path, help="每行一个文献编号的清单")
    selection.add_argument("--max-documents", type=int, help="最多处理排序后的前 N 篇")
    parser.add_argument("--model-version", choices=("pipeline", "vlm"), default="vlm")
    parser.add_argument("--language", help="文档语言，例如 ch 或 en；不填则使用服务端默认值")
    parser.add_argument("--ocr", action="store_true", help="启用 OCR")
    parser.add_argument("--page-ranges", help='页码范围，例如 "2,4-6"')
    parser.add_argument(
        "--extra-formats", nargs="*", choices=("docx", "html", "latex"), default=[]
    )
    parser.add_argument("--disable-formula", action="store_true")
    parser.add_argument("--disable-table", action="store_true")
    parser.add_argument("--no-extract", action="store_true", help="只下载 ZIP，不解压")
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--timeout-minutes", type=float, default=120.0)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir / "mineru_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(args.env_file.expanduser().resolve())
    token = os.environ.get("MINERU_API_KEY", "").strip()
    if not token:
        raise RuntimeError(".env 中的 MINERU_API_KEY 为空。")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })

    if args.batch_id:
        batch_id = args.batch_id
        expected_count = None
        print(f"继续查询已有批次：{batch_id}")
    else:
        pdf_files = select_pdfs(
            input_dir,
            ref_nos=args.ref_no,
            ref_list_path=args.ref_list,
            max_documents=args.max_documents,
        )
        print(f"发现 {len(pdf_files)} 个 PDF：{input_dir}")
        batch_id, upload_urls = apply_upload_urls(session, pdf_files, args)
        expected_count = len(pdf_files)
        write_json(
            output_dir / f"batch_{batch_id}_manifest.json",
            build_manifest(batch_id, input_dir, pdf_files, args),
        )
        print(f"已创建批次：{batch_id}")
        upload_files(session, pdf_files, upload_urls)
        print("全部文件已上传，MinerU 将自动提交解析任务。")

    results = poll_batch(
        session,
        batch_id,
        output_dir,
        args.poll_interval,
        args.timeout_minutes,
        expected_count,
    )
    done_count, failed_count = download_results(
        session, results, output_dir, extract_zip=not args.no_extract
    )
    print(f"完成：成功下载 {done_count} 个，失败 {failed_count} 个。")
    print(f"本地结果目录：{output_dir}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中止。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)



