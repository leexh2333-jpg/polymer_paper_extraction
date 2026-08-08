#!/usr/bin/env python3
"""
Stage -1: Reorganize MinerU Output

将 MinerU 的输出目录结构改造为统一的文献库格式，类似 2d/wenxian 的结构。

输入: mineru_output/
  ├── reference_no_0001016/
  │   ├── {uuid}_content_list.json
  │   ├── {uuid}_content_list_v2.json
  │   ├── full.md
  │   ├── {uuid}_origin.pdf
  │   └── images/*.jpg

输出: reorganized/
  ├── reference_no_0001016/
  │   ├── content.json          # 标准化的内容文件
  │   ├── content_v2.json       # MinerU v2 阅读顺序结构
  │   ├── reference_no_0001016.md
  │   ├── origin.pdf            # 原始PDF副本
  │   └── images/              # 图片（保留 MinerU 哈希文件名）
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any


def find_content_list_file(paper_dir: Path) -> Path:
    """在 paper 目录中查找 *_content_list.json 文件"""
    candidates = list(paper_dir.glob("*_content_list.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_content_list.json found in {paper_dir}")
    # 优先返回非 v2 版本
    for c in candidates:
        if "_content_list_v2.json" not in c.name:
            return c
    return candidates[0]


def find_origin_pdf(paper_dir: Path) -> Path:
    """查找原始 PDF 文件"""
    candidates = list(paper_dir.glob("*_origin.pdf"))
    if not candidates:
        raise FileNotFoundError(f"No *_origin.pdf found in {paper_dir}")
    return candidates[0]


def find_content_v2_file(paper_dir: Path) -> Path:
    """查找 MinerU v2 内容文件。"""
    candidates = sorted(paper_dir.glob("*_content_list_v2.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_content_list_v2.json found in {paper_dir}")
    return candidates[0]


def find_markdown_file(paper_dir: Path, ref_no: str) -> Path:
    """优先查找以文献编号命名的 Markdown。"""
    expected = paper_dir / f"{ref_no}.md"
    if expected.is_file():
        return expected
    candidates = sorted(paper_dir.glob("*.md"))
    if not candidates:
        raise FileNotFoundError(f"No Markdown file found in {paper_dir}")
    return candidates[0]


def find_images_dir(paper_dir: Path, ref_no: str) -> Path:
    """查找图片目录"""
    current_dir = paper_dir / "images"
    if current_dir.is_dir():
        return current_dir

    legacy_dir = paper_dir / f"{ref_no}_images"
    if legacy_dir.is_dir():
        return legacy_dir

    candidates = sorted(
        path
        for path in paper_dir.iterdir()
        if path.is_dir() and path.name.endswith("_images")
    )
    return candidates[0] if candidates else current_dir


def extract_table_and_figure_refs(blocks: List[Dict]) -> Dict[str, str]:
    """
    从 content_list 中提取图片路径，保留 MinerU 原始文件名。
    返回: {原始图片路径: 原始文件名}
    """
    mapping = {}
    for block in blocks:
        img_path = block.get("img_path")
        if block.get("type") in {"chart", "image", "table"} and img_path:
            mapping[img_path] = Path(img_path).name

    return mapping


def reorganize_paper(
    input_dir: Path,
    output_dir: Path,
    ref_no: str,
    keep_pdf: bool = True,
    keep_images: bool = True
) -> Dict[str, Any]:
    """
    重组单篇文献的输出结构

    Args:
        input_dir: MinerU 输出目录（如 mineru_output/reference_no_0001016）
        output_dir: 目标输出目录
        ref_no: 文献编号
        keep_pdf: 是否保留 PDF 副本
        keep_images: 是否保留图片

    Returns:
        统计信息字典
    """
    paper_input = input_dir / ref_no
    paper_output = output_dir / ref_no

    if not paper_input.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_input}")

    # 创建输出目录
    paper_output.mkdir(parents=True, exist_ok=True)

    stats = {
        "ref_no": ref_no,
        "blocks_count": 0,
        "images_copied": 0,
        "pdf_copied": False,
        "markdown_copied": False,
        "content_v2_copied": False,
        "warnings": [],
        "errors": []
    }

    try:
        # 1. 处理 content_list.json
        content_list_file = find_content_list_file(paper_input)
        with open(content_list_file, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        stats["blocks_count"] = len(content_list)

        # 2. 提取图片引用关系
        img_mapping = extract_table_and_figure_refs(content_list)
        image_name_mapping = {
            Path(old_path).name: new_name
            for old_path, new_name in img_mapping.items()
        }

        # 3. 复制图片，保留 MinerU 原始文件名
        if keep_images:
            images_input = find_images_dir(paper_input, ref_no)
            images_output = paper_output / "images"
            images_output.mkdir(exist_ok=True)

            if images_input.exists():
                for old_rel_path, new_name in img_mapping.items():
                    # img_path 格式可能是 "images/xxxx.jpg" 或 "reference_no_XXXX_images/xxxx.jpg"
                    old_filename = Path(old_rel_path).name
                    src = images_input / old_filename
                    dst = images_output / new_name

                    if src.exists():
                        shutil.copy2(src, dst)
                        stats["images_copied"] += 1
                    else:
                        stats["errors"].append(f"Image not found: {src}")

                # 复制剩余未在 content_list 中引用的图片
                mapped_names = {Path(path).name for path in img_mapping}
                for img_file in images_input.iterdir():
                    if img_file.is_file() and img_file.name not in mapped_names:
                        dst = images_output / img_file.name
                        if not dst.exists():
                            shutil.copy2(img_file, dst)
                            stats["images_copied"] += 1
                        image_name_mapping[img_file.name] = img_file.name

        # 4. 保留 Markdown，并同步整理后的图片路径
        try:
            markdown_src = find_markdown_file(paper_input, ref_no)
            markdown_text = markdown_src.read_text(encoding="utf-8")
            for old_name, new_name in image_name_mapping.items():
                markdown_text = markdown_text.replace(
                    f"{ref_no}_images/{old_name}",
                    f"images/{new_name}",
                )
                markdown_text = markdown_text.replace(
                    f"images/{old_name}",
                    f"images/{new_name}",
                )
            (paper_output / f"{ref_no}.md").write_text(markdown_text, encoding="utf-8")
            stats["markdown_copied"] = True
        except FileNotFoundError as e:
            stats["errors"].append(str(e))

        # 5. 保留 content_list_v2.json，供阅读顺序与行内公式对齐
        try:
            content_v2_src = find_content_v2_file(paper_input)
            shutil.copy2(content_v2_src, paper_output / "content_v2.json")
            stats["content_v2_copied"] = True
        except FileNotFoundError as e:
            stats["warnings"].append(str(e))

        # 6. 标准化 content.json（更新图片路径）
        normalized_content = []
        for block in content_list:
            block_copy = block.copy()

            # 更新图片路径
            if "img_path" in block_copy:
                old_path = block_copy["img_path"]
                if old_path in img_mapping:
                    block_copy["img_path"] = f"images/{img_mapping[old_path]}"

            normalized_content.append(block_copy)

        content_output = {
            "document_id": ref_no,
            "source": "mineru",
            "blocks": normalized_content
        }

        with open(paper_output / "content.json", "w", encoding="utf-8") as f:
            json.dump(content_output, f, ensure_ascii=False, indent=2)

        # 7. 复制 PDF
        if keep_pdf:
            try:
                pdf_src = find_origin_pdf(paper_input)
                pdf_dst = paper_output / "origin.pdf"
                shutil.copy2(pdf_src, pdf_dst)
                stats["pdf_copied"] = True
            except FileNotFoundError as e:
                stats["errors"].append(str(e))

    except Exception as e:
        stats["errors"].append(f"Failed to process {ref_no}: {str(e)}")

    return stats


def reorganize_batch(
    input_root: str,
    output_root: str,
    keep_pdf: bool = True,
    keep_images: bool = True,
    match_pattern: str = None
):
    """
    批量重组 MinerU 输出

    Args:
        input_root: MinerU 输出根目录（包含多个 reference_no_XXX 子目录）
        output_root: 输出根目录
        keep_pdf: 是否保留 PDF
        keep_images: 是否保留图片
        match_pattern: 只处理匹配此模式的文献（如 "0001016"）
    """
    input_path = Path(input_root)
    output_path = Path(output_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    output_path.mkdir(parents=True, exist_ok=True)

    # 扫描所有 reference_no_XXX 目录
    paper_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("reference_no_")]

    if match_pattern:
        paper_dirs = [d for d in paper_dirs if match_pattern in d.name]

    print(f"Found {len(paper_dirs)} papers to process")

    all_stats = []
    for paper_dir in sorted(paper_dirs):
        ref_no = paper_dir.name
        print(f"Processing {ref_no}...")

        stats = reorganize_paper(
            input_path,
            output_path,
            ref_no,
            keep_pdf=keep_pdf,
            keep_images=keep_images
        )

        all_stats.append(stats)

        if stats["errors"]:
            print(f"  [failed] Errors: {len(stats['errors'])}")
            for err in stats["errors"]:
                print(f"    - {err}")
        elif stats["warnings"]:
            print(f"  [warning] Warnings: {len(stats['warnings'])}")
            for warning in stats["warnings"]:
                print(f"    - {warning}")
        else:
            print(f"  [done] {stats['blocks_count']} blocks, {stats['images_copied']} images")

    # 输出汇总
    summary_path = output_path / "_reorganize_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_papers": len(all_stats),
            "details": all_stats
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSummary written to {summary_path}")
    print(f"Total: {len(all_stats)} papers processed")
    return all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reorganize MinerU output to unified structure")
    parser.add_argument(
        "--input",
        default=r"D:\1work\1_2026\polymer\polyinfo数据\sample_exprot_34\mineru_output",
        help="MinerU output directory"
    )
    parser.add_argument(
        "--output",
        default=r"D:\1work\1_2026\polymer\wenxian",
        help="Output directory for reorganized data"
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not copy PDF files (save disk space)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not copy image files (save disk space)"
    )
    parser.add_argument(
        "--match",
        help="Only process papers matching this pattern (e.g., '0001016')"
    )

    args = parser.parse_args()

    results = reorganize_batch(
        input_root=args.input,
        output_root=args.output,
        keep_pdf=not args.no_pdf,
        keep_images=not args.no_images,
        match_pattern=args.match
    )
    if any(item["errors"] for item in results):
        raise SystemExit(1)
