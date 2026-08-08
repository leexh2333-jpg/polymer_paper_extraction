from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFS = [
    line.strip()
    for line in (ROOT / "preview" / "demo_latest_20_refs.txt").read_text(encoding="utf-8-sig").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
issues: list[str] = []
if len(REFS) != 20 or len(set(REFS)) != 20:
    issues.append(f"ref list count/unique mismatch: {len(REFS)}/{len(set(REFS))}")
for ref in REFS:
    document = ROOT / "sample_data" / "processed_documents" / f"{ref}_document.json"
    pdf = ROOT / "source_pdfs" / f"{ref}.pdf"
    if not document.is_file():
        issues.append(f"missing document: {document.relative_to(ROOT)}")
    else:
        try:
            json.loads(document.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            issues.append(f"invalid document JSON {document.name}: {exc}")
    if not pdf.is_file():
        issues.append(f"missing PDF: {pdf.relative_to(ROOT)}")
    elif pdf.read_bytes()[:5] != b"%PDF-":
        issues.append(f"invalid PDF header: {pdf.name}")

for path in ROOT.rglob("*"):
    if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
        issues.append(f"cache included: {path.relative_to(ROOT)}")
    if path.is_file() and (path.suffix.lower() in {".pyc", ".pyo"} or path.name == ".env"):
        issues.append(f"forbidden file included: {path.relative_to(ROOT)}")

config = (ROOT / "extraction" / "config" / "pipeline.yaml").read_text(encoding="utf-8")
if re.search(r"(?im)^\s*(api[_-]?key|secret|password|authorization)\s*:\s*\S+", config):
    issues.append("pipeline.yaml contains a credential-like literal")
if re.search(r"(?i)[A-Z]:\\", config):
    issues.append("pipeline.yaml still contains a Windows absolute path")

if issues:
    print("交付结构检查失败：")
    for issue in issues:
        print("-", issue)
    raise SystemExit(1)
print(f"交付结构检查通过：20 个 JSON + 20 个 PDF；ref 数={len(REFS)}")
