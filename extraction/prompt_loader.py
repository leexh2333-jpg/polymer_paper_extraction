"""Markdown Prompt 扫描、校验、渲染与哈希。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


EXTRACTION_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = EXTRACTION_ROOT / "prompts"
FRONT_MATTER_RE = re.compile(
    r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n)?(.*)\Z",
    flags=re.DOTALL,
)
UNRESOLVED_TEMPLATE_RE = re.compile(r"{{\s*[^{}]+\s*}}")


class PromptError(RuntimeError):
    """Prompt front matter、引用或模板渲染错误。"""


@dataclass(frozen=True)
class PromptDocument:
    prompt_id: str
    version: str
    stage: str
    output_schema: str
    body: str
    path: Path


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    version: str
    stage: str
    output_schema_version: str
    text: str
    sha256: str


class PromptLoader:
    def __init__(self, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> None:
        self.prompts_dir = prompts_dir
        self._prompts = self._scan()

    def _scan(self) -> dict[str, PromptDocument]:
        if not self.prompts_dir.is_dir():
            raise PromptError(f"Prompt 目录不存在：{self.prompts_dir}")
        prompts: dict[str, PromptDocument] = {}
        for path in sorted(self.prompts_dir.glob("*.md")):
            document = self._parse(path)
            if document.prompt_id in prompts:
                raise PromptError(f"prompt_id 重复：{document.prompt_id}")
            prompts[document.prompt_id] = document
        return prompts

    @staticmethod
    def _parse(path: Path) -> PromptDocument:
        text = path.read_text(encoding="utf-8-sig")
        match = FRONT_MATTER_RE.match(text)
        if not match:
            raise PromptError(f"Prompt 缺少有效 YAML front matter：{path.name}")
        try:
            metadata = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise PromptError(f"Prompt front matter 无效：{path.name}") from exc
        required = ("prompt_id", "version", "stage", "output_schema")
        missing = [key for key in required if not str(metadata.get(key) or "").strip()]
        if missing:
            raise PromptError(f"Prompt 缺少字段 {missing}：{path.name}")
        body = match.group(2).strip()
        if not body:
            raise PromptError(f"Prompt 正文为空：{path.name}")
        return PromptDocument(
            prompt_id=str(metadata["prompt_id"]).strip(),
            version=str(metadata["version"]).strip(),
            stage=str(metadata["stage"]).strip(),
            output_schema=str(metadata["output_schema"]).strip(),
            body=body,
            path=path,
        )

    def get(self, prompt_id: str) -> PromptDocument:
        try:
            return self._prompts[prompt_id]
        except KeyError as exc:
            raise PromptError(f"找不到 prompt_id：{prompt_id}") from exc

    @property
    def prompt_ids(self) -> set[str]:
        return set(self._prompts)

    def render_stage_prompt(
        self,
        prompt_id: str,
        output_model: type[BaseModel],
        *,
        expected_stage: str,
        expected_output_schema: str,
    ) -> RenderedPrompt:
        common = self.get("polymer.common.guardrails")
        stage_prompt = self.get(prompt_id)
        if stage_prompt.stage != expected_stage:
            raise PromptError(
                f"{prompt_id} stage={stage_prompt.stage!r}，预期 {expected_stage!r}"
            )
        if stage_prompt.output_schema != expected_output_schema:
            raise PromptError(
                f"{prompt_id} output_schema={stage_prompt.output_schema!r}，"
                f"预期 {expected_output_schema!r}"
            )
        schema_json = json.dumps(
            output_model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stage_body = stage_prompt.body.replace("{{output_schema}}", schema_json)
        text = (
            "# Common Guardrails\n\n"
            + common.body
            + "\n\n# Stage Instructions\n\n"
            + stage_body
        )
        unresolved = UNRESOLVED_TEMPLATE_RE.findall(text)
        if unresolved:
            raise PromptError(f"Prompt 存在未替换模板变量：{unresolved}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return RenderedPrompt(
            prompt_id=stage_prompt.prompt_id,
            version=stage_prompt.version,
            stage=stage_prompt.stage,
            output_schema_version=stage_prompt.output_schema,
            text=text,
            sha256=digest,
        )
