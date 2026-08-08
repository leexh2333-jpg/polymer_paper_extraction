"""结构化抽取 Pipeline 的轻量 OpenAI-compatible LLM 客户端。"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "pipeline.yaml"
SUPPORTED_API_FORMATS = {
    "openai-chat-completions",
    "anthropic-messages",
}
SUPPORTED_THINKING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class LLMConfigurationError(RuntimeError):
    """LLM 非敏感配置或环境变量不完整。"""


class LLMRequestError(RuntimeError):
    """LLM 请求或响应解析失败。"""


class LLMOutputTruncatedError(LLMRequestError):
    """模型因输出长度限制而截断结构化响应。"""

    error_code = "output_truncated"


@dataclass(frozen=True)
class ResolvedLLMConfig:
    provider: str
    requested_model: str
    model: str
    base_url: str
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    api_format: str = "openai-chat-completions"
    thinking_effort: str | None = None
    stream: bool = False


@dataclass(frozen=True)
class LLMPricing:
    currency: str
    input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class LLMTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def billable_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.billable_input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMCallCost:
    currency: str
    input_per_million: Decimal
    output_per_million: Decimal
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


@dataclass(frozen=True)
class LLMCallRecord:
    provider: str
    model: str
    usage: LLMTokenUsage
    cost: LLMCallCost | None
    usage_available: bool = True


@dataclass(frozen=True)
class LLMJSONResponse:
    data: dict[str, Any]
    provider: str
    model: str
    usage: LLMTokenUsage = field(default_factory=LLMTokenUsage)
    cost: LLMCallCost | None = None


@dataclass(frozen=True)
class LLMRawResponse:
    """最近一次已收到的模型文本；不含请求、凭据或 HTTP headers。"""

    provider: str
    model: str
    finish_reason: str | None
    content: str | None
    usage: LLMTokenUsage
    cost: LLMCallCost | None


def llm_config_cache_payload(
    resolved: ResolvedLLMConfig,
) -> dict[str, Any]:
    """返回兼容新增协议字段前缓存的非敏感模型配置。"""
    payload = asdict(resolved)
    if resolved.api_format == "openai-chat-completions":
        payload.pop("api_format")
    if resolved.thinking_effort is None:
        payload.pop("thinking_effort")
    if not resolved.stream:
        payload.pop("stream")
    return payload


def load_pipeline_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise LLMConfigurationError(f"LLM 配置文件不存在：{path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise LLMConfigurationError(f"LLM 配置文件无效：{path}") from exc
    if not isinstance(data, dict):
        raise LLMConfigurationError("LLM 配置根节点必须是对象")
    return data


def resolve_pricing_config(
    config: dict[str, Any],
    model: str,
) -> LLMPricing | None:
    pricing = config.get("pricing")
    if pricing is None:
        return None
    if not isinstance(pricing, dict):
        raise LLMConfigurationError("pricing 必须是对象")
    currency = str(pricing.get("currency") or "").strip().upper()
    models = pricing.get("models")
    entry = models.get(model) if isinstance(models, dict) else None
    if not currency or not isinstance(entry, dict):
        raise LLMConfigurationError(f"缺少模型 {model} 的 pricing 配置")
    try:
        input_rate = Decimal(str(entry["input_per_million"]))
        output_rate = Decimal(str(entry["output_per_million"]))
    except (KeyError, InvalidOperation) as exc:
        raise LLMConfigurationError(f"模型 {model} 的 pricing 无效") from exc
    if input_rate < 0 or output_rate < 0:
        raise LLMConfigurationError("pricing 单价不得为负数")
    return LLMPricing(
        currency=currency,
        input_per_million=input_rate,
        output_per_million=output_rate,
    )


def resolve_llm_config(
    config: dict[str, Any],
    stage: str,
) -> ResolvedLLMConfig:
    llm = config.get("llm")
    if not isinstance(llm, dict):
        raise LLMConfigurationError("配置缺少 llm 节点")
    default = llm.get("default")
    if not isinstance(default, dict):
        raise LLMConfigurationError("配置缺少 llm.default 节点")
    resolved = dict(default)
    overrides = llm.get("stage_overrides") or {}
    if isinstance(overrides, dict) and isinstance(overrides.get(stage), dict):
        resolved.update(overrides[stage])

    provider = str(resolved.get("provider") or "").strip().lower()
    requested_model = str(resolved.get("model") or "").strip()
    base_url = str(resolved.get("base_url") or "").strip().rstrip("/")
    if not provider or not requested_model or not base_url:
        raise LLMConfigurationError("LLM provider、model、base_url 均为必填项")
    model = requested_model
    api_format = str(
        resolved.get("api_format") or "openai-chat-completions"
    ).strip().lower()
    if api_format not in SUPPORTED_API_FORMATS:
        raise LLMConfigurationError(
            f"不支持的 LLM api_format：{api_format}"
        )
    thinking_effort_value = resolved.get("thinking_effort")
    thinking_effort = (
        str(thinking_effort_value).strip().lower()
        if thinking_effort_value is not None
        else None
    )
    if thinking_effort and thinking_effort not in SUPPORTED_THINKING_EFFORTS:
        raise LLMConfigurationError(
            f"不支持的 LLM thinking_effort：{thinking_effort}"
        )
    if thinking_effort and api_format != "anthropic-messages":
        raise LLMConfigurationError(
            "thinking_effort 仅支持 anthropic-messages"
        )
    stream = resolved.get("stream", False)
    if not isinstance(stream, bool):
        raise LLMConfigurationError("LLM stream 必须是布尔值")
    if stream and api_format != "anthropic-messages":
        raise LLMConfigurationError("stream 目前仅支持 anthropic-messages")
    return ResolvedLLMConfig(
        provider=provider,
        requested_model=requested_model,
        model=model,
        base_url=base_url,
        timeout_seconds=max(1.0, float(resolved.get("timeout_seconds", 600))),
        max_retries=max(0, int(resolved.get("max_retries", 2))),
        retry_backoff_seconds=max(
            0.0,
            float(resolved.get("retry_backoff_seconds", 2)),
        ),
        api_format=api_format,
        thinking_effort=thinking_effort,
        stream=stream,
    )


def _api_key(provider: str) -> str:
    if provider == "volcengine":
        value = os.environ.get("ARK_API_KEY") or os.environ.get("LLM_API_KEY")
        expected_name = "ARK_API_KEY"
    elif provider == "dmx":
        value = os.environ.get("DMX_API_KEY")
        expected_name = "DMX_API_KEY"
    else:
        value = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        expected_name = "LLM_API_KEY 或 OPENAI_API_KEY"
    if not value or not value.strip():
        raise LLMConfigurationError(
            f"缺少 LLM API key 环境变量（{provider} 使用 {expected_name}）"
        )
    return value.strip()


_JSON_ESCAPE_CHARACTERS = frozenset('"\\/bfnrtu')


def _escape_invalid_json_string_backslashes(text: str) -> tuple[str, bool]:
    """转义 JSON 字符串中的非法单反斜杠，不改动合法 JSON escape。"""
    output: list[str] = []
    in_string = False
    changed = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            output.append(char)
            in_string = False
            index += 1
            continue
        if char != '\\':
            output.append(char)
            index += 1
            continue
        if index + 1 >= len(text):
            output.append(char)
            index += 1
            continue
        next_char = text[index + 1]
        if next_char in _JSON_ESCAPE_CHARACTERS:
            output.extend((char, next_char))
            index += 2
            continue
        output.append('\\\\')
        changed = True
        index += 1
    return ''.join(output), changed


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型响应文本解析 JSON 对象：先剥离 markdown 围栏，再回退到大括号切片。

    离线回放通道必须复用本函数。若回放直接 json.loads(raw content)，
    带围栏的响应会报出与真实截断完全相同的"不是完整 JSON"，
    使"围栏"与"截断"两种状态无法区分，污染失败归因。
    """
    stripped = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise LLMRequestError("LLM 响应中没有 JSON 对象")
        candidate = stripped[start:end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            repaired = candidate
            # 模型偶发在相邻字段间插入一个孤立双引号。仅修复两个
            # 已观测到的精确分隔符，且必须恰好出现一次；其他语法
            # 错误继续硬失败，避免掩盖截断或结构损坏。
            separator_repairs = (
                (',"},"confidence"', '},"confidence"'),
                (',","confidence"', ',"confidence"'),
            )
            changed = False
            for malformed, replacement in separator_repairs:
                if repaired.count(malformed) == 1:
                    repaired = repaired.replace(malformed, replacement, 1)
                    changed = True
            extra_point_close = '}}}]},{"confidence"'
            if extra_point_close in repaired:
                repaired = repaired.replace(
                    extra_point_close,
                    '}}]},{"confidence"',
                )
                changed = True
            missing_evidence_close = re.compile(
                r'("source_sentence"\s*:\s*"(?:\\.|[^"\\])*")'
                r'\s*,\s*"confidence"\s*:'
            )
            missing_close_matches = list(
                missing_evidence_close.finditer(repaired)
            )
            if missing_close_matches:
                repaired = missing_evidence_close.sub(
                    r'\1},"confidence":',
                    repaired,
                )
                changed = True
            repaired, invalid_backslashes_escaped = (
                _escape_invalid_json_string_backslashes(repaired)
            )
            changed = changed or invalid_backslashes_escaped
            if not changed:
                raise LLMRequestError("LLM 响应不是有效 JSON") from exc
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                raise LLMRequestError("LLM 响应不是有效 JSON") from exc
    if not isinstance(data, dict):
        raise LLMRequestError("LLM JSON 响应必须是对象")
    return data


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _parse_usage(
    body: dict[str, Any],
    api_format: str,
) -> LLMTokenUsage:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return LLMTokenUsage()
    if api_format == "anthropic-messages":
        return LLMTokenUsage(
            input_tokens=_nonnegative_int(usage.get("input_tokens")),
            output_tokens=_nonnegative_int(usage.get("output_tokens")),
            cache_creation_input_tokens=_nonnegative_int(
                usage.get("cache_creation_input_tokens")
            ),
            cache_read_input_tokens=_nonnegative_int(
                usage.get("cache_read_input_tokens")
            ),
        )
    return LLMTokenUsage(
        input_tokens=_nonnegative_int(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        ),
        output_tokens=_nonnegative_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
    )


def _read_anthropic_stream(response: requests.Response) -> dict[str, Any]:
    """把 Anthropic SSE 事件合并成与非流式响应一致的最小 body。"""

    model: str | None = None
    stop_reason: str | None = None
    text_parts: list[str] = []
    usage: dict[str, int] = {}
    for raw_line in response.iter_lines(decode_unicode=False):
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8")
        else:
            line = str(raw_line or "")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        event_type = event.get("type")
        if event_type == "error":
            error = event.get("error") or {}
            error_type = error.get("type") or "stream_error"
            raise LLMRequestError(f"LLM SSE 返回错误：{error_type}")
        if event_type == "message_start":
            message = event.get("message") or {}
            model = str(message.get("model") or "") or None
            initial_usage = message.get("usage") or {}
            if isinstance(initial_usage, dict):
                usage.update({
                    key: _nonnegative_int(value)
                    for key, value in initial_usage.items()
                })
        elif event_type == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "text" and isinstance(
                block.get("text"), str
            ):
                text_parts.append(block["text"])
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(
                delta.get("text"), str
            ):
                text_parts.append(delta["text"])
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason") is not None:
                stop_reason = str(delta["stop_reason"])
            final_usage = event.get("usage") or {}
            if isinstance(final_usage, dict):
                usage.update({
                    key: _nonnegative_int(value)
                    for key, value in final_usage.items()
                })
    return {
        "model": model,
        "content": [{"type": "text", "text": "".join(text_parts)}],
        "stop_reason": stop_reason,
        "usage": usage,
    }


def calculate_cost(
    usage: LLMTokenUsage,
    pricing: LLMPricing | None,
) -> LLMCallCost | None:
    if pricing is None:
        return None
    million = Decimal(1_000_000)
    input_cost = (
        Decimal(usage.billable_input_tokens)
        * pricing.input_per_million
        / million
    )
    output_cost = (
        Decimal(usage.output_tokens)
        * pricing.output_per_million
        / million
    )
    return LLMCallCost(
        currency=pricing.currency,
        input_per_million=pricing.input_per_million,
        output_per_million=pricing.output_per_million,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


class LLMClient:
    def __init__(
        self,
        resolved: ResolvedLLMConfig,
        *,
        session: requests.Session | None = None,
        api_key: str | None = None,
        pricing: LLMPricing | None = None,
    ) -> None:
        self.resolved = resolved
        self.session = session or requests.Session()
        self.api_key = api_key or _api_key(resolved.provider)
        self.pricing = pricing
        self.call_history: list[LLMCallRecord] = []
        self.last_response: LLMJSONResponse | None = None
        self.last_raw_response: LLMRawResponse | None = None

    @classmethod
    def from_pipeline_config(
        cls,
        *,
        stage: str,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> "LLMClient":
        config = load_pipeline_config(config_path)
        resolved = resolve_llm_config(config, stage)
        return cls(
            resolved,
            pricing=resolve_pricing_config(config, resolved.model),
        )

    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 4096,
    ) -> LLMJSONResponse:
        if self.resolved.api_format == "anthropic-messages":
            url = f"{self.resolved.base_url}/messages"
            payload = {
                "model": self.resolved.model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
                "max_tokens": max_tokens,
            }
            if self.resolved.stream:
                payload["stream"] = True
            if self.resolved.thinking_effort:
                payload["thinking"] = {
                    "type": "adaptive",
                    "display": "omitted",
                }
                payload["output_config"] = {
                    "effort": self.resolved.thinking_effort,
                }
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            }
        else:
            url = f"{self.resolved.base_url}/chat/completions"
            payload = {
                "model": self.resolved.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
            }
            if not (
                self.resolved.provider == "dmx"
                and self.resolved.model.casefold().startswith(
                    "claude-sonnet-5"
                )
            ):
                payload["temperature"] = 0
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

        last_error: Exception | None = None
        for attempt in range(self.resolved.max_retries + 1):
            try:
                request_kwargs = {
                    "headers": headers,
                    "json": payload,
                    "timeout": self.resolved.timeout_seconds,
                }
                if self.resolved.stream:
                    request_kwargs["stream"] = True
                response = self.session.post(url, **request_kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise LLMRequestError(f"LLM API 返回 HTTP {response.status_code}")
                if self.resolved.stream:
                    try:
                        body = _read_anthropic_stream(response)
                    finally:
                        response.close()
                else:
                    body = response.json()
                usage = _parse_usage(body, self.resolved.api_format)
                usage_available = isinstance(body.get("usage"), dict)
                cost = (
                    calculate_cost(usage, self.pricing)
                    if usage_available
                    else None
                )
                actual_model = str(
                    body.get("model") or self.resolved.model
                )
                self.call_history.append(LLMCallRecord(
                    provider=self.resolved.provider,
                    model=actual_model,
                    usage=usage,
                    cost=cost,
                    usage_available=usage_available,
                ))
                if self.resolved.api_format == "anthropic-messages":
                    blocks = body.get("content") or []
                    content = "\n".join(
                        str(block.get("text"))
                        for block in blocks
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    )
                    finish_reason = body.get("stop_reason")
                else:
                    choices = body.get("choices") or []
                    choice = (
                        choices[0]
                        if choices and isinstance(choices[0], dict)
                        else {}
                    )
                    message = choice.get("message") or {}
                    content = (
                        message.get("content")
                        if isinstance(message, dict)
                        else None
                    )
                    finish_reason = choice.get("finish_reason")
                self.last_raw_response = LLMRawResponse(
                    provider=self.resolved.provider,
                    model=actual_model,
                    finish_reason=(
                        str(finish_reason)
                        if finish_reason is not None
                        else None
                    ),
                    content=content if isinstance(content, str) else None,
                    usage=usage,
                    cost=cost,
                )
                if finish_reason in {"max_tokens", "length"}:
                    raise LLMOutputTruncatedError(
                        "LLM 响应达到输出长度限制"
                        f"（finish_reason={finish_reason}），"
                        "结构化 JSON 已截断"
                    )
                if not isinstance(content, str) or not content.strip():
                    raise LLMRequestError(
                        "LLM 响应缺少文本内容"
                        f"（api_format={self.resolved.api_format}, "
                        f"finish_reason={finish_reason!r}）"
                    )
                result = LLMJSONResponse(
                    data=extract_json_object(content),
                    provider=self.resolved.provider,
                    model=actual_model,
                    usage=usage,
                    cost=cost,
                )
                self.last_response = result
                return result
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= self.resolved.max_retries:
                    break
                time.sleep(self.resolved.retry_backoff_seconds * (2 ** attempt))
            except LLMRequestError:
                raise
        raise LLMRequestError(
            f"LLM 请求重试后失败（{type(last_error).__name__}）"
        ) from last_error


def llm_failure_artifact(
    client: LLMClient,
    *,
    stage: str,
    document_id: str,
    error: Exception,
    history_start: int = 0,
) -> dict[str, Any]:
    """生成不含请求与密钥的失败审计载荷。"""

    history = getattr(client, "call_history", [])
    call_count = max(0, len(history) - history_start)
    usage, cost = summarize_client_calls(
        client,
        history_start,
        call_count=call_count,
    )
    raw = getattr(client, "last_raw_response", None)
    raw_payload = None
    if raw is not None:
        raw_payload = {
            "provider": raw.provider,
            "model": raw.model,
            "finish_reason": raw.finish_reason,
            "content": raw.content,
            "usage": asdict(raw.usage),
            "cost": asdict(raw.cost) if raw.cost is not None else None,
        }
    payload = {
        "status": "failed",
        "stage": stage,
        "document_id": document_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "call_count": call_count,
        "usage": usage,
        "cost": cost,
        "raw_response": raw_payload,
    }
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        error_code = getattr(current, "error_code", None)
        if isinstance(error_code, str) and error_code:
            payload["error_code"] = error_code
            break
        current = current.__cause__ or current.__context__
    return _json_safe(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def summarize_client_calls(
    client: LLMClient,
    history_start: int,
    *,
    call_count: int,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """汇总一次文档处理期间的 token 和费用，不把缺失 usage 当作零费用。"""
    history = getattr(client, "call_history", None)
    pricing = getattr(client, "pricing", None)
    if not isinstance(history, list):
        return None, {
            "status": "unavailable",
            "currency": getattr(pricing, "currency", None),
            "input_per_million": getattr(
                pricing, "input_per_million", None
            ),
            "output_per_million": getattr(
                pricing, "output_per_million", None
            ),
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        }

    records = history[history_start:]
    if not records:
        if call_count > 0:
            status = "unavailable"
            amount: Decimal | None = None
        else:
            status = "not_applicable"
            amount = Decimal(0)
        usage = LLMTokenUsage()
        return asdict(usage) | {
            "billable_input_tokens": usage.billable_input_tokens,
            "total_tokens": usage.total_tokens,
        }, {
            "status": status,
            "currency": getattr(pricing, "currency", None),
            "input_per_million": getattr(
                pricing, "input_per_million", None
            ),
            "output_per_million": getattr(
                pricing, "output_per_million", None
            ),
            "input_cost": amount,
            "output_cost": amount,
            "total_cost": amount,
        }

    usage = LLMTokenUsage(
        input_tokens=sum(record.usage.input_tokens for record in records),
        output_tokens=sum(record.usage.output_tokens for record in records),
        cache_creation_input_tokens=sum(
            record.usage.cache_creation_input_tokens for record in records
        ),
        cache_read_input_tokens=sum(
            record.usage.cache_read_input_tokens for record in records
        ),
    )
    usage_payload = asdict(usage) | {
        "billable_input_tokens": usage.billable_input_tokens,
        "total_tokens": usage.total_tokens,
    }
    if any(
        not record.usage_available or record.cost is None
        for record in records
    ):
        return usage_payload, {
            "status": "unavailable",
            "currency": getattr(pricing, "currency", None),
            "input_per_million": getattr(
                pricing, "input_per_million", None
            ),
            "output_per_million": getattr(
                pricing, "output_per_million", None
            ),
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        }

    costs = [record.cost for record in records if record.cost is not None]
    first = costs[0]
    input_cost = sum(
        (cost.input_cost for cost in costs),
        start=Decimal(0),
    )
    output_cost = sum(
        (cost.output_cost for cost in costs),
        start=Decimal(0),
    )
    return usage_payload, {
        "status": "calculated",
        "currency": first.currency,
        "input_per_million": first.input_per_million,
        "output_per_million": first.output_per_million,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }
