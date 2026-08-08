"""测试假 LLM 响应的公共构造辅助。"""

from __future__ import annotations

from typing import Any


MODEL_OBJECT_LIST_KEYS = {
    "mentions",
    "entities",
    "samples",
    "process_steps",
    "measurement_conditions",
    "properties",
    "unresolved_properties",
    "property_series",
    "points",
    "characterizations",
}


def model_confidence(
    *,
    score: float = 0.9,
) -> dict[str, Any]:
    return {"score": score}


def add_model_confidence(payload: dict[str, Any]) -> dict[str, Any]:
    """递归给测试响应中的模型对象补合法 confidence。"""
    for key, value in payload.items():
        if key in MODEL_OBJECT_LIST_KEYS and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    item.setdefault("confidence", model_confidence())
        if isinstance(value, dict):
            add_model_confidence(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    add_model_confidence(item)
    return payload
