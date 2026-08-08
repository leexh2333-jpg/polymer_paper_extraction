import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch


from llm_client import (
    DEFAULT_CONFIG_PATH,
    LLMClient,
    LLMOutputTruncatedError,
    LLMRequestError,
    LLMPricing,
    ResolvedLLMConfig,
    _api_key,
    extract_json_object,
    llm_config_cache_payload,
    llm_failure_artifact,
    load_pipeline_config,
    resolve_llm_config,
    resolve_pricing_config,
)


class LLMClientTests(unittest.TestCase):
    def test_extract_json_repairs_extra_quote_before_object_close(self) -> None:
        result = extract_json_object(
            '{"evidence":{"source_sentence":"text","},'
            '"confidence":{"score":0.9}}'
        )

        self.assertEqual(result["evidence"]["source_sentence"], "text")

    def test_extract_json_repairs_extra_quote_between_fields(self) -> None:
        result = extract_json_object(
            '{"measurement_context":null,","confidence":{"score":0.9}}'
        )

        self.assertIsNone(result["measurement_context"])

    def test_extract_json_repairs_missing_evidence_object_close(self) -> None:
        result = extract_json_object(
            '{"samples":[{"evidence":{"block_id":"P_1",'
            '"source_sentence":"text","confidence":{"score":0.9}}]}'
        )

        self.assertEqual(
            result["samples"][0]["evidence"]["source_sentence"],
            "text",
        )
        self.assertEqual(result["samples"][0]["confidence"]["score"], 0.9)

    def test_extract_json_repairs_multiple_missing_evidence_closes(self) -> None:
        result = extract_json_object(
            '{"samples":['
            '{"evidence":{"source_sentence":"one",'
            '"confidence":{"score":0.8}},'
            '{"evidence":{"source_sentence":"two",'
            '"confidence":{"score":0.9}}]}'
        )

        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(result["samples"][1]["confidence"]["score"], 0.9)

    def test_extract_json_repairs_extra_point_evidence_close(self) -> None:
        result = extract_json_object(
            '{"points":[{"evidence":[{"table_locator":{}}}]},'
            '{"confidence":{"score":0.9}}]}'
        )

        self.assertEqual(len(result["points"]), 2)

    def test_extract_json_repairs_invalid_string_backslash(self) -> None:
        result = extract_json_object(
            r'{"text":"$\mathrm {x}$"}'
        )

        self.assertEqual(result["text"], r"$\mathrm {x}$")

    def test_extract_json_preserves_valid_escapes(self) -> None:
        result = extract_json_object(
            r'{"newline":"line\nnext","quote":"say \"hi\"",'
            r'"slash":"C:\\tmp"}'
        )

        self.assertEqual(result["newline"], "line\nnext")
        self.assertEqual(result["quote"], 'say "hi"')
        self.assertEqual(result["slash"], r"C:\tmp")

    def test_extract_json_does_not_hide_truncation(self) -> None:
        with self.assertRaises(LLMRequestError):
            extract_json_object(r'{"text":"\mathrm {x}"')

    def test_project_stage4_uses_single_long_request(self) -> None:
        config = load_pipeline_config(DEFAULT_CONFIG_PATH)
        resolved = resolve_llm_config(config, "stage4_property")
        pricing = resolve_pricing_config(config, resolved.model)
        configured_pricing = config["pricing"]["models"][resolved.model]

        self.assertIn(resolved.model, config["pricing"]["models"])
        self.assertEqual(resolved.timeout_seconds, 900)
        self.assertEqual(resolved.max_retries, 0)
        self.assertTrue(resolved.stream)
        self.assertEqual(
            config["stages"]["stage4_property"]["max_tokens"],
            128000,
        )
        self.assertEqual(
            pricing.input_per_million,
            Decimal(str(configured_pricing["input_per_million"])),
        )
        self.assertEqual(
            pricing.output_per_million,
            Decimal(str(configured_pricing["output_per_million"])),
        )
        self.assertEqual(
            config["stages"]["stage3_sample_process"]["max_tokens"],
            32768,
        )

        stage5 = resolve_llm_config(config, "stage5_characterization")
        self.assertIn(stage5.model, config["pricing"]["models"])
        self.assertEqual(stage5.timeout_seconds, 900)
        self.assertEqual(stage5.max_retries, 0)
        self.assertTrue(stage5.stream)

    def test_stage_override_inherits_default_fields(self) -> None:
        resolved = resolve_llm_config(
            {
                "llm": {
                    "default": {
                        "provider": "volcengine",
                        "model": "claude-sonnet-5",
                        "base_url": "https://example.test/v3",
                        "timeout_seconds": 12,
                        "max_retries": 1,
                        "retry_backoff_seconds": 0,
                    },
                    "stage_overrides": {
                        "meta_extract": {"max_retries": 0}
                    },
                }
            },
            "meta_extract",
        )

        self.assertEqual(resolved.provider, "volcengine")
        self.assertEqual(resolved.requested_model, "claude-sonnet-5")
        self.assertEqual(resolved.model, "claude-sonnet-5")
        self.assertEqual(resolved.timeout_seconds, 12)
        self.assertEqual(resolved.max_retries, 0)
        self.assertEqual(resolved.api_format, "openai-chat-completions")

    def test_call_json_does_not_expose_key_in_result(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "actual-model",
            "choices": [{"message": {"content": '```json\n{"title": "Demo"}\n```'}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        }
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="volcengine",
                requested_model="requested",
                model="resolved",
                base_url="https://example.test/v3",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
            ),
            session=session,
            api_key="secret",
            pricing=LLMPricing(
                currency="CNY",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("10"),
            ),
        )

        result = client.call_json("Return JSON", "source")

        self.assertEqual(result.data, {"title": "Demo"})
        self.assertEqual(result.model, "actual-model")
        self.assertFalse(hasattr(result, "api_key"))
        self.assertEqual(result.usage.input_tokens, 1000)
        self.assertEqual(result.usage.output_tokens, 100)
        self.assertEqual(result.cost.total_cost, Decimal("0.003"))
        self.assertEqual(len(client.call_history), 1)
        self.assertEqual(
            session.post.call_args.kwargs["json"]["temperature"],
            0,
        )

    def test_max_tokens_preserves_raw_response_usage_and_cost(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "actual-model",
            "choices": [{
                "message": {"content": '{"items": ['},
                "finish_reason": "max_tokens",
            }],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        }
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="test",
                requested_model="requested",
                model="resolved",
                base_url="https://example.test/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
            ),
            session=session,
            api_key="secret",
            pricing=LLMPricing(
                currency="CNY",
                input_per_million=Decimal("13.51"),
                output_per_million=Decimal("66.5"),
            ),
        )

        with self.assertRaises(LLMOutputTruncatedError) as raised:
            client.call_json("Return JSON", "source")

        self.assertEqual(client.last_raw_response.finish_reason, "max_tokens")
        self.assertEqual(client.last_raw_response.content, '{"items": [')
        artifact = llm_failure_artifact(
            client,
            stage="stage2_polymer_entity",
            document_id="reference_no_test",
            error=raised.exception,
        )
        self.assertEqual(artifact["error_code"], "output_truncated")
        self.assertEqual(artifact["call_count"], 1)
        self.assertEqual(artifact["usage"]["output_tokens"], 200)
        self.assertEqual(artifact["cost"]["status"], "calculated")
        self.assertEqual(
            artifact["raw_response"]["cost"]["output_per_million"],
            "66.5",
        )
        json.dumps(artifact)

    def test_length_finish_reason_is_output_truncated(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "actual-model",
            "choices": [{
                "message": {"content": '{"items": ['},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="test",
                requested_model="requested",
                model="resolved",
                base_url="https://example.test/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
            ),
            session=session,
            api_key="secret",
        )

        with self.assertRaises(LLMOutputTruncatedError):
            client.call_json("Return JSON", "source")

        self.assertEqual(client.last_raw_response.finish_reason, "length")
        self.assertEqual(client.last_raw_response.content, '{"items": [')

    def test_dmx_uses_dedicated_environment_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"DMX_API_KEY": "dmx-secret"},
            clear=True,
        ):
            self.assertEqual(_api_key("dmx"), "dmx-secret")

    def test_dmx_base_url_appends_chat_completions_once(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "actual-model",
            "choices": [{"message": {"content": '{"ok": true}'}}],
        }
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="dmx",
                requested_model="claude-sonnet-5",
                model="claude-sonnet-5",
                base_url="https://www.dmxapi.cn/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
            ),
            session=session,
            api_key="secret",
        )

        client.call_json("Return JSON", "source")

        self.assertEqual(
            session.post.call_args.args[0],
            "https://www.dmxapi.cn/v1/chat/completions",
        )
        self.assertNotIn(
            "temperature",
            session.post.call_args.kwargs["json"],
        )

    def test_dmx_anthropic_messages_request_and_response(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "claude-sonnet-5",
            "content": [
                {"type": "thinking", "thinking": "omitted"},
                {"type": "text", "text": '{"ok": true}'},
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 25,
            },
        }
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="dmx",
                requested_model="claude-sonnet-5",
                model="claude-sonnet-5",
                base_url="https://www.dmxapi.cn/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
                api_format="anthropic-messages",
                thinking_effort="low",
            ),
            session=session,
            api_key="secret",
            pricing=LLMPricing(
                currency="CNY",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("10"),
            ),
        )

        result = client.call_json("Return JSON", "source")

        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(result.usage.billable_input_tokens, 175)
        self.assertEqual(result.usage.total_tokens, 195)
        self.assertEqual(result.cost.total_cost, Decimal("0.00055"))
        self.assertEqual(
            session.post.call_args.args[0],
            "https://www.dmxapi.cn/v1/messages",
        )
        request = session.post.call_args.kwargs
        self.assertEqual(request["headers"]["Authorization"], "secret")
        self.assertEqual(request["json"]["system"], "Return JSON")
        self.assertEqual(
            request["json"]["messages"],
            [{"role": "user", "content": "source"}],
        )
        self.assertEqual(
            request["json"]["thinking"],
            {"type": "adaptive", "display": "omitted"},
        )
        self.assertEqual(
            request["json"]["output_config"],
            {"effort": "low"},
        )
        self.assertNotIn("temperature", request["json"])

    def test_dmx_anthropic_stream_merges_text_usage_and_cost(self) -> None:
        response = Mock(status_code=200)
        response.iter_lines.return_value = [line.encode("utf-8") for line in [
            'event: message_start',
            'data: {"type":"message_start","message":{"model":"claude-sonnet-5","usage":{"input_tokens":100,"cache_creation_input_tokens":50,"cache_read_input_tokens":25}}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hidden"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"{\\"value\\":\\"35 ± "}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"0.01°C\\"}"}}',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}',
            'data: {"type":"message_stop"}',
        ]]
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="dmx",
                requested_model="claude-sonnet-5",
                model="claude-sonnet-5",
                base_url="https://www.dmxapi.cn/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
                api_format="anthropic-messages",
                stream=True,
            ),
            session=session,
            api_key="secret",
            pricing=LLMPricing(
                currency="CNY",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("10"),
            ),
        )

        result = client.call_json("Return JSON", "source")

        self.assertEqual(result.data, {"value": "35 ± 0.01°C"})
        self.assertEqual(result.usage.billable_input_tokens, 175)
        self.assertEqual(result.usage.output_tokens, 20)
        self.assertEqual(result.cost.total_cost, Decimal("0.00055"))
        request = session.post.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertTrue(request["json"]["stream"])
        response.close.assert_called_once_with()
        self.assertIs(client.last_response, result)

    def test_dmx_anthropic_stream_rejects_error_event(self) -> None:
        response = Mock(status_code=200)
        response.iter_lines.return_value = [
            'data: {"type":"error","error":{"type":"overloaded_error"}}',
        ]
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="dmx",
                requested_model="claude-sonnet-5",
                model="claude-sonnet-5",
                base_url="https://www.dmxapi.cn/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
                api_format="anthropic-messages",
                stream=True,
            ),
            session=session,
            api_key="secret",
        )

        with self.assertRaisesRegex(LLMRequestError, "overloaded_error"):
            client.call_json("Return JSON", "source")
        response.close.assert_called_once_with()

    def test_dmx_anthropic_stream_reports_max_tokens(self) -> None:
        response = Mock(status_code=200)
        response.iter_lines.return_value = [
            'data: {"type":"message_start","message":{"model":"claude-sonnet-5","usage":{"input_tokens":10}}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"{\\"partial\\": true"}}',
            'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":5}}',
        ]
        session = Mock()
        session.post.return_value = response
        client = LLMClient(
            ResolvedLLMConfig(
                provider="dmx",
                requested_model="claude-sonnet-5",
                model="claude-sonnet-5",
                base_url="https://www.dmxapi.cn/v1",
                timeout_seconds=10,
                max_retries=0,
                retry_backoff_seconds=0,
                api_format="anthropic-messages",
                stream=True,
            ),
            session=session,
            api_key="secret",
        )

        with self.assertRaisesRegex(LLMRequestError, "max_tokens"):
            client.call_json("Return JSON", "source")
        self.assertEqual(client.call_history[0].usage.output_tokens, 5)

    def test_cache_payload_preserves_legacy_openai_hash(self) -> None:
        common = {
            "provider": "dmx",
            "requested_model": "claude-sonnet-5",
            "model": "claude-sonnet-5",
            "base_url": "https://www.dmxapi.cn/v1",
            "timeout_seconds": 10,
            "max_retries": 0,
            "retry_backoff_seconds": 0,
        }

        openai_payload = llm_config_cache_payload(
            ResolvedLLMConfig(**common)
        )
        anthropic_payload = llm_config_cache_payload(
            ResolvedLLMConfig(
                **common,
                api_format="anthropic-messages",
                thinking_effort="low",
            )
        )

        self.assertNotIn("api_format", openai_payload)
        self.assertNotIn("thinking_effort", openai_payload)
        self.assertNotIn("stream", openai_payload)
        self.assertEqual(
            anthropic_payload["api_format"],
            "anthropic-messages",
        )
        self.assertEqual(anthropic_payload["thinking_effort"], "low")

    def test_pricing_resolves_only_configured_model(self) -> None:
        pricing = resolve_pricing_config(
            {
                "pricing": {
                    "currency": "CNY",
                    "models": {
                        "claude-sonnet-5": {
                            "input_per_million": "2",
                            "output_per_million": "10",
                        }
                    },
                }
            },
            "claude-sonnet-5",
        )

        self.assertEqual(pricing.currency, "CNY")
        self.assertEqual(pricing.input_per_million, Decimal("2"))
        self.assertEqual(pricing.output_per_million, Decimal("10"))


if __name__ == "__main__":
    unittest.main()
