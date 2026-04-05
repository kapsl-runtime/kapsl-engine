#!/usr/bin/env python3
"""Prompt formatting sidecar implementing the kapsl extension JSON protocol."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional


METHOD_VALIDATE_CONFIG = "ValidateConfig"
METHOD_AUTH_START = "AuthStart"
METHOD_AUTH_CALLBACK = "AuthCallback"
METHOD_LIST_SOURCES = "ListSources"
METHOD_SYNC = "Sync"
METHOD_FETCH_DOCUMENT = "FetchDocument"
METHOD_TRANSFORM_PROMPT = "TransformPrompt"
METHOD_RESOLVE_ACL = "ResolveAcl"
METHOD_HEALTH = "Health"

FORMAT_GEMMA = "gemma"
FORMAT_CHATML = "chatml"
FORMAT_LLAMA3 = "llama3"
FORMAT_CUSTOM = "custom"
VALID_FORMATS = {FORMAT_GEMMA, FORMAT_CHATML, FORMAT_LLAMA3, FORMAT_CUSTOM}


class ConnectorFailure(Exception):
    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class TransformConfig:
    format: str
    trim_input: bool = True
    bos_token: Optional[str] = None
    user_prefix: Optional[str] = None
    user_suffix: Optional[str] = None
    assistant_prefix: Optional[str] = None
    think_suffix: str = ""


class PromptTransformerConnector:
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(request.get("id", ""))
        method = request.get("method")
        params = request.get("params") or {}

        try:
            if method == METHOD_VALIDATE_CONFIG:
                parse_config(params.get("config"))
                return ok_response(request_id, "Unit")

            if method == METHOD_TRANSFORM_PROMPT:
                config = parse_config(params.get("config"))
                prompt = _as_non_empty_string(
                    params.get("prompt"), "prompt", code="invalid_input"
                )
                transformed = transform_prompt(config, prompt)
                return ok_response(
                    request_id,
                    "PromptTransform",
                    {"prompt": transformed},
                )

            if method == METHOD_AUTH_START:
                raise ConnectorFailure("AuthStart is not implemented.", code="unsupported")

            if method == METHOD_AUTH_CALLBACK:
                raise ConnectorFailure("AuthCallback is not implemented.", code="unsupported")

            if method == METHOD_LIST_SOURCES:
                raise ConnectorFailure("ListSources is not implemented.", code="unsupported")

            if method == METHOD_SYNC:
                raise ConnectorFailure("Sync is not implemented.", code="unsupported")

            if method == METHOD_FETCH_DOCUMENT:
                raise ConnectorFailure("FetchDocument is not implemented.", code="unsupported")

            if method == METHOD_RESOLVE_ACL:
                acl = params.get("acl")
                if not isinstance(acl, dict):
                    raise ConnectorFailure("acl must be an object", code="invalid_input")
                return ok_response(request_id, "Acl", normalize_acl(acl))

            if method == METHOD_HEALTH:
                return ok_response(request_id, "Health", "ok")

            raise ConnectorFailure(f"Unsupported method: {method}", code="unsupported")
        except ConnectorFailure as err:
            return err_response(request_id, err.message, err.code)
        except Exception as err:  # pragma: no cover - defensive fallback
            return err_response(request_id, f"Internal error: {err}", code="internal")


def parse_config(raw_config: Any) -> TransformConfig:
    if not isinstance(raw_config, dict):
        raise ConnectorFailure("config must be an object", code="invalid_config")

    format_name = _as_non_empty_string(raw_config.get("format"), "format")
    if format_name not in VALID_FORMATS:
        raise ConnectorFailure(
            f"format must be one of: {', '.join(sorted(VALID_FORMATS))}",
            code="invalid_config",
        )

    trim_input = raw_config.get("trim_input", True)
    if not isinstance(trim_input, bool):
        raise ConnectorFailure("trim_input must be a boolean", code="invalid_config")

    config = TransformConfig(
        format=format_name,
        trim_input=trim_input,
        bos_token=_as_optional_string(raw_config.get("bos_token"), "bos_token"),
        user_prefix=_as_optional_string(raw_config.get("user_prefix"), "user_prefix"),
        user_suffix=_as_optional_string(raw_config.get("user_suffix"), "user_suffix"),
        assistant_prefix=_as_optional_string(
            raw_config.get("assistant_prefix"), "assistant_prefix"
        ),
        think_suffix=_as_optional_string(raw_config.get("think_suffix"), "think_suffix") or "",
    )

    if config.format == FORMAT_CUSTOM:
        if not config.user_prefix:
            raise ConnectorFailure(
                "user_prefix is required when format=custom",
                code="invalid_config",
            )
        if config.user_suffix is None:
            raise ConnectorFailure(
                "user_suffix is required when format=custom",
                code="invalid_config",
            )
        if not config.assistant_prefix:
            raise ConnectorFailure(
                "assistant_prefix is required when format=custom",
                code="invalid_config",
            )

    return config


def transform_prompt(config: TransformConfig, prompt: str) -> str:
    content = prompt.strip() if config.trim_input else prompt
    prefix, suffix = build_template(config)
    if content.startswith(prefix) and content.endswith(suffix):
        return content
    return f"{prefix}{content}{suffix}"


def build_template(config: TransformConfig) -> tuple[str, str]:
    if config.format == FORMAT_GEMMA:
        bos_token = config.bos_token or "<bos>"
        return (
            f"{bos_token}\n<start_of_turn>user\n",
            "<end_of_turn>\n<start_of_turn>model\n",
        )

    if config.format == FORMAT_CHATML:
        bos_token = config.bos_token or ""
        return (
            f"{bos_token}<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        )

    if config.format == FORMAT_LLAMA3:
        bos_token = config.bos_token or "<|begin_of_text|>"
        return (
            f"{bos_token}<|start_header_id|>user<|end_header_id|>\n\n",
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    bos_token = config.bos_token or ""
    prefix = f"{bos_token}{config.user_prefix or ''}"
    suffix = f"{config.user_suffix or ''}{config.assistant_prefix or ''}{config.think_suffix}"
    return prefix, suffix


def normalize_acl(acl: Dict[str, Any]) -> Dict[str, list[str]]:
    normalized = {
        "allow_users": [],
        "allow_groups": [],
        "deny_users": [],
        "deny_groups": [],
    }
    for key in normalized:
        value = acl.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ConnectorFailure(f"acl.{key} must be an array", code="invalid_input")
        normalized[key] = [str(item) for item in value]
    return normalized


def ok_response(request_id: str, result_type: str, value: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": result_type}
    if result_type != "Unit":
        payload["value"] = value
    return {"id": request_id, "status": "Ok", "result": payload}


def err_response(request_id: str, message: str, code: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": request_id,
        "status": "Err",
        "result": {"message": message, "code": code},
    }


def _as_non_empty_string(value: Any, field_name: str, code: str = "invalid_config") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConnectorFailure(f"{field_name} must be a non-empty string", code=code)
    return value.strip()


def _as_optional_string(
    value: Any,
    field_name: str,
    code: str = "invalid_config",
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConnectorFailure(f"{field_name} must be a string", code=code)
    stripped = value.strip()
    return stripped or None


def main() -> int:
    connector = PromptTransformerConnector()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as err:
            response = err_response("", f"Invalid JSON request: {err}", code="invalid_json")
        else:
            response = connector.handle(request)

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
