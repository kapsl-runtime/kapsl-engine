#!/usr/bin/env python3
"""S3 sidecar connector implementing the kapsl-rag JSON protocol over stdio."""

from __future__ import annotations

import base64
import datetime as dt
import json
import mimetypes
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # pragma: no cover - handled at runtime
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception


METHOD_VALIDATE_CONFIG = "ValidateConfig"
METHOD_AUTH_START = "AuthStart"
METHOD_AUTH_CALLBACK = "AuthCallback"
METHOD_LIST_SOURCES = "ListSources"
METHOD_SYNC = "Sync"
METHOD_FETCH_DOCUMENT = "FetchDocument"
METHOD_RESOLVE_ACL = "ResolveAcl"
METHOD_HEALTH = "Health"


class ConnectorFailure(Exception):
    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class S3Config:
    bucket: str
    region: Optional[str] = None
    prefix: str = ""
    endpoint_url: Optional[str] = None
    profile: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    max_keys: int = 1000
    max_sync_documents: int = 1000
    include_extensions: List[str] = field(default_factory=list)
    force_path_style: bool = False

    def normalized_prefix(self) -> str:
        return self.prefix.lstrip("/")

    def effective_source_id(self) -> str:
        if self.source_id:
            return self.source_id
        prefix = self.normalized_prefix()
        return f"s3://{self.bucket}/{prefix}" if prefix else f"s3://{self.bucket}"

    def effective_source_name(self) -> str:
        if self.source_name:
            return self.source_name
        prefix = self.normalized_prefix()
        return f"S3 {self.bucket}/{prefix}" if prefix else f"S3 {self.bucket}"


class S3Connector:
    def __init__(self) -> None:
        self._config: Optional[S3Config] = None
        self._client: Any = None

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(request.get("id", ""))
        method = request.get("method")
        params = request.get("params") or {}

        try:
            if method == METHOD_VALIDATE_CONFIG:
                config = params.get("config")
                self.validate_config(config)
                return ok_response(request_id, "Unit")

            if method == METHOD_AUTH_START:
                raise ConnectorFailure(
                    "AuthStart is not implemented. Provide AWS credentials via config or environment.",
                    code="unsupported",
                )

            if method == METHOD_AUTH_CALLBACK:
                raise ConnectorFailure(
                    "AuthCallback is not implemented.",
                    code="unsupported",
                )

            if method == METHOD_LIST_SOURCES:
                config = params.get("config")
                sources = self.list_sources(config)
                return ok_response(request_id, "Sources", sources)

            if method == METHOD_SYNC:
                source_id = _as_non_empty_string(params.get("source_id"), "source_id")
                cursor = _as_optional_string(params.get("cursor"), "cursor")
                deltas = self.sync(source_id, cursor)
                return ok_response(request_id, "Deltas", deltas)

            if method == METHOD_FETCH_DOCUMENT:
                document_id = _as_non_empty_string(params.get("document_id"), "document_id")
                document = self.fetch_document(document_id)
                return ok_response(request_id, "Document", document)

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

    def validate_config(self, raw_config: Any) -> None:
        config = parse_config(raw_config)
        client = self._build_client(config)
        try:
            client.head_bucket(Bucket=config.bucket)
        except ClientError as err:
            raise ConnectorFailure(
                f"Unable to access bucket '{config.bucket}': {format_client_error(err)}",
                code="invalid_config",
            ) from err
        except BotoCoreError as err:
            raise ConnectorFailure(
                f"AWS SDK error while validating config: {err}",
                code="invalid_config",
            ) from err
        self._config = config
        self._client = client

    def list_sources(self, raw_config: Any) -> List[Dict[str, Any]]:
        config = parse_config(raw_config)
        self._config = config
        self._client = self._build_client(config)

        metadata: Dict[str, str] = {
            "bucket": config.bucket,
            "prefix": config.normalized_prefix(),
        }
        if config.region:
            metadata["region"] = config.region
        if config.endpoint_url:
            metadata["endpoint_url"] = config.endpoint_url

        return [
            {
                "id": config.effective_source_id(),
                "name": config.effective_source_name(),
                "kind": "s3",
                "metadata": metadata,
            }
        ]

    def sync(self, source_id: str, cursor: Optional[str]) -> List[Dict[str, Any]]:
        config, client = self._require_runtime()

        expected_source_id = config.effective_source_id()
        if source_id != expected_source_id:
            raise ConnectorFailure(
                f"Unknown source_id '{source_id}'. Expected '{expected_source_id}'.",
                code="invalid_input",
            )

        cursor_time = parse_cursor(cursor)
        normalized_prefix = config.normalized_prefix()

        paginator = client.get_paginator("list_objects_v2")
        paging_args: Dict[str, Any] = {
            "Bucket": config.bucket,
            "MaxKeys": config.max_keys,
        }
        if normalized_prefix:
            paging_args["Prefix"] = normalized_prefix

        deltas: List[Dict[str, Any]] = []

        try:
            for page in paginator.paginate(**paging_args):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if not isinstance(key, str) or not key or key.endswith("/"):
                        continue
                    if not should_include_key(key, config.include_extensions):
                        continue

                    modified_at = obj.get("LastModified")
                    modified_iso = to_iso8601(modified_at)
                    if cursor_time and modified_at:
                        modified_utc = ensure_utc(modified_at)
                        if modified_utc <= cursor_time:
                            continue

                    metadata = {
                        "bucket": config.bucket,
                        "key": key,
                        "size": str(obj.get("Size", "")),
                    }
                    if config.region:
                        metadata["region"] = config.region
                    if obj.get("StorageClass"):
                        metadata["storage_class"] = str(obj["StorageClass"])

                    deltas.append(
                        {
                            "id": key,
                            "op": "upsert",
                            "etag": normalize_etag(obj.get("ETag")),
                            "modified_at": modified_iso,
                            "metadata": metadata,
                            "acl": empty_acl(),
                        }
                    )

                    if len(deltas) >= config.max_sync_documents:
                        return deltas
        except ClientError as err:
            raise ConnectorFailure(
                f"Unable to list objects from '{config.bucket}': {format_client_error(err)}",
                code="runtime_error",
            ) from err
        except BotoCoreError as err:
            raise ConnectorFailure(
                f"AWS SDK error during sync: {err}",
                code="runtime_error",
            ) from err

        return deltas

    def fetch_document(self, document_id: str) -> Dict[str, Any]:
        config, client = self._require_runtime()

        try:
            response = client.get_object(Bucket=config.bucket, Key=document_id)
        except ClientError as err:
            raise ConnectorFailure(
                f"Unable to fetch object '{document_id}': {format_client_error(err)}",
                code="runtime_error",
            ) from err
        except BotoCoreError as err:
            raise ConnectorFailure(
                f"AWS SDK error while fetching '{document_id}': {err}",
                code="runtime_error",
            ) from err

        body = response.get("Body")
        if body is None:
            raise ConnectorFailure(
                f"Object '{document_id}' has no body in the S3 response",
                code="runtime_error",
            )

        try:
            content_bytes = body.read()
        except Exception as err:  # pragma: no cover - SDK specific behavior
            raise ConnectorFailure(
                f"Unable to read object '{document_id}' body: {err}",
                code="runtime_error",
            ) from err

        if not isinstance(content_bytes, (bytes, bytearray)):
            raise ConnectorFailure(
                f"Invalid object body type for '{document_id}'",
                code="runtime_error",
            )

        content_type = response.get("ContentType")
        if not isinstance(content_type, str) or not content_type:
            guessed, _ = mimetypes.guess_type(document_id)
            content_type = guessed or "application/octet-stream"

        metadata: Dict[str, str] = {
            "bucket": config.bucket,
            "key": document_id,
            "size": str(response.get("ContentLength", len(content_bytes))),
        }
        if response.get("ETag"):
            metadata["etag"] = normalize_etag(response.get("ETag")) or ""
        if response.get("VersionId"):
            metadata["version_id"] = str(response["VersionId"])
        if response.get("LastModified"):
            metadata["last_modified"] = to_iso8601(response.get("LastModified")) or ""

        return {
            "id": document_id,
            "content_type": content_type,
            "bytes_b64": base64.b64encode(bytes(content_bytes)).decode("ascii"),
            "metadata": metadata,
            "acl": empty_acl(),
        }

    def _build_client(self, config: S3Config) -> Any:
        if boto3 is None:
            raise ConnectorFailure(
                "Missing dependency 'boto3'. Install requirements before running the connector.",
                code="missing_dependency",
            )

        if bool(config.access_key_id) != bool(config.secret_access_key):
            raise ConnectorFailure(
                "Both access_key_id and secret_access_key must be provided together.",
                code="invalid_config",
            )

        session_kwargs: Dict[str, Any] = {}
        if config.profile:
            session_kwargs["profile_name"] = config.profile

        session = boto3.session.Session(**session_kwargs)

        client_kwargs: Dict[str, Any] = {}
        if config.region:
            client_kwargs["region_name"] = config.region
        if config.endpoint_url:
            client_kwargs["endpoint_url"] = config.endpoint_url
        if config.access_key_id and config.secret_access_key:
            client_kwargs["aws_access_key_id"] = config.access_key_id
            client_kwargs["aws_secret_access_key"] = config.secret_access_key
            if config.session_token:
                client_kwargs["aws_session_token"] = config.session_token
        if config.force_path_style and BotoConfig is not None:
            client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

        return session.client("s3", **client_kwargs)

    def _require_runtime(self) -> Tuple[S3Config, Any]:
        if self._config is None or self._client is None:
            raise ConnectorFailure(
                "Connector is not configured yet. Call ValidateConfig first.",
                code="invalid_state",
            )
        return self._config, self._client


def parse_config(raw_config: Any) -> S3Config:
    if not isinstance(raw_config, dict):
        raise ConnectorFailure("config must be an object", code="invalid_config")

    bucket = _as_non_empty_string(raw_config.get("bucket"), "bucket")
    region = _as_optional_string(raw_config.get("region"), "region")
    prefix = _as_optional_string(raw_config.get("prefix"), "prefix") or ""
    endpoint_url = _as_optional_string(raw_config.get("endpoint_url"), "endpoint_url")
    profile = _as_optional_string(raw_config.get("profile"), "profile")
    access_key_id = _as_optional_string(raw_config.get("access_key_id"), "access_key_id")
    secret_access_key = _as_optional_string(
        raw_config.get("secret_access_key"), "secret_access_key"
    )
    session_token = _as_optional_string(raw_config.get("session_token"), "session_token")
    source_id = _as_optional_string(raw_config.get("source_id"), "source_id")
    source_name = _as_optional_string(raw_config.get("source_name"), "source_name")

    max_keys = _as_optional_int(raw_config.get("max_keys"), "max_keys")
    if max_keys is None:
        max_keys = 1000
    if max_keys < 1 or max_keys > 1000:
        raise ConnectorFailure("max_keys must be between 1 and 1000", code="invalid_config")

    max_sync_documents = _as_optional_int(
        raw_config.get("max_sync_documents"), "max_sync_documents"
    )
    if max_sync_documents is None:
        max_sync_documents = 1000
    if max_sync_documents < 1:
        raise ConnectorFailure(
            "max_sync_documents must be at least 1", code="invalid_config"
        )

    force_path_style = bool(raw_config.get("force_path_style", False))

    include_extensions_raw = raw_config.get("include_extensions")
    include_extensions: List[str] = []
    if include_extensions_raw is not None:
        if not isinstance(include_extensions_raw, list):
            raise ConnectorFailure(
                "include_extensions must be an array of strings", code="invalid_config"
            )
        for item in include_extensions_raw:
            value = _as_non_empty_string(item, "include_extensions[]").lower()
            include_extensions.append(value if value.startswith(".") else f".{value}")

    return S3Config(
        bucket=bucket,
        region=region,
        prefix=prefix,
        endpoint_url=endpoint_url,
        profile=profile,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        source_id=source_id,
        source_name=source_name,
        max_keys=max_keys,
        max_sync_documents=max_sync_documents,
        include_extensions=include_extensions,
        force_path_style=force_path_style,
    )


def should_include_key(key: str, include_extensions: List[str]) -> bool:
    if not include_extensions:
        return True
    key_lower = key.lower()
    return any(key_lower.endswith(ext) for ext in include_extensions)


def normalize_etag(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip('"')


def parse_cursor(cursor: Optional[str]) -> Optional[dt.datetime]:
    if not cursor:
        return None

    text = cursor.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as err:
        raise ConnectorFailure(
            "cursor must be an ISO-8601 datetime string",
            code="invalid_input",
        ) from err

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)

    return parsed.astimezone(dt.timezone.utc)


def ensure_utc(value: Any) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise ConnectorFailure("S3 timestamp is invalid", code="runtime_error")
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def to_iso8601(value: Any) -> Optional[str]:
    if not isinstance(value, dt.datetime):
        return None
    utc_value = ensure_utc(value)
    return utc_value.isoformat().replace("+00:00", "Z")


def empty_acl() -> Dict[str, List[str]]:
    return {
        "allow_users": [],
        "allow_groups": [],
        "deny_users": [],
        "deny_groups": [],
    }


def normalize_acl(acl: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized = empty_acl()
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
    return {
        "id": request_id,
        "status": "Ok",
        "result": payload,
    }


def err_response(request_id: str, message: str, code: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": request_id,
        "status": "Err",
        "result": {
            "message": message,
            "code": code,
        },
    }


def format_client_error(err: Exception) -> str:
    if not hasattr(err, "response"):
        return str(err)

    try:
        response = getattr(err, "response") or {}
        error = response.get("Error") or {}
        code = error.get("Code", "Unknown")
        message = error.get("Message", str(err))
        return f"{code}: {message}"
    except Exception:
        return str(err)


def _as_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConnectorFailure(f"{field_name} must be a non-empty string", code="invalid_config")
    return value.strip()


def _as_optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConnectorFailure(f"{field_name} must be a string", code="invalid_config")
    stripped = value.strip()
    return stripped or None


def _as_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConnectorFailure(f"{field_name} must be an integer", code="invalid_config")
    return value


def main() -> int:
    connector = S3Connector()

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
