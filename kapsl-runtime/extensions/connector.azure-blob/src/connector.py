#!/usr/bin/env python3
"""Azure Blob Storage sidecar connector implementing the kapsl-rag JSON protocol."""

from __future__ import annotations

import base64
import datetime as dt
import json
import mimetypes
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from azure.core.exceptions import AzureError, HttpResponseError
    from azure.storage.blob import BlobServiceClient
except Exception:  # pragma: no cover - optional dependency guard
    AzureError = Exception
    HttpResponseError = Exception
    BlobServiceClient = None


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
class BlobConfig:
    container: str
    connection_string: Optional[str] = None
    account_url: Optional[str] = None
    credential: Optional[str] = None
    prefix: str = ""
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    include_extensions: List[str] = field(default_factory=list)
    max_sync_documents: int = 1000

    def normalized_prefix(self) -> str:
        return self.prefix.lstrip("/")

    def effective_source_id(self) -> str:
        if self.source_id:
            return self.source_id
        prefix = self.normalized_prefix()
        if prefix:
            return f"azureblob://{self.container}/{prefix}"
        return f"azureblob://{self.container}"

    def effective_source_name(self) -> str:
        if self.source_name:
            return self.source_name
        prefix = self.normalized_prefix()
        if prefix:
            return f"Azure Blob {self.container}/{prefix}"
        return f"Azure Blob {self.container}"


class AzureBlobConnector:
    def __init__(self) -> None:
        self._config: Optional[BlobConfig] = None
        self._container_client: Any = None

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
                    "AuthStart is not implemented. Use connection string, account key/SAS, or environment identity.",
                    code="unsupported",
                )

            if method == METHOD_AUTH_CALLBACK:
                raise ConnectorFailure("AuthCallback is not implemented.", code="unsupported")

            if method == METHOD_LIST_SOURCES:
                config = params.get("config")
                sources = self.list_sources(config)
                return ok_response(request_id, "Sources", sources)

            if method == METHOD_SYNC:
                source_id = _as_non_empty_string(
                    params.get("source_id"), "source_id", code="invalid_input"
                )
                cursor = _as_optional_string(
                    params.get("cursor"), "cursor", code="invalid_input"
                )
                deltas = self.sync(source_id, cursor)
                return ok_response(request_id, "Deltas", deltas)

            if method == METHOD_FETCH_DOCUMENT:
                document_id = _as_non_empty_string(
                    params.get("document_id"), "document_id", code="invalid_input"
                )
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
        container_client = self._build_container_client(config)

        try:
            container_client.get_container_properties()
        except HttpResponseError as err:
            raise ConnectorFailure(
                f"Unable to access container '{config.container}': {format_azure_error(err)}",
                code="invalid_config",
            ) from err
        except AzureError as err:
            raise ConnectorFailure(
                f"Azure SDK error while validating config: {err}",
                code="invalid_config",
            ) from err

        self._config = config
        self._container_client = container_client

    def list_sources(self, raw_config: Any) -> List[Dict[str, Any]]:
        config = parse_config(raw_config)

        metadata: Dict[str, str] = {
            "container": config.container,
            "prefix": config.normalized_prefix(),
        }
        if config.account_url:
            metadata["account_url"] = config.account_url

        return [
            {
                "id": config.effective_source_id(),
                "name": config.effective_source_name(),
                "kind": "azure-blob",
                "metadata": metadata,
            }
        ]

    def sync(self, source_id: str, cursor: Optional[str]) -> List[Dict[str, Any]]:
        config, container_client = self._require_runtime()

        expected_source_id = config.effective_source_id()
        if source_id != expected_source_id:
            raise ConnectorFailure(
                f"Unknown source_id '{source_id}'. Expected '{expected_source_id}'.",
                code="invalid_input",
            )

        cursor_time = parse_cursor(cursor)
        prefix = config.normalized_prefix() or None

        deltas: List[Dict[str, Any]] = []

        try:
            blob_iter = container_client.list_blobs(name_starts_with=prefix)
            for blob in blob_iter:
                name = getattr(blob, "name", None)
                if not isinstance(name, str) or not name:
                    continue
                if name.endswith("/"):
                    continue
                if not should_include_name(name, config.include_extensions):
                    continue

                modified = getattr(blob, "last_modified", None)
                modified_iso = to_iso8601(modified)

                if cursor_time and modified:
                    modified_utc = ensure_utc(modified)
                    if modified_utc <= cursor_time:
                        continue

                metadata = {
                    "container": config.container,
                    "blob_name": name,
                    "size": str(getattr(blob, "size", "")),
                }
                if config.account_url:
                    metadata["account_url"] = config.account_url

                deltas.append(
                    {
                        "id": name,
                        "op": "upsert",
                        "etag": normalize_etag(getattr(blob, "etag", None)),
                        "modified_at": modified_iso,
                        "metadata": metadata,
                        "acl": empty_acl(),
                    }
                )

                if len(deltas) >= config.max_sync_documents:
                    break
        except HttpResponseError as err:
            raise ConnectorFailure(
                f"Unable to list blobs from '{config.container}': {format_azure_error(err)}",
                code="runtime_error",
            ) from err
        except AzureError as err:
            raise ConnectorFailure(
                f"Azure SDK error during sync: {err}",
                code="runtime_error",
            ) from err

        return deltas

    def fetch_document(self, document_id: str) -> Dict[str, Any]:
        config, container_client = self._require_runtime()

        blob_client = container_client.get_blob_client(document_id)
        try:
            blob_properties = blob_client.get_blob_properties()
            data = blob_client.download_blob().readall()
        except HttpResponseError as err:
            raise ConnectorFailure(
                f"Unable to fetch blob '{document_id}': {format_azure_error(err)}",
                code="runtime_error",
            ) from err
        except AzureError as err:
            raise ConnectorFailure(
                f"Azure SDK error while fetching '{document_id}': {err}",
                code="runtime_error",
            ) from err

        if not isinstance(data, (bytes, bytearray)):
            raise ConnectorFailure(
                f"Invalid blob content type for '{document_id}'",
                code="runtime_error",
            )

        content_type = None
        content_settings = getattr(blob_properties, "content_settings", None)
        if content_settings is not None:
            content_type = getattr(content_settings, "content_type", None)
        if not isinstance(content_type, str) or not content_type:
            guessed, _ = mimetypes.guess_type(document_id)
            content_type = guessed or "application/octet-stream"

        metadata: Dict[str, str] = {
            "container": config.container,
            "blob_name": document_id,
            "size": str(getattr(blob_properties, "size", len(data))),
        }

        etag = normalize_etag(getattr(blob_properties, "etag", None))
        if etag:
            metadata["etag"] = etag

        last_modified = to_iso8601(getattr(blob_properties, "last_modified", None))
        if last_modified:
            metadata["last_modified"] = last_modified

        version_id = getattr(blob_properties, "version_id", None)
        if isinstance(version_id, str) and version_id:
            metadata["version_id"] = version_id

        return {
            "id": document_id,
            "content_type": content_type,
            "bytes_b64": base64.b64encode(bytes(data)).decode("ascii"),
            "metadata": metadata,
            "acl": empty_acl(),
        }

    def _build_container_client(self, config: BlobConfig) -> Any:
        if BlobServiceClient is None:
            raise ConnectorFailure(
                "Missing dependency 'azure-storage-blob'. Install requirements before running the connector.",
                code="missing_dependency",
            )

        if config.connection_string:
            service_client = BlobServiceClient.from_connection_string(config.connection_string)
        else:
            if not config.account_url:
                raise ConnectorFailure(
                    "Either connection_string or account_url must be provided.",
                    code="invalid_config",
                )
            if config.credential:
                service_client = BlobServiceClient(
                    account_url=config.account_url,
                    credential=config.credential,
                )
            else:
                service_client = BlobServiceClient(account_url=config.account_url)

        return service_client.get_container_client(config.container)

    def _require_runtime(self) -> Tuple[BlobConfig, Any]:
        if self._config is None or self._container_client is None:
            raise ConnectorFailure(
                "Connector is not configured yet. Call ValidateConfig first.",
                code="invalid_state",
            )
        return self._config, self._container_client


def parse_config(raw_config: Any) -> BlobConfig:
    if not isinstance(raw_config, dict):
        raise ConnectorFailure("config must be an object", code="invalid_config")

    container = _as_non_empty_string(raw_config.get("container"), "container")

    connection_string = _as_optional_string(
        raw_config.get("connection_string"), "connection_string"
    )
    account_url = _as_optional_string(raw_config.get("account_url"), "account_url")
    credential = _as_optional_string(raw_config.get("credential"), "credential")

    if not connection_string and not account_url:
        raise ConnectorFailure(
            "Either connection_string or account_url must be provided.",
            code="invalid_config",
        )

    prefix = _as_optional_string(raw_config.get("prefix"), "prefix") or ""
    source_id = _as_optional_string(raw_config.get("source_id"), "source_id")
    source_name = _as_optional_string(raw_config.get("source_name"), "source_name")

    include_extensions_raw = raw_config.get("include_extensions")
    include_extensions: List[str] = []
    if include_extensions_raw is not None:
        if not isinstance(include_extensions_raw, list):
            raise ConnectorFailure(
                "include_extensions must be an array of strings",
                code="invalid_config",
            )
        for item in include_extensions_raw:
            value = _as_non_empty_string(item, "include_extensions[]").lower()
            include_extensions.append(value if value.startswith(".") else f".{value}")

    max_sync_documents = _as_optional_int(
        raw_config.get("max_sync_documents"), "max_sync_documents"
    )
    if max_sync_documents is None:
        max_sync_documents = 1000
    if max_sync_documents < 1:
        raise ConnectorFailure(
            "max_sync_documents must be at least 1", code="invalid_config"
        )

    return BlobConfig(
        container=container,
        connection_string=connection_string,
        account_url=account_url,
        credential=credential,
        prefix=prefix,
        source_id=source_id,
        source_name=source_name,
        include_extensions=include_extensions,
        max_sync_documents=max_sync_documents,
    )


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
        raise ConnectorFailure("timestamp value is invalid", code="runtime_error")
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def to_iso8601(value: Any) -> Optional[str]:
    if not isinstance(value, dt.datetime):
        return None
    return ensure_utc(value).isoformat().replace("+00:00", "Z")


def should_include_name(name: str, include_extensions: List[str]) -> bool:
    if not include_extensions:
        return True
    lower_name = name.lower()
    return any(lower_name.endswith(ext) for ext in include_extensions)


def normalize_etag(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip('"')


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


def format_azure_error(err: Exception) -> str:
    status_code = getattr(err, "status_code", None)
    reason = getattr(err, "reason", None)
    text = str(err)
    if status_code is not None and reason is not None:
        return f"{status_code} {reason}: {text}"
    if status_code is not None:
        return f"{status_code}: {text}"
    return text


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


def _as_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConnectorFailure(f"{field_name} must be an integer", code="invalid_config")
    return value


def main() -> int:
    connector = AzureBlobConnector()

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
