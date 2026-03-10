#!/usr/bin/env python3
"""Local filesystem sidecar connector implementing the kapsl-rag JSON protocol."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import mimetypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
class LocalConfig:
    root_path: Path
    prefix: str = ""
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    include_extensions: List[str] = field(default_factory=list)
    max_sync_documents: int = 1000
    follow_symlinks: bool = False

    def normalized_prefix(self) -> str:
        return self.prefix.strip("/")

    def scan_root(self) -> Path:
        prefix = self.normalized_prefix()
        if not prefix:
            return self.root_path
        return self.root_path / prefix

    def effective_source_id(self) -> str:
        if self.source_id:
            return self.source_id
        prefix = self.normalized_prefix()
        if prefix:
            return f"local://{self.root_path}/{prefix}"
        return f"local://{self.root_path}"

    def effective_source_name(self) -> str:
        if self.source_name:
            return self.source_name
        prefix = self.normalized_prefix()
        if prefix:
            return f"Local {self.root_path}/{prefix}"
        return f"Local {self.root_path}"


class LocalStorageConnector:
    def __init__(self) -> None:
        self._config: Optional[LocalConfig] = None

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
                    "AuthStart is not implemented for local storage.",
                    code="unsupported",
                )

            if method == METHOD_AUTH_CALLBACK:
                raise ConnectorFailure(
                    "AuthCallback is not implemented for local storage.",
                    code="unsupported",
                )

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

        if not config.root_path.exists():
            raise ConnectorFailure(
                f"root_path does not exist: {config.root_path}",
                code="invalid_config",
            )
        if not config.root_path.is_dir():
            raise ConnectorFailure(
                f"root_path must be a directory: {config.root_path}",
                code="invalid_config",
            )

        scan_root = config.scan_root()
        if not scan_root.exists():
            raise ConnectorFailure(
                f"prefix path does not exist under root_path: {scan_root}",
                code="invalid_config",
            )
        if not scan_root.is_dir():
            raise ConnectorFailure(
                f"prefix path must resolve to a directory: {scan_root}",
                code="invalid_config",
            )

        self._config = config

    def list_sources(self, raw_config: Any) -> List[Dict[str, Any]]:
        config = parse_config(raw_config)

        metadata = {
            "root_path": str(config.root_path),
            "prefix": config.normalized_prefix(),
            "follow_symlinks": str(config.follow_symlinks).lower(),
        }

        return [
            {
                "id": config.effective_source_id(),
                "name": config.effective_source_name(),
                "kind": "local",
                "metadata": metadata,
            }
        ]

    def sync(self, source_id: str, cursor: Optional[str]) -> List[Dict[str, Any]]:
        config = self._require_runtime()
        expected_source_id = config.effective_source_id()

        if source_id != expected_source_id:
            raise ConnectorFailure(
                f"Unknown source_id '{source_id}'. Expected '{expected_source_id}'.",
                code="invalid_input",
            )

        cursor_time = parse_cursor(cursor)
        scan_root = config.scan_root()
        deltas: List[Dict[str, Any]] = []

        for dirpath, _, filenames in os.walk(scan_root, followlinks=config.follow_symlinks):
            dir_path = Path(dirpath)
            for filename in filenames:
                if not should_include_name(filename, config.include_extensions):
                    continue

                file_path = dir_path / filename
                try:
                    stat_info = file_path.stat()
                except OSError:
                    continue

                if not file_path.is_file():
                    continue

                modified = dt.datetime.fromtimestamp(stat_info.st_mtime, tz=dt.timezone.utc)
                if cursor_time and modified <= cursor_time:
                    continue

                document_id = to_document_id(config.root_path, file_path)
                metadata = {
                    "root_path": str(config.root_path),
                    "relative_path": document_id,
                    "size": str(stat_info.st_size),
                }

                deltas.append(
                    {
                        "id": document_id,
                        "op": "upsert",
                        "etag": file_etag(document_id, stat_info.st_mtime_ns, stat_info.st_size),
                        "modified_at": modified.isoformat().replace("+00:00", "Z"),
                        "metadata": metadata,
                        "acl": empty_acl(),
                    }
                )

                if len(deltas) >= config.max_sync_documents:
                    return deltas

        return deltas

    def fetch_document(self, document_id: str) -> Dict[str, Any]:
        config = self._require_runtime()
        file_path = resolve_document_path(config.root_path, document_id)

        if not file_path.exists() or not file_path.is_file():
            raise ConnectorFailure(
                f"Document not found: {document_id}",
                code="runtime_error",
            )

        try:
            data = file_path.read_bytes()
            stat_info = file_path.stat()
        except OSError as err:
            raise ConnectorFailure(
                f"Unable to read document '{document_id}': {err}",
                code="runtime_error",
            ) from err

        content_type, _ = mimetypes.guess_type(file_path.name)
        if not content_type:
            content_type = "application/octet-stream"

        modified = dt.datetime.fromtimestamp(stat_info.st_mtime, tz=dt.timezone.utc)

        metadata = {
            "root_path": str(config.root_path),
            "relative_path": document_id,
            "size": str(stat_info.st_size),
            "last_modified": modified.isoformat().replace("+00:00", "Z"),
            "etag": file_etag(document_id, stat_info.st_mtime_ns, stat_info.st_size),
        }

        return {
            "id": document_id,
            "content_type": content_type,
            "bytes_b64": base64.b64encode(data).decode("ascii"),
            "metadata": metadata,
            "acl": empty_acl(),
        }

    def _require_runtime(self) -> LocalConfig:
        if self._config is None:
            raise ConnectorFailure(
                "Connector is not configured yet. Call ValidateConfig first.",
                code="invalid_state",
            )
        return self._config


def parse_config(raw_config: Any) -> LocalConfig:
    if not isinstance(raw_config, dict):
        raise ConnectorFailure("config must be an object", code="invalid_config")

    root_raw = _as_non_empty_string(raw_config.get("root_path"), "root_path")
    root_path = Path(root_raw).expanduser().resolve()

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
            "max_sync_documents must be at least 1",
            code="invalid_config",
        )

    follow_symlinks = raw_config.get("follow_symlinks", False)
    if not isinstance(follow_symlinks, bool):
        raise ConnectorFailure(
            "follow_symlinks must be a boolean",
            code="invalid_config",
        )

    return LocalConfig(
        root_path=root_path,
        prefix=prefix,
        source_id=source_id,
        source_name=source_name,
        include_extensions=include_extensions,
        max_sync_documents=max_sync_documents,
        follow_symlinks=follow_symlinks,
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


def resolve_document_path(root_path: Path, document_id: str) -> Path:
    normalized = document_id.replace("\\", "/").strip("/")
    candidate = (root_path / normalized).resolve()

    root_resolved = root_path.resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ConnectorFailure(
            "document_id resolves outside root_path",
            code="invalid_input",
        )

    return candidate


def to_document_id(root_path: Path, file_path: Path) -> str:
    return file_path.resolve().relative_to(root_path.resolve()).as_posix()


def should_include_name(name: str, include_extensions: List[str]) -> bool:
    if not include_extensions:
        return True
    lower_name = name.lower()
    return any(lower_name.endswith(ext) for ext in include_extensions)


def file_etag(document_id: str, mtime_ns: int, size: int) -> str:
    payload = f"{document_id}:{mtime_ns}:{size}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


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
    connector = LocalStorageConnector()

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
