#!/usr/bin/env python3
"""
RAG end-to-end test using the local-storage connector.

This script:
1) Creates a small temp document to index.
2) Installs and configures the local-storage RAG extension.
3) Launches and syncs the connector.
4) Queries /api/rag/query and validates results.

Optional: run an inference request with RAG enabled if --model-id is provided.
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://localhost:9095"


def get_auth_headers() -> dict:
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def http_post(url: str, payload: dict, timeout: int = 30) -> requests.Response:
    return requests.post(url, json=payload, headers=get_auth_headers(), timeout=timeout)


def http_get(url: str, timeout: int = 30) -> requests.Response:
    return requests.get(url, headers=get_auth_headers(), timeout=timeout)


def ensure_ok(
    resp: requests.Response,
    step_name: str,
    allow_already_installed: bool = False,
    allow_already_running: bool = False,
) -> None:
    if 200 <= resp.status_code < 300:
        return

    body = resp.text.strip()
    body_lower = body.lower()
    if allow_already_installed and "already" in body_lower:
        print(f"{step_name}: extension already installed, continuing.")
        return
    if allow_already_running and "already running" in body_lower:
        print(f"{step_name}: connector already running, continuing.")
        return

    print(f"{step_name} failed with status {resp.status_code}.")
    if body:
        print(body)
    sys.exit(1)


def make_temp_doc() -> tuple[str, str]:
    temp_root = tempfile.mkdtemp(prefix="kapsl-rag-test-")
    token = f"RAG_TEST_TOKEN_{int(time.time())}"
    doc_path = Path(temp_root) / "rag_test_doc.txt"
    doc_path.write_text(
        f"This is a RAG test document.\nToken: {token}\n",
        encoding="utf-8",
    )
    return temp_root, token


def query_has_token(response_json: dict, token: str) -> bool:
    matches = response_json.get("matches", [])
    for match in matches:
        text = match.get("text", "")
        if token in text:
            return True
    return False


def decode_string_tensor(response_json: dict) -> str | None:
    if response_json.get("dtype") != "string":
        return None
    data = response_json.get("data", [])
    if not isinstance(data, list):
        return None
    try:
        return bytes(data).decode("utf-8", errors="replace")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG test.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("KAPSL_BASE_URL")
        or os.getenv("KAPSL_BASE_URL")
        or DEFAULT_BASE_URL,
        help="Base URL for the kapsl runtime (default: http://localhost:9095)",
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Workspace ID to use for the test",
    )
    parser.add_argument(
        "--extension-path",
        default=str(
            Path(__file__).resolve().parents[1]
            / "extensions"
            / "local-storage-rag-extension"
        ),
        help="Path to local-storage RAG extension directory",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=None,
        help="Optional model ID to test /infer with RAG enabled",
    )
    parser.add_argument(
        "--print-rag-matches",
        action="store_true",
        help="Print RAG query matches",
    )
    parser.add_argument(
        "--print-infer-response",
        action="store_true",
        help="Print the raw /infer response JSON",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    workspace_id = (args.workspace_id or "").strip()

    print("RAG test starting.")
    resp = http_get(f"{base_url}/api/models")
    if not (200 <= resp.status_code < 300):
        print("API not reachable at /api/models.")
        print(resp.text)
        sys.exit(1)

    temp_root, token = make_temp_doc()
    if not workspace_id:
        workspace_id = f"rag-test-{int(time.time())}"
    print(f"Using temp root: {temp_root}")
    print(f"Token: {token}")
    print(f"Workspace ID: {workspace_id}")

    install_payload = {"path": args.extension_path}
    resp = http_post(f"{base_url}/api/extensions/install", install_payload)
    ensure_ok(resp, "Install extension", allow_already_installed=True)

    config_payload = {
        "workspace_id": workspace_id,
        "config": {
            "root_path": temp_root,
            "include_extensions": [".txt"],
            "max_sync_documents": 100,
            "follow_symlinks": False,
        },
    }
    resp = http_post(
        f"{base_url}/api/extensions/connector.local-storage/config", config_payload
    )
    ensure_ok(resp, "Configure extension")

    resp = http_post(
        f"{base_url}/api/extensions/connector.local-storage/launch",
        {"workspace_id": workspace_id},
    )
    ensure_ok(resp, "Launch extension", allow_already_running=True)

    resp = http_post(
        f"{base_url}/api/extensions/connector.local-storage/sync",
        {"workspace_id": workspace_id},
    )
    ensure_ok(resp, "Sync extension")

    time.sleep(0.5)

    query_payload = {
        "workspace_id": workspace_id,
        "query": token,
        "top_k": 4,
    }
    resp = http_post(f"{base_url}/api/rag/query", query_payload)
    ensure_ok(resp, "RAG query")

    response_json = resp.json()
    count = response_json.get("count", 0)
    print(f"Query returned {count} matches.")
    if args.print_rag_matches:
        print(json.dumps(response_json, indent=2))
    if not query_has_token(response_json, token):
        print("Token not found in RAG results.")
        print(json.dumps(response_json, indent=2))
        sys.exit(1)

    print("RAG query test passed.")

    if args.model_id is not None:
        prompt = f"Tell me about {token}."
        infer_payload = {
            "input": {
                "shape": [1, 1],
                "dtype": "string",
                "data_base64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
            },
            "session_id": "rag-test",
            "rag": {"workspace_id": workspace_id, "top_k": 4},
        }
        resp = http_post(
            f"{base_url}/api/models/{args.model_id}/infer",
            infer_payload,
            timeout=60,
        )
        ensure_ok(resp, "Infer with RAG")
        if args.print_infer_response:
            infer_json = resp.json()
            decoded = decode_string_tensor(infer_json)
            if decoded is not None:
                print("Decoded output:")
                print(decoded)
            print(json.dumps(infer_json, indent=2))
        print("Infer with RAG completed.")

    print("RAG test complete.")


if __name__ == "__main__":
    main()
