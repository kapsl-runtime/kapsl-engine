#!/usr/bin/env python3
import argparse
import base64
import os
import sys
import time

import requests

API_BASE = "http://localhost:9095/api/models"
MAX_GENERATION_TIMEOUT = 1000  # seconds


def get_auth_headers():
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def list_models():
    try:
        resp = requests.get(API_BASE, headers=get_auth_headers(), timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"Failed to query models: {exc}")
        return None


def select_model(models, preferred_keywords, fallback_keywords=None):
    preferred = None
    fallback = None
    for model in models:
        name = str(model.get("name", "")).lower()
        if preferred_keywords and all(k in name for k in preferred_keywords):
            preferred = model
            break
        if fallback_keywords and all(k in name for k in fallback_keywords):
            fallback = model
    return preferred or fallback


def get_llama_model_id(forced_id=None):
    if forced_id is not None:
        return forced_id, str(forced_id)

    models = list_models()
    if models is None:
        return None, None

    selected = select_model(models, ["llama"], ["llama-3.2", "it"])
    if not selected:
        print("No Llama model found active in kapsl-runtime.")
        print("Available models:", [m.get("name") for m in models])
        return None, None
    return selected["id"], selected.get("name", selected["id"])


def make_string_request(api_url, prompt, timeout=MAX_GENERATION_TIMEOUT):
    payload = {
        "input": {
            "shape": [1, 1],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        },
        "session_id": None,
    }

    try:
        resp = requests.post(
            api_url,
            json=payload,
            headers=get_auth_headers(),
            timeout=timeout,
        )
    except Exception as exc:
        print(f"Request failed: {exc}")
        return None, None

    if resp.status_code != 200:
        print(f"Server returned HTTP {resp.status_code}: {resp.text}")
        return None, resp

    try:
        result = resp.json()
    except Exception as exc:
        print(f"Failed to parse JSON response: {exc}")
        return None, resp

    dtype = result.get("dtype")
    if dtype != "string":
        print(f"Unexpected dtype in response: {dtype}. Full response: {result}")
        return None, resp

    data = result.get("data", [])
    if not isinstance(data, list):
        print("Unexpected 'data' format in response; expected list of byte ints.")
        return None, resp

    try:
        out_bytes = bytes(data)
        text = out_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"Failed to decode bytes to UTF-8: {exc}")
        return None, resp

    return text, resp


def run_once(api_url, prompt, timeout):
    print("AI:", end=" ", flush=True)
    start = time.time()
    text, resp = make_string_request(api_url, prompt, timeout=timeout)
    elapsed = time.time() - start

    if text is None:
        print("[NO RESPONSE]")
        if resp is not None:
            try:
                print("Raw response:", resp.text)
            except Exception:
                pass
        return 1

    print(text)
    print(f"\n[info] generation time: {elapsed:.2f}s")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Send chat prompts to a Llama model via kapsl-runtime."
    )
    parser.add_argument("--prompt", "-p", help="Single prompt to send; skips REPL.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=MAX_GENERATION_TIMEOUT,
        help="Request timeout in seconds.",
    )
    parser.add_argument("--model-id", type=int, help="Override auto-selected model id.")
    args = parser.parse_args()

    model_id, model_name = get_llama_model_id(args.model_id)
    if model_id is None:
        sys.exit(1)

    api_url = f"{API_BASE}/{model_id}/infer"
    print(f"Using model {model_name} (id={model_id}) at {api_url}")

    if args.prompt:
        sys.exit(run_once(api_url, args.prompt, args.timeout))

    print("Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            print()
            break

        if prompt.strip().lower() in ("exit", "quit"):
            break

        if not prompt.strip():
            continue

        run_once(api_url, prompt, args.timeout)


if __name__ == "__main__":
    main()
