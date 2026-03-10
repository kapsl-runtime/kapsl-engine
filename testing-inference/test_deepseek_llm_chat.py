#!/usr/bin/env python3
import base64
import os
import sys
import time

import requests

API_BASE = "http://localhost:9095/api/models"
MAX_GENERATION_TIMEOUT = 1000  # seconds
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
EOS_TOKEN = "<\uff5cend▁of▁sentence\uff5c>"


def get_auth_headers():
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def wrap_prompt(prompt: str) -> str:
    if USER_TOKEN in prompt or ASSISTANT_TOKEN in prompt:
        return prompt
    return f"{USER_TOKEN}{prompt}{ASSISTANT_TOKEN}"


def get_deepseek_model_id():
    try:
        resp = requests.get(API_BASE, headers=get_auth_headers(), timeout=5)
        resp.raise_for_status()
        models = resp.json()
        for m in models:
            if "deepseek" in m.get("name", "").lower():
                return m["id"], m.get("name", m["id"])
        print("❌ No deepseek model found active in kapsl-runtime!")
        print("Available models:", [m.get("name") for m in models])
        return None, None
    except Exception as e:
        print(f"❌ Failed to query models: {e}")
        return None, None


def make_string_request(api_url, prompt, timeout=500):
    """
    Send the prompt as a UTF-8 string payload and return decoded text from the model.
    The runtime expects {"input": {"shape":[1,1], "dtype":"string", "data_base64": "..."}, ... }.
    """
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
    except Exception as e:
        print(f"Request failed: {e}")
        return None, resp if "resp" in locals() else None

    if resp.status_code != 200:
        print(f"Server returned HTTP {resp.status_code}: {resp.text}")
        return None, resp

    try:
        result = resp.json()
    except Exception as e:
        print(f"Failed to parse JSON response: {e}")
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
    except Exception as e:
        print(f"Failed to decode bytes to UTF-8: {e}")
        return None, resp

    return text, resp


def main():
    model_id, model_name = get_deepseek_model_id()
    if model_id is None:
        sys.exit(1)

    api_url = f"{API_BASE}/{model_id}/infer"
    print(f"Using model {model_name} (id={model_id}) at {api_url}")
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

        prompt = wrap_prompt(prompt)

        # Send the prompt as a string and wait for the model's response
        print("AI: ", end="", flush=True)
        start = time.time()
        text, resp = make_string_request(
            api_url, prompt, timeout=MAX_GENERATION_TIMEOUT
        )
        elapsed = time.time() - start

        if text is None:
            print("[NO RESPONSE]")
            # optionally show raw response text for debugging
            if resp is not None:
                try:
                    print("Raw response:", resp.text)
                except Exception:
                    pass
            continue

        print(text)
        print(f"\n[info] generation time: {elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
