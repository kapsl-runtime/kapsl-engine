#!/usr/bin/env python3
import argparse
import base64
import os
import struct
import sys
import time
import uuid
from pathlib import Path

import requests

API_BASE = "http://localhost:9095/api/models"
MAX_GENERATION_TIMEOUT = 2000  # seconds
DEFAULT_MAX_NEW_TOKENS = 1024
BOS_TOKEN = "<bos>"
START_TOKEN = "<start_of_turn>"
END_TOKEN = "<end_of_turn>"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMMA_KAPSL = REPO_ROOT / "kapsl-runtime/models/gemma-llm/gemma-3-4b-it.aimod"
DEFAULT_GEMMA_TOKENIZER = REPO_ROOT / "kapsl-runtime/models/gemma-llm/tokenizer.json"


def get_auth_headers():
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def wrap_prompt(prompt: str) -> str:
    if START_TOKEN in prompt or BOS_TOKEN in prompt:
        return prompt
    # Match Gemma tokenizer chat template shape:
    # <bos>\n<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
    content = prompt.strip()
    return f"{BOS_TOKEN}\n{START_TOKEN}user\n{content}{END_TOKEN}\n{START_TOKEN}model\n"


def get_gemma_model_id():
    try:
        resp = requests.get(API_BASE, headers=get_auth_headers(), timeout=5)
        resp.raise_for_status()
        models = resp.json()
    except Exception as exc:
        print(f"❌ Failed to query models: {exc}")
        return None, None

    preferred = None
    fallback = None
    for model in models:
        name = model.get("name", "").lower()
        if "gemma-3-4b-it" in name:
            preferred = model
            break
        if "gemma" in name:
            fallback = model

    selected = preferred or fallback
    if not selected:
        print("❌ No Gemma model found active in kapsl-runtime!")
        print("Available models:", [m.get("name") for m in models])
        if DEFAULT_GEMMA_KAPSL.exists():
            print("\nTip: start the runtime with Gemma loaded, for example:")
            print(f"  kapsl run --model {DEFAULT_GEMMA_KAPSL}")
            print("Or start it dynamically via POST /api/models/start with:")
            print(f'  {{"model_path":"{DEFAULT_GEMMA_KAPSL}"}}')
        return None, None

    return selected["id"], selected.get("name", selected["id"])


def build_generation_metadata(
    max_new_tokens=None,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=None,
    seed=None,
):
    metadata = {}
    if max_new_tokens is not None:
        metadata["max_new_tokens"] = int(max_new_tokens)
    if temperature is not None:
        metadata["temperature"] = float(temperature)
    if top_p is not None:
        metadata["top_p"] = float(top_p)
    if top_k is not None:
        metadata["top_k"] = int(top_k)
    if repetition_penalty is not None:
        metadata["repetition_penalty"] = float(repetition_penalty)
    if seed is not None:
        metadata["seed"] = int(seed)
    return metadata or None


def make_string_request(
    api_url,
    prompt,
    session_id,
    timeout=MAX_GENERATION_TIMEOUT,
    metadata=None,
):
    payload = {
        "input": {
            "shape": [1, 1],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        },
        "session_id": session_id,
    }
    if metadata is not None:
        payload["metadata"] = metadata

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


def make_int64_request(
    api_url,
    token_ids,
    session_id,
    timeout=MAX_GENERATION_TIMEOUT,
    metadata=None,
):
    data_bytes = bytearray()
    for token_id in token_ids:
        data_bytes.extend(struct.pack("<q", token_id))

    payload = {
        "input": {
            "shape": [1, len(token_ids)],
            "dtype": "int64",
            "data_base64": base64.b64encode(bytes(data_bytes)).decode("ascii"),
        },
        "session_id": session_id,
    }
    if metadata is not None:
        payload["metadata"] = metadata

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


def load_tokenizer(tokenizer_path):
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError(
            "Python package 'tokenizers' is required for --mode tokens."
        ) from exc

    return Tokenizer.from_file(tokenizer_path)


def tokenize_prompt(prompt, tokenizer_path):
    tokenizer = load_tokenizer(tokenizer_path)
    encoded = tokenizer.encode(prompt)
    return encoded.ids


def run_once(api_url, prompt, timeout, mode, tokenizer_path, session_id, metadata):
    print("AI:", end=" ", flush=True)
    start = time.time()
    if mode == "tokens":
        token_ids = tokenize_prompt(prompt, tokenizer_path)
        if not token_ids:
            print("[EMPTY TOKEN LIST]")
            return 1
        text, resp = make_int64_request(
            api_url,
            token_ids,
            session_id=session_id,
            timeout=timeout,
            metadata=metadata,
        )
    else:
        text, resp = make_string_request(
            api_url,
            prompt,
            session_id=session_id,
            timeout=timeout,
            metadata=metadata,
        )
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
        description="Send chat prompts to the Gemma 3 4B IT model via kapsl-runtime."
    )
    parser.add_argument("--prompt", "-p", help="Single prompt to send; skips REPL.")
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Send the prompt without the Gemma chat template.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MAX_GENERATION_TIMEOUT,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--mode",
        choices=("string", "tokens"),
        default="string",
        help="Send prompt as a UTF-8 string or pre-tokenized int64 ids.",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_GEMMA_TOKENIZER),
        help="Tokenizer path used when --mode tokens.",
    )
    parser.add_argument(
        "--session-id",
        help="Session ID for multi-turn memory. Defaults to a new ID each run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "LLM generation token budget sent in request metadata. "
            f"Default: {DEFAULT_MAX_NEW_TOKENS}."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, help="LLM sampling temperature override."
    )
    parser.add_argument(
        "--top-p", type=float, help="LLM nucleus sampling top-p override."
    )
    parser.add_argument("--top-k", type=int, help="LLM top-k sampling override.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="LLM repetition penalty override.",
    )
    parser.add_argument("--seed", type=int, help="LLM random seed override.")
    args = parser.parse_args()

    model_id, model_name = get_gemma_model_id()
    if model_id is None:
        sys.exit(1)

    api_url = f"{API_BASE}/{model_id}/infer"
    print(f"Using model {model_name} (id={model_id}) at {api_url}")
    session_id = args.session_id or f"chat-{uuid.uuid4().hex[:8]}"
    print(f"Using session_id={session_id}")
    metadata = build_generation_metadata(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    if metadata:
        print(f"Using metadata={metadata}")

    if args.prompt:
        prompt = args.prompt
        if not args.no_template:
            prompt = wrap_prompt(prompt)
        sys.exit(
            run_once(
                api_url,
                prompt,
                args.timeout,
                args.mode,
                args.tokenizer,
                session_id,
                metadata,
            )
        )

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

        if not args.no_template:
            prompt = wrap_prompt(prompt)

        run_once(
            api_url,
            prompt,
            args.timeout,
            args.mode,
            args.tokenizer,
            session_id,
            metadata,
        )


if __name__ == "__main__":
    main()
