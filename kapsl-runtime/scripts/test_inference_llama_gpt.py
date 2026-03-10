#!/usr/bin/env python3
"""
Run a simple LLM inference smoke test for Llama and GPT packages.

Requires kapsl_runtime Python module:
  cd crates/kapsl-pyo3 && maturin develop --release
"""

import argparse
import sys
import time

try:
    from kapsl_runtime import KapslClient
except ImportError:
    print("❌ kapsl_runtime module not found.")
    print("   Build it with: cd crates/kapsl-pyo3 && maturin develop --release")
    sys.exit(1)


def decode_chunk(chunk):
    if isinstance(chunk, bytes):
        return chunk
    return bytes(chunk)


def run_infer(client, model_id, prompt, stream):
    data = prompt.encode("utf-8")
    shape = [1, len(data)]
    dtype = "string"

    if stream:
        start = time.perf_counter()
        chunks = []
        for token in client.infer_stream(
            model_id=model_id,
            shape=shape,
            dtype=dtype,
            data=data,
        ):
            chunks.append(decode_chunk(token))
        elapsed_ms = (time.perf_counter() - start) * 1000
        output = b"".join(chunks)
    else:
        start = time.perf_counter()
        output = client.infer(model_id, shape, dtype, data)
        elapsed_ms = (time.perf_counter() - start) * 1000

    text = output.decode("utf-8", errors="replace")
    return text, elapsed_ms


def run_case(client, name, model_id, prompt, stream):
    print(f"\n=== {name} (model_id={model_id}) ===")
    print(f"Prompt: {prompt}")
    try:
        text, elapsed_ms = run_infer(client, model_id, prompt, stream)
    except Exception as exc:
        print(f"❌ {name} inference failed: {exc}")
        return False

    print(f"Latency: {elapsed_ms:.2f}ms")
    print("Output:")
    print(text)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test Llama/GPT inference via kapsl_runtime.",
    )
    parser.add_argument("--socket", default="/tmp/kapsl.sock")
    parser.add_argument("--llama-id", type=int, default=0)
    parser.add_argument("--gpt-id", type=int, default=1)
    parser.add_argument(
        "--llama-prompt",
        default="Write a short haiku about a lighthouse.",
    )
    parser.add_argument(
        "--gpt-prompt",
        default="Summarize the benefits of unit tests in one sentence.",
    )
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--stream", action="store_true")

    args = parser.parse_args()

    client = KapslClient(args.socket)
    ok = True

    if not args.skip_llama:
        ok &= run_case(
            client,
            "Llama",
            args.llama_id,
            args.llama_prompt,
            args.stream,
        )

    if not args.skip_gpt:
        ok &= run_case(
            client,
            "GPT",
            args.gpt_id,
            args.gpt_prompt,
            args.stream,
        )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
