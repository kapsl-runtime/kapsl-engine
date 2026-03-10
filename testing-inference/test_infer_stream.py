#!/usr/bin/env python3
"""
Inference stream test harness using `KapslClient.infer_stream`.

Features:
- string or uint32 token input modes
- repeated runs for latency sampling
- validation gates (min chunks, max latency, non-empty output)
- optional expected substring checks for string output
"""

import argparse
import multiprocessing as mp
import struct
import sys
import time
from typing import List, Optional

try:
    from kapsl_runtime import KapslClient
except ImportError:
    print("❌ kapsl_runtime module not found.")
    print(
        "   Build it with: cd kapsl-runtime/crates/kapsl-pyo3 && maturin develop --release"
    )
    sys.exit(1)


def decode_chunk(chunk):
    if isinstance(chunk, bytes):
        return chunk
    return bytes(chunk)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return sorted(values)[rank]


def parse_token_ids(value):
    tokens = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    if not tokens:
        raise ValueError("no token ids parsed")
    return tokens


def build_request(prompt: str, token_ids: Optional[List[int]]):
    if token_ids is not None:
        dtype = "uint32"
        data_bytes = b"".join(struct.pack("<I", token) for token in token_ids)
        shape = [1, len(token_ids)]
    else:
        dtype = "string"
        data_bytes = prompt.encode("utf-8")
        shape = [1, len(data_bytes)]
    return dtype, shape, data_bytes


def run_stream_once(client, model_id, dtype, shape, data_bytes, verbose):
    if dtype == "uint32":
        if verbose:
            print("Streaming token ids: ", end="", flush=True)
    else:
        if verbose:
            print("Streaming text: ", end="", flush=True)

    start = time.perf_counter()
    first_chunk_ms = None
    chunks = []
    tokens_out = []

    for chunk in client.infer_stream(model_id, shape, dtype, data_bytes):
        chunk_bytes = decode_chunk(chunk)
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - start) * 1000
        chunks.append(chunk_bytes)

        if dtype == "uint32":
            for i in range(0, len(chunk_bytes), 4):
                if i + 4 > len(chunk_bytes):
                    continue
                token_id = struct.unpack("<I", chunk_bytes[i : i + 4])[0]
                tokens_out.append(token_id)
                if verbose:
                    print(f"{token_id} ", end="", flush=True)
        else:
            text = chunk_bytes.decode("utf-8", errors="replace")
            if verbose:
                print(text, end="", flush=True)

    elapsed_ms = (time.perf_counter() - start) * 1000
    if verbose:
        print()

    output_bytes = b"".join(chunks)
    output_text = (
        output_bytes.decode("utf-8", errors="replace") if dtype == "string" else ""
    )
    return {
        "elapsed_ms": elapsed_ms,
        "first_chunk_ms": first_chunk_ms if first_chunk_ms is not None else elapsed_ms,
        "chunk_count": len(chunks),
        "output_bytes": output_bytes,
        "output_text": output_text,
        "tokens_out": tokens_out,
        "output_size": len(output_bytes),
    }


def run_stream_worker(queue, socket_path, model_id, dtype, shape, data_bytes, verbose):
    try:
        client = KapslClient(socket_path)
        result = run_stream_once(client, model_id, dtype, shape, data_bytes, verbose)
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def run_stream_with_timeout(
    socket_path,
    model_id,
    dtype,
    shape,
    data_bytes,
    verbose,
    timeout_seconds,
):
    if timeout_seconds is None or timeout_seconds <= 0:
        client = KapslClient(socket_path)
        return run_stream_once(client, model_id, dtype, shape, data_bytes, verbose)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=run_stream_worker,
        args=(
            queue,
            socket_path,
            model_id,
            dtype,
            shape,
            data_bytes,
            verbose,
        ),
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        raise TimeoutError(
            "stream run exceeded timeout "
            f"({timeout_seconds}s) before completion; "
            "increase --run-timeout-s (for larger LLMs, try 240+ seconds)"
        )

    if queue.empty():
        raise RuntimeError(
            f"stream worker exited without result (exit_code={proc.exitcode})"
        )

    payload = queue.get()
    if payload.get("ok"):
        return payload["result"]
    raise RuntimeError(payload.get("error", "stream worker failed"))


def validate_result(result, dtype, min_chunks, max_latency_ms, expected_substrings):
    errors = []
    if result["chunk_count"] < min_chunks:
        errors.append(
            f"chunk_count={result['chunk_count']} is below required min_chunks={min_chunks}"
        )
    if result["output_size"] == 0:
        errors.append("stream produced empty output")
    if max_latency_ms is not None and result["elapsed_ms"] > max_latency_ms:
        errors.append(
            f"latency {result['elapsed_ms']:.2f}ms exceeds max {max_latency_ms:.2f}ms"
        )

    if dtype == "uint32":
        if len(result["tokens_out"]) == 0:
            errors.append("no tokens decoded from uint32 stream output")
    else:
        if not result["output_text"].strip():
            errors.append("string stream output is blank")
        for needle in expected_substrings:
            if needle not in result["output_text"]:
                errors.append(f"expected substring not found: {needle!r}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Test streaming inference via KapslClient.infer_stream.",
    )
    parser.add_argument("--socket", default="/tmp/kapsl.sock")
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="Write a short haiku about a lighthouse.",
        help="Text prompt to send as UTF-8 bytes (dtype=string).",
    )
    parser.add_argument(
        "--token-ids",
        help="Comma-separated uint32 token ids (overrides --prompt).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of repeated stream requests to run.",
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=1,
        help="Fail if a run yields fewer chunks than this.",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        help="Optional latency SLO; fail any run exceeding this total latency.",
    )
    parser.add_argument(
        "--expect-substring",
        action="append",
        default=[],
        help="Require substring in string output (repeatable).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide streamed chunk text/token printing; show summaries only.",
    )
    parser.add_argument(
        "--run-timeout-s",
        type=float,
        default=240.0,
        help=(
            "Per-run hard timeout in seconds to avoid indefinite stalls. "
            "Set <=0 to disable."
        ),
    )

    args = parser.parse_args()
    if args.iterations < 1:
        print("❌ --iterations must be >= 1")
        sys.exit(1)
    if args.min_chunks < 1:
        print("❌ --min-chunks must be >= 1")
        sys.exit(1)

    token_ids = None
    if args.token_ids:
        try:
            token_ids = parse_token_ids(args.token_ids)
        except ValueError as exc:
            print(f"❌ Invalid --token-ids: {exc}")
            sys.exit(1)

    try:
        _ = KapslClient(args.socket)
    except Exception as exc:
        print(f"❌ Failed to create client: {exc}")
        sys.exit(1)

    dtype, shape, data_bytes = build_request(args.prompt, token_ids)
    if dtype == "uint32":
        print(f"Input tokens: {token_ids}")
    else:
        print(f"Input prompt: {args.prompt}")
    print(f"Model ID: {args.model_id}")
    print(
        "Validation config: "
        f"iterations={args.iterations}, min_chunks={args.min_chunks}, "
        f"max_latency_ms={args.max_latency_ms}, run_timeout_s={args.run_timeout_s}"
    )
    if args.quiet:
        print("Quiet mode enabled: streamed text/tokens are hidden.")
    if dtype == "string" and args.expect_substring:
        print(f"Expected substrings: {args.expect_substring}")

    latencies = []
    first_chunk_latencies = []
    any_failure = False

    for i in range(args.iterations):
        print(f"\n--- Run {i + 1}/{args.iterations} ---")
        try:
            if args.quiet:
                print("Waiting for stream result...")
            result = run_stream_with_timeout(
                args.socket,
                args.model_id,
                dtype,
                shape,
                data_bytes,
                verbose=not args.quiet,
                timeout_seconds=args.run_timeout_s,
            )
        except Exception as exc:
            print(f"❌ Streaming inference failed on run {i + 1}: {exc}")
            any_failure = True
            continue

        latencies.append(result["elapsed_ms"])
        first_chunk_latencies.append(result["first_chunk_ms"])

        print(
            "Run summary: "
            f"chunks={result['chunk_count']}, bytes={result['output_size']}, "
            f"first_chunk={result['first_chunk_ms']:.2f}ms, total={result['elapsed_ms']:.2f}ms"
        )
        if dtype == "uint32":
            print(f"Tokens decoded: {len(result['tokens_out'])}")
        elif args.quiet:
            preview = result["output_text"][:200].replace("\n", "\\n")
            print(f"Output preview: {preview}")

        errors = validate_result(
            result,
            dtype,
            args.min_chunks,
            args.max_latency_ms,
            args.expect_substring,
        )
        if errors:
            any_failure = True
            print("❌ Validation failed:")
            for err in errors:
                print(f"   - {err}")
        else:
            print("✅ Validation passed")

    if not latencies:
        print("\n❌ No successful stream runs completed.")
        sys.exit(1)

    print("\n=== Aggregate Summary ===")
    print(f"Runs: {len(latencies)} / {args.iterations} succeeded")
    print(
        "Latency total (ms): "
        f"min={min(latencies):.2f}, p50={percentile(latencies, 50):.2f}, "
        f"p95={percentile(latencies, 95):.2f}, max={max(latencies):.2f}"
    )
    print(
        "Time-to-first-chunk (ms): "
        f"min={min(first_chunk_latencies):.2f}, "
        f"p50={percentile(first_chunk_latencies, 50):.2f}, "
        f"p95={percentile(first_chunk_latencies, 95):.2f}, "
        f"max={max(first_chunk_latencies):.2f}"
    )

    if any_failure:
        print("\n❌ One or more stream runs failed validation.")
        sys.exit(1)
    print("\n✅ All stream runs passed.")


if __name__ == "__main__":
    main()
