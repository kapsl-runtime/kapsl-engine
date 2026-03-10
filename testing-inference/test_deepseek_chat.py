import base64
import json
import os
import struct
import sys
import time

import numpy as np
import requests
from transformers import AutoTokenizer

# Configuration
# Configuration
API_BASE = "http://localhost:9095/api/models"
# Use a compatible tokenizer. Deepseek V2/V3 uses specific ones, but for general deepseek:
# If user has a specific local path, they can change this.
# Falling back to a standard one or attempting to load from local if valid.
# Using Qwen tokenizer as the model seems to be Qwen-based (Vocab 151936)
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 50


def get_auth_headers():
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_deepseek_model_id():
    try:
        resp = requests.get(API_BASE, headers=get_auth_headers())
        if resp.status_code == 200:
            models = resp.json()
            for m in models:
                if "deepseek" in m["name"].lower():
                    return m["id"]
            print("\n❌ No deepseek model found active in kapsl-runtime!")
            print("Available models:", [m["name"] for m in models])
            return None
    except Exception as e:
        print(f"❌ Failed to query models: {e}")
        return None
    return None


def get_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer from Hub: {e}")
        print("Please ensure you have internet access or the tokenizer is cached.")
        sys.exit(1)


def infer_next_token(input_ids):
    """
    Sends the input_ids to the model and returns the next token ID (greedy decoding).
    """
    # Shape: [1, seq_len]
    seq_len = len(input_ids)
    batch_size = 1

    # Convert to bytes (int64, little-endian)
    data_bytes = bytearray()
    for token_id in input_ids:
        data_bytes.extend(struct.pack("<q", token_id))

    # Create request payload matching test_deepseek_inference.py
    request_payload = {
        "input": {
            "shape": [batch_size, seq_len],
            "dtype": "int64",
            "data_base64": base64.b64encode(bytes(data_bytes)).decode("ascii"),
        },
        "session_id": None,
    }

    try:
        response = requests.post(
            API_URL,
            json=request_payload,
            headers=get_auth_headers(),
            timeout=30,
        )

        if response.status_code != 200:
            print(f"\nError: {response.text}")
            return None

        result = response.json()

        # Verify output
        if result.get("dtype") != "float32":
            print(f"\nUnexpected dtype: {result.get('dtype')}")
            return None

        shape = result["shape"]
        data = result["data"]  # This is a list of byte values (integers)

        # Convert list of bytes back to float32
        data_bytes = bytes(data)
        logits = np.frombuffer(data_bytes, dtype=np.float32)
        logits = logits.reshape(shape)
        print(f"[Debug] Logits Shape: {logits.shape}")

        # Get logits for the LAST token
        # Shape: [batch, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :].copy()

        # Repetition Penalty (Simple)
        for token_id in set(input_ids):
            if next_token_logits[token_id] > 0:
                next_token_logits[token_id] /= 1.2
            else:
                next_token_logits[token_id] *= 1.2

        # Debug: Top 5
        top_k = 5
        top_indices = np.argsort(next_token_logits)[-top_k:][::-1]
        top_probs = next_token_logits[top_indices]
        print(f"\n[Debug] Top {top_k} logits: {list(zip(top_indices, top_probs))}")

        next_token_id = np.argmax(next_token_logits)

        return next_token_id

    except Exception as e:
        print(f"\nException during inference: {e}")
        return None


def main():
    print("\n" + "=" * 50)
    print("🤖 Deepseek Chat Interface")
    print("=" * 50)

    # Find model
    model_name = "Unknown"
    model_id = None

    try:
        resp = requests.get(API_BASE, headers=get_auth_headers())
        if resp.status_code == 200:
            models = resp.json()
            for m in models:
                if "deepseek" in m["name"].lower():
                    model_id = m["id"]
                    model_name = m["name"]
                    break
    except:
        pass

    if model_id is None:
        print("❌ No deepseek model found active in kapsl-runtime!")
        sys.exit(1)

    global API_URL
    API_URL = f"{API_BASE}/{model_id}/infer"
    print(f"🎯 Using Model: {model_name} (ID: {model_id})")

    tokenizer = get_tokenizer()
    print(f"📚 Tokenizer vocab size: {tokenizer.vocab_size}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break

        if prompt.strip().lower() in ["exit", "quit"]:
            break

        print("AI: ", end="", flush=True)

        if prompt.strip() == "TEST_FIXED":
            print("Sending fixed tokens [15496, 1234]...")
            input_ids = [15496, 1234]
            print(f"[Debug] Decoded FIXED: '{tokenizer.decode(input_ids)}'")
        else:
            # Encode with chat template if available
            try:
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                print("[Debug] Used apply_chat_template")
            except Exception as e:
                print(f"[Debug] Chat template failed ({e}), using raw encode")
                input_ids = tokenizer.encode(prompt)

        # Sanitize Input: Remove tokens >= vocab_size
        # vocab_size = tokenizer.vocab_size
        # start_count = len(input_ids)
        # input_ids = [id for id in input_ids if id < vocab_size]
        # if len(input_ids) < start_count:
        #     print(f"[WARN] Removed {start_count - len(input_ids)} out-of-bounds tokens (>= {vocab_size})")

        print(f"\n[Debug] Input IDs: {input_ids}")
        # print(f"[Debug] Max ID: {max(input_ids) if input_ids else 0}, Vocab Size: {vocab_size}")

        if not input_ids:
            print("Error: No valid input ids left!")
            continue

        # Generation Loop
        generated_ids = []

        for _ in range(MAX_NEW_TOKENS):
            # Prepare full context (stateless)
            current_context = input_ids + generated_ids

            next_token = infer_next_token(current_context)

            if next_token is None:
                break

            print(f"[Debug] Token ID: {next_token}", end=" -> ")

            generated_ids.append(next_token)

            token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            print(f"'{token_text}'")

            # Stop conditions
            if next_token == tokenizer.eos_token_id:
                print(" [EOS]")
                break

        print("\n")


if __name__ == "__main__":
    main()
