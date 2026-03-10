#!/usr/bin/env python3
"""Generate test payload JSON files for known image models."""

from __future__ import annotations

import argparse
import base64
import json
import struct
from pathlib import Path
from typing import Dict, List


def float32_bytes(value: float) -> bytes:
    return struct.pack("@f", value)


def build_payload(shape: List[int], fill: float) -> Dict[str, object]:
    elements = 1
    for dim in shape:
        elements *= dim

    data = float32_bytes(fill) * elements
    return {
        "input": {
            "shape": shape,
            "dtype": "float32",
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    }


def write_payload(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate payload JSONs for model tests."
    )
    parser.add_argument(
        "--out-dir",
        default="payloads",
        help="Output directory for generated payload files (default: payloads)",
    )
    parser.add_argument(
        "--fill",
        type=float,
        default=0.5,
        help="Float value used to fill tensors (default: 0.5)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    targets = [
        (0, "mnist", [1, 1, 28, 28]),
        (1, "squeezenet", [1, 3, 224, 224]),
    ]

    for model_id, label, shape in targets:
        payload = build_payload(shape, args.fill)
        path = out_dir / f"model_{model_id}_{label}.json"
        write_payload(path, payload)
        data_bytes = 4
        for dim in shape:
            data_bytes *= dim
        print(f"wrote {path} shape={shape} dtype=float32 data_bytes={data_bytes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
