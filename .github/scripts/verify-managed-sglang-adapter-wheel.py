#!/usr/bin/env python3
"""Fail closed unless a built Kapsl SGLang adapter wheel matches its release tuple."""

from __future__ import annotations

import argparse
import ast
import configparser
import email.parser
import pathlib
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=pathlib.Path, required=True)
    parser.add_argument("--adapter-version", required=True)
    parser.add_argument("--sglang-version", required=True)
    parser.add_argument("--kv-abi-major", type=int, required=True)
    parser.add_argument("--kv-abi-minor", type=int, required=True)
    return parser.parse_args()


def one(names: list[str], suffix: str, wheel: pathlib.Path) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"{wheel}: expected exactly one {suffix}, found {matches!r}")
    return matches[0]


def literal_assignment(source: str, name: str, path: str, wheel: pathlib.Path):
    tree = ast.parse(source, filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            if value is None:
                break
            return ast.literal_eval(value)
    raise SystemExit(f"{wheel}: {path} omits literal {name}")


def main() -> None:
    args = parse_args()
    package = "kapsl_sglang_adapter"
    with zipfile.ZipFile(args.wheel) as wheel:
        names = wheel.namelist()

        def read(suffix: str) -> tuple[str, str]:
            path = one(names, suffix, args.wheel)
            return path, wheel.read(path).decode("utf-8")

        metadata_path = one(names, ".dist-info/METADATA", args.wheel)
        metadata = email.parser.BytesParser().parsebytes(wheel.read(metadata_path))
        init_path, init_source = read(f"{package}/__init__.py")
        contract_path, contract_source = read(f"{package}/contract.py")
        capacity_path, capacity_source = read(f"{package}/capacity.py")
        config_path, config_source = read(f"{package}/participant_config.py")
        one(names, f"{package}/launch.py", args.wheel)
        one(names, f"{package}/hooks.py", args.wheel)
        _, entry_points_source = read(".dist-info/entry_points.txt")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(entry_points_source)

    actual = {
        "distribution": metadata.get("Name"),
        "distribution_version": metadata.get("Version"),
        "adapter": literal_assignment(init_source, "ADAPTER_VERSION", init_path, args.wheel),
        "kv_abi": literal_assignment(contract_source, "ABI_VERSION", contract_path, args.wheel),
        "pins_sglang": args.sglang_version
        in literal_assignment(
            capacity_source, "PINNED_SGLANG_VERSIONS", capacity_path, args.wheel
        ),
        "participant_config_env": literal_assignment(
            config_source, "PARTICIPANT_CONFIG_ENV", config_path, args.wheel
        ),
        # SGLang loads plugins from this entry point group in every process.
        "plugin_entry_point": entry_points.get(
            "sglang.srt.plugins", "kapsl_kv", fallback=None
        ),
        # Kapsl starts SGLang through this module; it refuses to start an
        # ungoverned server.
        "launcher_entry_point": entry_points.get(
            "console_scripts", "kapsl-sglang-serve", fallback=None
        ),
    }
    expected = {
        "distribution": "kapsl-sglang-adapter",
        "distribution_version": args.adapter_version,
        "adapter": args.adapter_version,
        "kv_abi": {"major": args.kv_abi_major, "minor": args.kv_abi_minor},
        "pins_sglang": True,
        "participant_config_env": "KAPSL_KV_PARTICIPANT_CONFIG",
        "plugin_entry_point": "kapsl_sglang_adapter.hooks:activate",
        "launcher_entry_point": "kapsl_sglang_adapter.launch:main",
    }
    if actual != expected:
        raise SystemExit(f"Kapsl SGLang adapter wheel mismatch: {actual!r} != {expected!r}")


if __name__ == "__main__":
    main()
