#!/usr/bin/env python3
"""Fail closed unless a built managed-vLLM connector wheel matches its release tuple."""

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
    parser.add_argument("--connector-version", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--planner-schema", type=int, required=True)
    return parser.parse_args()


def one(names: list[str], suffix: str, wheel: pathlib.Path) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(
            f"{wheel}: expected exactly one {suffix}, found {matches!r}"
        )
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
    with zipfile.ZipFile(args.wheel) as wheel:
        names = wheel.namelist()
        metadata_path = one(names, ".dist-info/METADATA", args.wheel)
        metadata = email.parser.BytesParser().parsebytes(wheel.read(metadata_path))
        connector_path = one(
            names, "kapsl_vllm_connector/connector.py", args.wheel
        )
        planning_path = one(names, "kapsl_vllm_connector/planning.py", args.wheel)
        entry_points_path = one(names, ".dist-info/entry_points.txt", args.wheel)
        connector_source = wheel.read(connector_path).decode("utf-8")
        planning_source = wheel.read(planning_path).decode("utf-8")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(wheel.read(entry_points_path).decode("utf-8"))

    actual = {
        "distribution": metadata.get("Name"),
        "distribution_version": metadata.get("Version"),
        "connector": literal_assignment(
            connector_source, "ADAPTER_VERSION", connector_path, args.wheel
        ),
        "profile": literal_assignment(
            connector_source, "ADAPTER_PROFILE_ID", connector_path, args.wheel
        ),
        "planner_schema_version": literal_assignment(
            planning_source, "PLANNER_SCHEMA_VERSION", planning_path, args.wheel
        ),
        "planner_entry_point": entry_points.get(
            "console_scripts", "kapsl-vllm-plan", fallback=None
        ),
    }
    expected = {
        "distribution": "kapsl-vllm-connector",
        "distribution_version": args.connector_version,
        "connector": args.connector_version,
        "profile": args.profile,
        "planner_schema_version": args.planner_schema,
        "planner_entry_point": "kapsl_vllm_connector.plan:main",
    }
    if actual != expected:
        raise SystemExit(
            f"managed-vLLM connector wheel mismatch: {actual!r} != {expected!r}"
        )


if __name__ == "__main__":
    main()
