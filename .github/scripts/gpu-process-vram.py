#!/usr/bin/env python3
"""Attribute NVML memory, including an explicitly exclusive container GPU."""

import argparse
import csv
import io
import json
import pathlib


def rows(snapshot):
    result = []
    for row in csv.reader(io.StringIO(snapshot)):
        if not row:
            continue
        assert len(row) == 3, "invalid NVML process snapshot"
        gpu, pid, memory = (value.strip() for value in row)
        assert gpu.startswith("GPU-") and pid.isdecimal() and memory.isdecimal(), (
            "NVML process identity or memory is unavailable"
        )
        result.append((gpu, int(pid), int(memory) * 1024 * 1024))
    return result


def measure(baseline, current, runtime_pid, previous=None, exclusive=False):
    before, active = rows(baseline), rows(current)
    if exclusive:
        # NVML reports host PIDs on some container providers. This mode requires
        # a dedicated GPU with no processes before launch and exactly one now.
        assert not before, "exclusive GPU was already in use before launch"
        assert len(active) == 1, "exclusive GPU process identity is ambiguous"
        selected = active[0]
        method = "initially-idle-exclusive-gpu"
    else:
        matches = [row for row in active if row[1] == runtime_pid]
        assert len(matches) == 1, "runtime PID is absent or ambiguous in NVML"
        selected = matches[0]
        method = "runtime-pid"
    gpu, nvml_pid, memory = selected
    identity = {
        "runtime_pid": runtime_pid,
        "nvml_pid": nvml_pid,
        "gpu_uuid": gpu,
        "method": method,
    }
    assert previous is None or identity == previous, "GPU process identity changed"
    return identity, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--snapshot", required=True, type=pathlib.Path)
    parser.add_argument("--identity", required=True, type=pathlib.Path)
    parser.add_argument("--runtime-pid", required=True, type=int)
    parser.add_argument("--exclusive", action="store_true")
    args = parser.parse_args()
    previous = json.loads(args.identity.read_text()) if args.identity.exists() else None
    identity, memory = measure(
        args.baseline.read_text(),
        args.snapshot.read_text(),
        args.runtime_pid,
        previous,
        args.exclusive,
    )
    if previous is None:
        with args.identity.open("x") as output:
            output.write(json.dumps(identity, indent=2) + "\n")
    print(memory)


if __name__ == "__main__":
    main()
