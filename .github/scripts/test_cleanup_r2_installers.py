#!/usr/bin/env python3
"""Execute the cleanup workflow against a fake bucket; never contact R2."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/cleanup-r2-installers.yml"
NOW = 1_800_000_000
ROOT = "s3://kapsl-installer/runtime/"

FAKE_COMMAND = r'''
import datetime
import json
import os
from pathlib import Path
import re
import sys

command = Path(sys.argv[0]).name
args = sys.argv[1:]
fixture = json.loads(Path(os.environ["FIXTURE"]).read_text())
if command == "date":
    value = args[args.index("-d") + 1]
    if value.endswith(" days ago"):
        print(fixture["now"] - int(value.split()[0]) * 86400)
    else:
        print(int(datetime.datetime.fromisoformat(value).timestamp()))
elif command == "sort":
    assert args == ["-V"], args
    lines = sys.stdin.read().splitlines()
    print("\n".join(sorted(lines, key=lambda s: [
        (1, int(p)) if p.isdigit() else (0, p) for p in re.split(r"(\d+)", s)
    ])))
elif args[:2] == ["s3", "ls"]:
    print("\n".join("PRE " + v + "/" for v in fixture["official"]))
elif args[:2] == ["s3", "cp"]:
    assert args[2:4] == ["s3://kapsl-installer/runtime/beta/latest.txt", "-"], args
    print(fixture["latest"])
elif args[:2] == ["s3api", "list-objects-v2"]:
    assert args[args.index("--prefix") + 1] == "runtime/beta/v", args
    if fixture.get("listing_failure"):
        sys.exit(1)
    for key, timestamp in fixture["objects"]:
        modified = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat()
        print(key + "\t" + modified)
elif args[:2] == ["s3", "rm"]:
    with open(os.environ["DELETIONS"], "a") as output:
        output.write(args[2] + "\n")
else:
    raise AssertionError(args)
'''


class CleanupTests(unittest.TestCase):
    def run_cleanup(self, *, keep="3", retention="1", latest="0.2.8-beta.1",
                    objects=None, listing_failure=False):
        if objects is None:
            objects = [
                (f"runtime/beta/v0.2.{n}-beta.1/installer.pkg", NOW - 10 * 86400)
                for n in range(6, 14)
            ]
            # A recent object protects an otherwise old, partially uploaded version.
            objects.append(("runtime/beta/v0.2.9-beta.1/installer.sha256", NOW))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({
                "now": NOW, "latest": latest, "objects": objects,
                "official": [f"v0.2.{n}" for n in range(1, 7)],
                "listing_failure": listing_failure,
            }))
            deletions = root / "deletions.txt"
            for name in ("aws", "date", "sort"):
                command = root / name
                command.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
                command.chmod(0o755)
            environment = os.environ | {
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "FIXTURE": str(fixture), "DELETIONS": str(deletions),
                "KEEP_LATEST": keep, "BETA_RETENTION_DAYS": retention,
                "R2_ENDPOINT": "https://example.invalid",
            }
            # Exercise every shell step, including any future top-level cleanup.
            steps = re.findall(r"(?m)^        run: \|\n((?:^          .*\n|^\n)+)",
                               WORKFLOW.read_text())
            self.assertTrue(steps)
            result = None
            for source in steps:
                result = subprocess.run(
                    [shutil.which("bash"), "-e", "-o", "pipefail", "-c", textwrap.dedent(source)],
                    env=environment, text=True, capture_output=True,
                )
                if result.returncode:
                    break
            removed = deletions.read_text().splitlines() if deletions.exists() else []
            return result, removed

    def test_official_releases_survive_any_retention_count(self):
        for keep in ("1", "3", "100"):
            with self.subTest(keep=keep):
                result, removed = self.run_cleanup(keep=keep)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(all(key.startswith(ROOT + "beta/") for key in removed), removed)
                self.assertNotIn(ROOT + "v0.2.3/", removed)

    def test_only_expired_superseded_betas_outside_rollback_window_are_deleted(self):
        result, removed = self.run_cleanup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(removed, [ROOT + f"beta/v0.2.{n}-beta.1/" for n in (6, 7, 10)])

    def test_empty_bucket_does_not_delete_anything(self):
        result, removed = self.run_cleanup(objects=[])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(removed, [])

    def test_invalid_inputs_fail_before_deleting(self):
        for options in ({"keep": "0"}, {"keep": "-1"}, {"keep": "oops"},
                        {"retention": "-1"}, {"retention": "oops"},
                        {"latest": ""}, {"latest": "../../v0.2.3"}):
            with self.subTest(options=options):
                result, removed = self.run_cleanup(**options)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(removed, [])

    def test_listing_failure_does_not_delete_anything(self):
        result, removed = self.run_cleanup(listing_failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
