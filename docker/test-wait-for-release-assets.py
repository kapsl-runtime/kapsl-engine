#!/usr/bin/env python3
"""Host-only release waiter tests; no GitHub requests, sleeps, or GPU jobs."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40
MOCK_COMMAND = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ["WAIT_TEST_ROOT"])
name = Path(sys.argv[0]).name
log = root / (name + ".json")
calls = json.loads(log.read_text()) if log.exists() else []
calls.append(sys.argv[1:])
log.write_text(json.dumps(calls))
if name == "gh":
    if os.environ.get("MOCK_GH_FAILURE") == "true":
        sys.exit(1)
    snapshots = json.loads((root / "runs.json").read_text())
    snapshot = snapshots[min(len(calls) - 1, len(snapshots) - 1)]
    query = sys.argv[sys.argv.index("--jq") + 1]
    result = subprocess.run(["jq", "-r", query], input=json.dumps(snapshot), text=True)
    sys.exit(result.returncode)
if name == "curl":
    sys.exit(0 if os.environ.get("MOCK_ASSETS_READY") == "true" else 22)
"""


def installer_run(status="completed", conclusion="success", **overrides):
    return dict(
        id=100,
        head_branch="v0.2.4",
        head_sha=SHA,
        status=status,
        conclusion=conclusion,
        html_url="https://github.com/kapsl-runtime/kapsl-engine/actions/runs/100",
    ) | overrides


class ReleaseAssetWaitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("gh", "curl", "sleep"):
            executable = bin_dir / name
            executable.write_text(MOCK_COMMAND)
            executable.chmod(0o755)
        self.env = os.environ | {
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "WAIT_TEST_ROOT": str(self.root),
            "KAPSL_VERSION": "0.2.4",
            "KAPSL_INSTALLER_RUN_SHA": SHA,
            "GITHUB_REPOSITORY": "kapsl-runtime/kapsl-engine",
            "KAPSL_WAIT_ATTEMPTS": "1",
            "MOCK_ASSETS_READY": "true",
            "MOCK_GH_FAILURE": "false",
        }
        self.env.pop("KAPSL_WAIT_DELAY_SECONDS", None)

    def run_waiter(self, snapshots, **env):
        (self.root / "runs.json").write_text(json.dumps([
            {"workflow_runs": runs} for runs in snapshots
        ]))
        return subprocess.run(
            ["sh", str(ROOT / "docker/wait-for-release-assets.sh")],
            env=self.env | env, text=True, capture_output=True, timeout=20,
        )

    def calls(self, command):
        log = self.root / (command + ".json")
        return json.loads(log.read_text()) if log.exists() else []

    def test_failed_installer_stops_without_asset_requests(self):
        for conclusion in ("startup_failure", "failure", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                result = self.run_waiter([[installer_run(conclusion=conclusion)]])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Installer workflow did not succeed", result.stderr)
                self.assertIn(conclusion, result.stderr)
                self.assertIn("/actions/runs/100", result.stderr)
                self.assertEqual(self.calls("curl"), [])
                self.assertEqual(self.calls("sleep"), [])

    def test_pending_installer_is_polled_every_thirty_seconds(self):
        result = self.run_waiter([
            [installer_run(status="in_progress", conclusion=None)],
            [installer_run()],
        ], KAPSL_WAIT_ATTEMPTS="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("sleep"), [["30"]])
        self.assertEqual(len(self.calls("curl")), 10)
        self.assertEqual(len(self.calls("gh")), 2)
        self.assertIn("head_sha=" + SHA, self.calls("gh")[0])
        self.assertIn("event=push", self.calls("gh")[0])
        self.assertIn(
            "repos/kapsl-runtime/kapsl-engine/actions/workflows/release-runtime-installers.yml/runs",
            self.calls("gh")[0],
        )

    def test_failure_after_starting_also_stops(self):
        result = self.run_waiter([
            [installer_run(status="queued", conclusion=None)],
            [installer_run(conclusion="failure")],
        ], KAPSL_WAIT_ATTEMPTS="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Installer workflow did not succeed", result.stderr)
        self.assertEqual(self.calls("curl"), [])

    def test_missing_run_and_unrelated_tags_or_commits_do_not_authorize(self):
        for runs in (
            [],
            [installer_run(head_branch="v0.2.3")],
            [installer_run(head_sha="b" * 40)],
        ):
            with self.subTest(runs=runs):
                result = self.run_waiter([runs])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not_started", result.stdout)
                self.assertEqual(self.calls("curl"), [])

    def test_newest_matching_run_supersedes_old_failure(self):
        result = self.run_waiter([[
            installer_run(id=101),
            installer_run(id=100, conclusion="failure"),
        ]])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls("curl")), 10)

    def test_github_api_error_fails_closed(self):
        result = self.run_waiter([[]], MOCK_GH_FAILURE="true")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unable to check the installer workflow", result.stderr)
        self.assertEqual(self.calls("curl"), [])

    def test_successful_producer_still_requires_all_assets(self):
        result = self.run_waiter([[installer_run()]], MOCK_ASSETS_READY="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Release assets are not complete", result.stdout)
        self.assertEqual(len(self.calls("curl")), 10)

    def test_manual_existing_release_does_not_require_an_installer_run(self):
        result = self.run_waiter([[]], KAPSL_INSTALLER_RUN_SHA="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("gh"), [])
        self.assertEqual(len(self.calls("curl")), 10)

    def test_workflow_passes_push_commit_and_read_only_permissions(self):
        workflow = (ROOT / ".github/workflows/release-docker-images.yml").read_text()
        job = workflow.split("  wait-for-release-assets:\n", 1)[1].split(
            "  build-and-push:\n", 1
        )[0]
        self.assertIn("permissions:\n      actions: read\n      contents: read", job)
        self.assertIn("github.event_name == 'push' && github.sha || ''", job)
        self.assertIn("GH_TOKEN: ${{ github.token }}", job)


if __name__ == "__main__":
    unittest.main()
