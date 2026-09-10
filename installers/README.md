# Public installers

This directory contains the source scripts published at
`https://downloads.kapsl.net/`. Repository paths are internal; the release
workflows publish each script under its existing filename, so public install
commands remain stable.

- `install.sh` and `install.ps1` contain the shared installer implementation.
- `install-cuda.*` and `install-tensorrt.ps1` select stable accelerator builds.
- `install-beta*` selects the beta channel and its accelerator variants.

Keep public filenames backward compatible. If a filename must change, retain
the old published entry point as a forwarding wrapper for at least one release
cycle.

Run `.github/scripts/test-install-script.sh` after changing a shell installer.
