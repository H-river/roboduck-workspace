#!/usr/bin/env bash
# Patch 0001 (mjlab_microduck commit "Restore files deleted upstream: robot/ and README.md").
# A script instead of a .patch: robot/ is ~13 MB of meshes, and both items are verbatim upstream files.
# Run inside a checkout of the upstream mjlab_microduck history (git subtree split of
# AI-FanGe/Microduck-build-tutorial@4967821, see patches/check_patches.sh).
set -euo pipefail
git checkout 6991771 -- src/mjlab_microduck/robot   # 6991771 = the split commit of upstream 34656ce^
git show ac02803:README.md > README.md              # ac02803 = the split commit of upstream's first commit
git add -A src/mjlab_microduck/robot README.md
git -c user.name=patch -c user.email=patch@localhost commit -q -m "Restore files deleted upstream: robot/ and README.md"
