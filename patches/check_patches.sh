#!/usr/bin/env bash
# Rebuild mjlab_microduck / microduck from the clean upstream commit and the files in patches/,
# and check that the result is identical (git tree) to the submodules at their pinned commits.
#
#   patches/check_patches.sh            # needs network (clones github.com/AI-FanGe/Microduck-build-tutorial)
set -euo pipefail
WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UP_URL=https://github.com/AI-FanGe/Microduck-build-tutorial.git
UP_COMMIT=4967821   # main, 2026-09-11 (the zip this workspace started from)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone -q "$UP_URL" "$tmp/up" && git -C "$tmp/up" checkout -q "$UP_COMMIT"
ok=1
for sub in mjlab_microduck microduck; do
  git -C "$tmp/up" subtree split -q --prefix="$sub" "$UP_COMMIT" -b "split-$sub" > /dev/null
  git clone -q -b "split-$sub" "$tmp/up" "$tmp/$sub"
  if [[ -d "$WS/patches/$sub" ]]; then
    for f in "$WS/patches/$sub"/*; do
      case "$f" in
        *.sh) (cd "$tmp/$sub" && bash "$f") ;;
        *.patch) git -C "$tmp/$sub" -c user.name=patch -c user.email=patch@localhost am -q "$f" ;;
      esac
    done
  fi
  want=$(git -C "$WS/$sub" rev-parse 'HEAD^{tree}'); got=$(git -C "$tmp/$sub" rev-parse 'HEAD^{tree}')
  if [[ "$want" == "$got" ]]; then echo "$sub: OK (tree $got = upstream $UP_COMMIT + patches/$sub)"; else echo "$sub: MISMATCH (submodule tree $want, rebuilt $got)"; ok=0; fi
done
[[ $ok == 1 ]]
