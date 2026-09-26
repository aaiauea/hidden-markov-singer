#!/usr/bin/env bash
# Build libhms_world.so -- the real WORLD vocoder behind a flat C ABI.
#
# HMS needs no Python headers and no pip package for this: we compile the WORLD
# C++ sources (BSD-3-Clause, https://github.com/mmorise/World) together with
# tools/world_native/hms_world_capi.cpp and load the result over ctypes.
#
# Source resolution order:
#   1. $HMS_WORLD_SRC            -- a WORLD (or pyworld sdist) checkout
#   2. tools/world_native/World  -- previously unpacked sources
#   3. PyPI sdist of pyworld     -- downloads and unpacks the vendored WORLD
#
# Usage:  tools/build_world.sh [output_dir]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
out_dir="${1:-$root/hms/vocoder/_native}"
src_dir="$here/world_native/World"

mkdir -p "$out_dir"

if [[ -n "${HMS_WORLD_SRC:-}" ]]; then
  echo "using WORLD sources from HMS_WORLD_SRC=$HMS_WORLD_SRC"
  cp -r "$HMS_WORLD_SRC" "$src_dir"
elif [[ ! -d "$src_dir/src/world" ]]; then
  echo "downloading WORLD sources (vendored inside the pyworld sdist) ..."
  tmp="$(mktemp -d)"
  # NOTE: no --no-binary here; we want the sdist *file*, not a built wheel.
  python3 -m pip download pyworld --no-deps -d "$tmp" >/dev/null 2>&1
  tar xzf "$tmp"/pyworld-*.tar.gz -C "$tmp"
  rm -rf "$src_dir"
  cp -r "$(echo "$tmp"/pyworld-*/lib/World)" "$src_dir"
  rm -rf "$tmp"
fi

echo "compiling ..."
cxx="${CXX:-g++}"
mkdir -p "$out_dir"
"$cxx" -O2 -fPIC -shared -std=c++11 \
  -I"$src_dir/src" \
  "$here/world_native/hms_world_capi.cpp" \
  "$src_dir"/src/*.cpp \
  -o "$out_dir/libhms_world.so"

echo "wrote $out_dir/libhms_world.so"
ls -la "$out_dir/libhms_world.so"
