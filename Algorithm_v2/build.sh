#!/bin/bash
set -euo pipefail

# Directory where all build files go
BUILD_DIR="out"

function show_help() {
  cat <<EOF
Usage: $0 [clean] [<cmake-args>...]

Commands:
  clean        Remove the entire build directory
  <no args>    Configure & build
EOF
  exit 1
}

# If first argument is "clean", just remove build/ and exit
if [[ "${1:-}" == "clean" ]]; then
  echo "Cleaning build directory..."
  rm -rf "$BUILD_DIR"
  echo "Removed $BUILD_DIR/"
  exit 0
fi

# Otherwise, configure & build
echo "Configuring..."
mkdir -p "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR"

echo "Building..."
cmake --build "$BUILD_DIR" -- -j"$(nproc)"

echo "Build complete. Executable is in $BUILD_DIR"
