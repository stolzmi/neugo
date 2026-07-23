#!/usr/bin/env bash
# examples/wasm_demo/build.sh
#
# Runs the full pipeline documented in README.md: retrain+export the XOR
# model, locate this Go toolchain's wasm_exec.js (its path moved between
# Go versions: lib/wasm/ since Go 1.24, misc/wasm/ before that), and
# compile the browser wrapper for GOOS=js GOARCH=wasm. Run from anywhere;
# it cd's to the repo root itself.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

echo "==> retraining + exporting the XOR model"
go run ./examples/wasm_demo/train

echo "==> locating wasm_exec.js"
goroot="$(go env GOROOT)"
wasm_exec=""
for candidate in "$goroot/lib/wasm/wasm_exec.js" "$goroot/misc/wasm/wasm_exec.js"; do
  if [ -f "$candidate" ]; then
    wasm_exec="$candidate"
    break
  fi
done
if [ -z "$wasm_exec" ]; then
  echo "error: wasm_exec.js not found under $goroot (checked lib/wasm and misc/wasm)" >&2
  exit 1
fi
cp "$wasm_exec" examples/wasm_demo/wasm_exec.js
echo "    copied from $wasm_exec"

echo "==> compiling the browser wrapper (GOOS=js GOARCH=wasm)"
GOOS=js GOARCH=wasm go build -o examples/wasm_demo/predict.wasm ./examples/wasm_demo/wasm

echo
echo "Done. Serve examples/wasm_demo/ over HTTP and open index.html, e.g.:"
echo "    cd examples/wasm_demo && python3 -m http.server 8080"
echo "    open http://localhost:8080/"
