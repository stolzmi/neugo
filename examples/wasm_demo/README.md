# WASM browser demo

Trains the same tiny XOR model as `examples/xor`, exports it to
dependency-free Go source via the `export` package, compiles that to
WebAssembly, and runs it live in a browser tab — no server-side inference,
no Python, nothing but the static files in this directory once built.

## Layout

- `train/main.go` — trains the model and writes `model/model_gen.go` via
  `export.GenerateGo` (see `docs/EXPORT_GUIDE.md`).
- `model/model_gen.go` — the generated, dependency-free inference code
  (checked in so this example builds without a training step first;
  regenerate it any time by re-running `train`).
- `wasm/main.go` — the `syscall/js` bridge exposing `model.Predict` as a
  `predict(...)` function in JavaScript (the exact pattern from
  `docs/EXPORT_GUIDE.md`'s "Browser WASM" section).
- `index.html` — a minimal UI: pick A/B, click Predict, see the model's
  output computed entirely client-side.
- `build.sh` — runs the whole pipeline below in one command.

## Build and run

```bash
./examples/wasm_demo/build.sh
cd examples/wasm_demo && python3 -m http.server 8080
# open http://localhost:8080/ in a browser
```

`build.sh` does three things:

1. Re-runs `train` (regenerates `model/model_gen.go` from a fresh training
   run — safe to skip if you just want to rebuild the existing model).
2. Copies this Go toolchain's `wasm_exec.js` next to `index.html` (its
   path moved from `misc/wasm/` to `lib/wasm/` in Go 1.24; the script
   checks both).
3. Compiles `wasm/main.go` for `GOOS=js GOARCH=wasm` into
   `predict.wasm`.

Any static file server works for the last step — the browser needs to
fetch `predict.wasm` over HTTP(S), not `file://`, for
`WebAssembly.instantiateStreaming` to work in most browsers.

## Why this architecture

`export.GenerateGo` currently supports `Linear` layers and the pointwise
activations (ReLU/Sigmoid/Tanh/LeakyReLU/GELU/Softmax) plus `Dropout`
(a no-op at inference) — see `docs/EXPORT_GUIDE.md`'s supported/unsupported
module lists. The XOR MLP (`Linear → ReLU → Linear → Sigmoid`) is exactly
that, chosen so this demo showcases the real, currently-supported export
path rather than something that would fail to generate.
