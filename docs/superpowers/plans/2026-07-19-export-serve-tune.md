# Export / Serve / Tune Implementation Plan (v2 — flax-restructure base)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship neugo's three differentiating features: (1) `export` — compile a trained model to a dependency-free Go source file that cross-compiles anywhere (native, WASM, TinyGo) plus a `neugo export` CLI; (2) `serve` — in-process model serving with online learning and atomic hot weight swap; (3) `tune` — goroutine-parallel hyperparameter search with ASHA early stopping.

**Architecture:** v2 targets the flax-restructure codebase: packages `nn` (Module interface, `SequentialModel`, `Tensor{Data []float32, Shape []int}`, JSON `Save`/`Load` of a `{type, config, params, modules}` module tree) and `train` (`Trainer` with `New(model, opt, loss)`, `Fit`, `Predict`, `Evaluate`; `Loss` interface; `Optimizer` with `SGD(lr)`; `Metrics{Loss, Accuracy, ...}`). Task 1 adds the missing primitives (`nn.Marshal`/`nn.Unmarshal`/`nn.Clone`); export is a code generator consuming the saved model JSON directly (its own decoder — no dependency on `nn` internals); serve wraps a model behind `atomic.Pointer` + `sync.Pool` of clones (required: `LinearLayer.Forward` caches `l.input` even in Inference mode, so a shared model races under concurrent requests); tune is model-agnostic.

**Tech Stack:** Go 1.25 stdlib only. No external dependencies anywhere.

## Global Constraints

- Module path is `neugo`; existing packages `nn`, `train`, `data` stay as-is. New packages: `export`, `serve`, `tune`, `cmd/neugo`.
- Stdlib only. No third-party imports in any new file.
- Model numerics are `float32` (tune's param math may use `float64`).
- Generated code (export) must build with `GOOS=js GOARCH=wasm` and TinyGo: no reflection, no maps, no goroutines, imports at most `math`.
- Generated code must reproduce the engine's outputs **exactly** (same float32 ops in the same order as the `nn` forward passes; weight literals emitted as hex floats). Parity tests assert `==`, not epsilon.
- TDD per task: failing test → implement → pass → commit. Run from repo root: `go test ./<pkg>/ -v` (add `-race` for `serve` and `tune`).
- Examples follow the existing convention: one directory per example, `examples/<name>/main.go`.

## Task ordering

Task 1 blocks Tasks 6–9 (serve). Tasks 2–5 (export) depend only on the existing `nn` JSON format. Tasks 10–13 (tune) are independent of everything. Task 14 is final integration. **Not in this plan:** performance work (vectorized/blocked matmul) — separate plan later.

---

### Task 1: `nn.Marshal` / `nn.Unmarshal` / `nn.Clone`

**Files:**
- Modify: `nn/serialize.go` (refactor `Save`/`Load` to delegate; do not change their behavior)
- Test: `nn/serialize_clone_test.go` (new file)

**Interfaces:**
- Consumes: private `encodeModule`/`decodeModule` in `nn/serialize.go`, `NewRNG` (`nn/rng.go`).
- Produces (serve depends on these exact signatures):
  - `func Marshal(model *SequentialModel) ([]byte, error)` — the same JSON `Save` writes (indentation may differ; use non-indented `json.Marshal`).
  - `func Unmarshal(data []byte) (*SequentialModel, error)` — same semantics as `Load` minus file I/O (including the root-must-be-sequential check; use `NewRNG(0)` as the throwaway reconstruction RNG, as `Load` does).
  - `func Clone(model *SequentialModel) (*SequentialModel, error)` — `Marshal` then `Unmarshal`: fully independent deep copy.

- [ ] **Step 1: Write the failing test**

```go
// nn/serialize_clone_test.go
package nn

import (
	"bytes"
	"testing"
)

func testModel(t *testing.T) *SequentialModel {
	t.Helper()
	rng := NewRNG(7)
	m, err := Sequential([]int{1, 3},
		Linear(rng, 3, 4, nil), ReLU(),
		Linear(rng, 4, 2, nil), Sigmoid(),
	)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func forward1(t *testing.T, m *SequentialModel, in []float32) []float32 {
	t.Helper()
	x, err := NewTensorFromData(in, []int{1, len(in)})
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(&Context{Mode: Inference}, x)
	if err != nil {
		t.Fatal(err)
	}
	return out.Data
}

func TestMarshalUnmarshalRoundTrip(t *testing.T) {
	m := testModel(t)
	data, err := Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{0.5, -1, 2}
	a, b := forward1(t, m, in), forward1(t, m2, in)
	if !bytes.Equal(float32Bytes(a), float32Bytes(b)) {
		t.Fatalf("round-trip changed outputs: %v vs %v", a, b)
	}
}

func TestCloneIsDeepAndIndependent(t *testing.T) {
	m := testModel(t)
	c, err := Clone(m)
	if err != nil {
		t.Fatal(err)
	}
	in := []float32{1, 1, 1}
	before := forward1(t, m, in)
	// Mutate every clone param; original outputs must not move.
	for _, p := range c.Params() {
		for i := range p.Value.Data {
			p.Value.Data[i] += 100
		}
	}
	after := forward1(t, m, in)
	if !bytes.Equal(float32Bytes(before), float32Bytes(after)) {
		t.Fatalf("mutating clone changed original: %v vs %v", before, after)
	}
}

func TestUnmarshalRejectsNonSequentialRoot(t *testing.T) {
	if _, err := Unmarshal([]byte(`{"type":"linear"}`)); err == nil {
		t.Fatal("want error for non-sequential root")
	}
}

func float32Bytes(v []float32) []byte {
	b := make([]byte, 0, len(v)*4)
	for _, x := range v {
		u := math.Float32bits(x)
		b = append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	return b
}
```

(Add `"math"` to the imports.)

- [ ] **Step 2: Run to verify failure** — `go test ./nn/ -run 'Marshal|Clone' -v` → FAIL: `Marshal` undefined.
- [ ] **Step 3: Implement** in `nn/serialize.go`: `Marshal` = `encodeModule(model)` + `json.Marshal` (nil-model check like `Save`); `Unmarshal` = `json.Unmarshal` into `moduleDoc` + `decodeModule(doc, NewRNG(0))` + root-type check (move the shared logic out of `Save`/`Load` so it exists once; `Save`/`Load` keep their exact current behavior incl. indented file output and error prefixes); `Clone` = `Marshal`+`Unmarshal`.
- [ ] **Step 4: Run to verify pass** — `go test ./nn/ -v` → all PASS (existing serialize tests must still pass).
- [ ] **Step 5: Commit** — `git add nn/ && git commit -m "feat(nn): Marshal/Unmarshal/Clone for in-memory model copies"`

---

### Task 2: `export.GenerateGo` codegen

**Files:**
- Create: `export/codegen.go`
- Test: `export/codegen_test.go`

**Interfaces:**
- Consumes: the model JSON format written by `nn.Save`/`nn.Marshal` — a tree of `{"type", "config", "params", "modules"}` nodes (see `nn/serialize.go:10-53` for the exact schema and config field names). `export` defines its own decoder structs for this JSON; it must NOT import `nn` in `codegen.go` (the test file may).
- Produces:
  - `type Options struct { Package string; FuncName string }` (defaults `"model"`, `"Predict"`)
  - `func GenerateGo(modelJSON []byte, opts Options) ([]byte, error)` — gofmt-formatted (`go/format.Source`) Go source.

**Supported module types (v1):** `sequential` (root), `linear`, `relu`, `sigmoid`, `tanh`, `leaky_relu` (per-instance alpha from config), `gelu`, `softmax`, `dropout` (identity at inference — emit nothing). Any other type (`conv2d`, `maxpool2d`, `avgpool2d`, `flatten`, `batchnorm`, nested `sequential`) → `error` naming the unsupported type. Missing/malformed params (`W`, `B` absent, wrong lengths vs config) → error.

**Generated file contract** (single sample, no batch dim):

```go
// Code generated by neugo export. DO NOT EDIT.
package <pkg>

import "math" // only if sigmoid/tanh/gelu/softmax present

var w0 = []float32{<hex floats>} // linear W, row-major [in*out], layout W[i*out+o]
var b0 = []float32{<hex floats>}

// <Fn> runs inference. len(input) must be <first linear in_features>;
// returns <last layer width> values.
func <Fn>(input []float32) []float32 {
	cur := input
	cur = nnLinear(cur, w0, b0, <in>, <out>)
	nnReLU(cur)
	cur = nnLinear(cur, w1, b1, <in>, <out>)
	nnSigmoid(cur)
	return cur
}

func nnLinear(x, w, b []float32, in, out int) []float32 { ... }
// plus one helper per activation actually used
```

**Exactness requirements** (Global Constraints bind here):
- Weight/bias literals: `strconv.FormatFloat(float64(v), 'x', -1, 32)` (hex float, exact round-trip).
- `nnLinear` must use the same accumulation order as `nn.LinearLayer.Forward` (`nn/linear.go:60-68`) specialized to batch=1: `sum := b[o]; for i { sum += x[i] * w[i*out+o] }`.
- Each activation helper must copy the exact float32 arithmetic from `nn/activation.go` (relu/sigmoid/tanh/leaky_relu/gelu — same casts, same operation order; read the file, don't transcribe from memory). Softmax: copy the formula from `nn.SoftmaxModule.Forward` (`nn/activation.go:126-152`) specialized to one row (max-subtract, exp, normalize — same order).
- Emit only the helpers actually used; import `math` only if a used helper needs it.

- [ ] **Step 1: Write the failing test**

```go
// export/codegen_test.go
package export

import (
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"neugo/nn"
)

func modelJSON(t *testing.T, m *nn.SequentialModel) []byte {
	t.Helper()
	data, err := nn.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestGenerateGoProducesValidSource(t *testing.T) {
	rng := nn.NewRNG(1)
	m, err := nn.Sequential([]int{1, 2},
		nn.Linear(rng, 2, 3, nil), nn.ReLU(),
		nn.Linear(rng, 3, 1, nil), nn.Sigmoid(),
	)
	if err != nil {
		t.Fatal(err)
	}
	src, err := GenerateGo(modelJSON(t, m), Options{Package: "xormodel", FuncName: "Predict"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parser.ParseFile(token.NewFileSet(), "gen.go", src, 0); err != nil {
		t.Fatalf("generated code does not parse: %v\n%s", err, src)
	}
	if _, err := format.Source(src); err != nil {
		t.Fatalf("not gofmt-clean: %v", err)
	}
	for _, want := range []string{"package xormodel", "func Predict(input []float32) []float32", "0x"} {
		if !strings.Contains(string(src), want) {
			t.Fatalf("missing %q in generated code", want)
		}
	}
}

func TestGenerateGoOmitsMathWhenUnused(t *testing.T) {
	rng := nn.NewRNG(1)
	m, _ := nn.Sequential([]int{1, 2}, nn.Linear(rng, 2, 2, nil), nn.ReLU())
	src, err := GenerateGo(modelJSON(t, m), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(src), `"math"`) {
		t.Fatal("math imported but no used activation needs it")
	}
}

func TestGenerateGoRejectsUnsupportedModule(t *testing.T) {
	rng := nn.NewRNG(1)
	m, _ := nn.Sequential([]int{1, 1, 4, 4},
		nn.Conv2D(rng, 1, 2, 3, 1, nil),
	)
	_, err := GenerateGo(modelJSON(t, m), Options{})
	if err == nil || !strings.Contains(err.Error(), "conv2d") {
		t.Fatalf("want unsupported-type error naming conv2d, got %v", err)
	}
}
```

Before finalizing this test: check `nn.Conv2D`'s real constructor signature in `nn/conv.go` and the `Sequential` input shape it expects (see `nn/conv_test.go` for a working call) and adjust the third test accordingly.

- [ ] **Step 2: Run to verify failure** — `go test ./export/ -v` → FAIL: package missing.
- [ ] **Step 3: Implement** `export/codegen.go` per the contract above. Structure: local structs `moduleDoc{Type string; Config json.RawMessage; Params map[string]paramDoc; Modules []moduleDoc}`, `paramDoc{Shape []int; Data []float32}` with the same JSON tags as `nn/serialize.go`; walk root's `Modules`, build two buffers (declarations + body statements), track used helpers; validate linear params (`len(W.Data) == in*out`, `len(B.Data) == out`, widths chain correctly).
- [ ] **Step 4: Run to verify pass** — `go test ./export/ -v` → PASS (3 tests).
- [ ] **Step 5: Commit** — `git add export/ && git commit -m "feat(export): generate dependency-free Go inference source from model JSON"`

---

### Task 3: Export compile-and-run parity test

**Files:**
- Create: `export/parity_test.go`

**Interfaces:** consumes `GenerateGo`, `nn.Marshal`, `nn` forward. Proves generated code output `==` engine output (bitwise, float32).

- [ ] **Step 1: Write the test**

```go
// export/parity_test.go
package export

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"neugo/nn"
)

func TestGeneratedCodeMatchesEngineExactly(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles a subprocess")
	}
	rng := nn.NewRNG(42)
	m, err := nn.Sequential([]int{1, 3},
		nn.Linear(rng, 3, 8, nil), nn.ReLU(),
		nn.Linear(rng, 8, 5, nil), nn.Tanh(),
		nn.Linear(rng, 5, 3, nil), nn.Softmax(),
	)
	if err != nil {
		t.Fatal(err)
	}
	data, err := nn.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	src, err := GenerateGo(data, Options{Package: "main", FuncName: "Predict"})
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "model_gen.go"), src)
	writeFile(t, filepath.Join(dir, "go.mod"), []byte("module gen\n\ngo 1.25\n"))
	writeFile(t, filepath.Join(dir, "main.go"), []byte(`package main

import (
	"encoding/json"
	"os"
)

func main() {
	var inputs [][]float32
	json.NewDecoder(os.Stdin).Decode(&inputs)
	outs := make([][]float32, len(inputs))
	for i, in := range inputs {
		outs[i] = Predict(in)
	}
	json.NewEncoder(os.Stdout).Encode(outs)
}
`))

	inputs := [][]float32{{0, 0, 0}, {1, -1, 0.5}, {0.25, 0.75, -2}}
	stdin, _ := json.Marshal(inputs)
	cmd := exec.Command("go", "run", ".")
	cmd.Dir = dir
	cmd.Stdin = bytes.NewReader(stdin)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go run failed: %v", err)
	}
	var got [][]float32
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("bad subprocess output: %v\n%s", err, out)
	}
	ctx := &nn.Context{Mode: nn.Inference}
	for i, in := range inputs {
		x, _ := nn.NewTensorFromData(append([]float32(nil), in...), []int{1, len(in)})
		wantT, err := m.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		for k, want := range wantT.Data {
			if got[i][k] != want { // exact float32 equality, no epsilon
				t.Fatalf("input %d output %d: generated %v != engine %v", i, k, got[i][k], want)
			}
		}
	}
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
```

- [ ] **Step 2: Run** — `go test ./export/ -run Parity -v` (and the exact test name). Expected: PASS. A mismatch means a codegen bug (accumulation order or activation math differs from `nn`) — fix `codegen.go`, never widen the test to epsilon.
- [ ] **Step 3: Commit** — `git add export/parity_test.go && git commit -m "test(export): exact bitwise parity between generated code and engine"`

---

### Task 4: `neugo` CLI with `export` subcommand

**Files:**
- Create: `cmd/neugo/main.go`
- Test: `cmd/neugo/main_test.go`

**Interfaces:**
- Consumes: `export.GenerateGo` (reads the model JSON file directly — no `nn` import needed).
- Produces: `neugo export -model <model.json> -out <file.go> [-pkg model] [-fn Predict]`; testing seam `func run(args []string, stderr io.Writer) error`.

- [ ] **Step 1: Write the failing test** — two tests: (a) `TestExportSubcommand`: build a small model with `nn.Sequential`+`nn.Save` into `t.TempDir()`, call `run([]string{"export", "-model", mPath, "-out", oPath, "-pkg", "mymodel"}, io.Discard)`, assert no error and the output file contains `package mymodel`; (b) `TestUnknownSubcommandErrors`: `run([]string{"bogus"}, io.Discard)` returns an error.
- [ ] **Step 2: Run to verify failure** — `go test ./cmd/neugo/ -v` → FAIL: `run` undefined.
- [ ] **Step 3: Implement** — `main()` calls `run(os.Args[1:], os.Stderr)`, exits 1 on error; `run` switches on `args[0]`; `export` uses a `flag.FlagSet` (ContinueOnError, output to stderr param) with the four flags; `-model` required; `os.ReadFile` model JSON → `export.GenerateGo` → `os.WriteFile` 0644; success message to the stderr writer.
- [ ] **Step 4: Run to verify pass** — `go test ./cmd/neugo/ -v` → PASS. Smoke: train nothing — just `go run ./cmd/neugo export` with a missing model → clean error, not panic.
- [ ] **Step 5: Commit** — `git add cmd/ && git commit -m "feat(cmd): neugo CLI with export subcommand"`

---

### Task 5: Export cross-target proof (WASM) + guide

**Files:**
- Create: `export/crosstarget_test.go`
- Create: `docs/EXPORT_GUIDE.md`

- [ ] **Step 1: Write the WASM build test** — like the parity test's temp-module setup but only the generated file + go.mod (package `model`, no main); run `go build ./...` with `cmd.Env = append(cmd.Environ(), "GOOS=js", "GOARCH=wasm")`; skip under `testing.Short()`. Model: `nn.Linear`+`nn.Sigmoid` suffices.
- [ ] **Step 2: Run** — `go test ./export/ -run Wasm -v` → PASS.
- [ ] **Step 3: Write `docs/EXPORT_GUIDE.md`** — sections: train & save (`nn.Save`), `go run ./cmd/neugo export -model m.json -out model_gen.go -pkg model`, drop into any Go project (zero deps, only `math`); cross-compile recipes: native (`GOOS=linux GOARCH=arm64 go build`), browser WASM (`GOOS=js GOARCH=wasm`), TinyGo (`tinygo build -target=<board>`, noted as not CI-covered); supported module types list + the v1 unsupported list (conv/pool/flatten/batchnorm) verbatim from Task 2; exactness guarantee (hex-float literals, bitwise parity with the engine).
- [ ] **Step 4: Commit** — `git add export/crosstarget_test.go docs/EXPORT_GUIDE.md && git commit -m "test(export): WASM cross-compile proof; add export guide"`

---

### Task 6: `serve` metrics primitives

**Files:**
- Create: `serve/metrics.go`
- Test: `serve/metrics_test.go`

**Interfaces:**
- Produces: unexported `type metrics struct` with `atomic.Int64` fields `predictTotal, feedbackTotal, feedbackDropped, swapTotal, swapRejectedTotal` and `atomic.Uint64` field `modelGen`; method `writePrometheus(w io.Writer)` emitting, per metric, a `# TYPE neugo_<name> counter` (or `gauge` for `neugo_model_generation`) line then `neugo_<name> <value>`. Metric names: `neugo_predict_total`, `neugo_feedback_total`, `neugo_feedback_dropped_total`, `neugo_swap_total`, `neugo_swap_rejected_total`, `neugo_model_generation`.

- [ ] **Step 1: Failing test** — bump `predictTotal` twice and `modelGen` to 7; capture `writePrometheus` into a `bytes.Buffer`; assert it contains `neugo_predict_total 2` and `neugo_model_generation 7`; assert every non-`#` line matches `^[a-z_]+ [0-9]+$` (`regexp`).
- [ ] **Step 2: Run** — `go test ./serve/ -v` → FAIL (package missing).
- [ ] **Step 3: Implement** as specified. No locks — atomics only.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add serve/ && git commit -m "feat(serve): stdlib prometheus-format metrics"`

---

### Task 7: `serve` core — lock-free hot-swap serving

**Files:**
- Create: `serve/server.go`
- Test: `serve/server_test.go`

**Interfaces:**
- Consumes: `nn.Marshal`, `nn.Unmarshal` (Task 1), `nn.SequentialModel.Forward`, `nn.Context{Mode: nn.Inference}`, `nn.NewTensorFromData`, `metrics` (Task 6).
- Produces (Tasks 8–9 build on these exact signatures):

```go
type Sample struct{ X, Y []float32 }

type Config struct {
	InputDim           int          // required; length every input must have
	Loss               train.Loss   // required for online learning (Task 8); nil = serve-only
	Holdout            []Sample     // required for online learning; nil = serve-only
	BufferSize         int          // ring buffer capacity, default 1024
	RetrainEvery       int          // retrain after N feedback samples, default 256
	Epochs             int          // per retrain, default 5
	LearningRate       float32      // default 0.05
	MaxValLossIncrease float32      // gate slack, default 0 (candidate must be at least as good)
}

func New(model *nn.SequentialModel, cfg Config) (*Server, error) // errors if InputDim <= 0 or Marshal fails
func (s *Server) Handler() http.Handler                          // routes below
func (s *Server) ListenAndServe(addr string) error
func (s *Server) Predict(x []float32) ([]float32, uint64, error) // output, model generation
func (s *Server) Generation() uint64
// internal: func (s *Server) swapIn(doc []byte)  — installs a new generation
```

Routes (JSON bodies; wrong input length or malformed JSON → 400 `{"error":"..."}`):

| Method/Path | Body → Response |
|---|---|
| POST `/predict` | `{"input":[...]}` → `{"output":[...],"model_gen":3}` |
| GET `/healthz` | → 200 `{"model_gen":3}` |

(`/feedback`, `/metrics`, `/admin/rollback` arrive in Tasks 8–9.)

**Core design (must be preserved):** `modelVersion{gen uint64; doc []byte; pool sync.Pool}` where `pool.New = func() any { m, err := nn.Unmarshal(v.doc); ... }` (on unmarshal error — impossible for a doc that round-tripped once — panic with context). `Server.current` is `atomic.Pointer[modelVersion]`. `Predict` loads current, gets a clone from its pool, builds a `[1, InputDim]` tensor, `Forward` with Inference mode, copies out `out.Data`, puts the clone back. Swap = store a new `modelVersion`; a clone fetched from an old version mid-request stays valid and is returned to the old version's pool (GC'd with it) — no locking anywhere on the predict path. Clones are needed because `nn` modules cache forward inputs even in Inference mode (`nn/linear.go:57`).

- [ ] **Step 1: Failing tests**

```go
// serve/server_test.go — required cases (write real code for each):
// helper: tinyModel(t) — nn.Sequential([]int{1,2}, nn.Linear(rng,2,4,nil), nn.ReLU(), nn.Linear(rng,4,1,nil), nn.Sigmoid())
func TestPredictHTTP(t *testing.T)           // httptest: POST /predict {"input":[1,0]} → 200, 1 output, model_gen 1
func TestPredictWrongLength(t *testing.T)    // {"input":[1]} → 400 with error JSON
func TestHealthz(t *testing.T)               // → {"model_gen":1}
func TestSwapBumpsGeneration(t *testing.T)   // swapIn(marshal of a different model) → Predict returns gen 2 and different output
func TestPredictConcurrent(t *testing.T)     // below — the load-bearing one; -race must pass
```

```go
func TestPredictConcurrent(t *testing.T) {
	s := newTestServer(t) // helper wrapping New(tinyModel(t), Config{InputDim: 2})
	want, _, err := s.Predict([]float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				got, _, err := s.Predict([]float32{1, 0})
				if err != nil || got[0] != want[0] {
					t.Errorf("racy or failed prediction: %v %v", got, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
```

- [ ] **Step 2: Run** — `go test ./serve/ -race -v` → FAIL: `New` undefined.
- [ ] **Step 3: Implement** `serve/server.go` per the design block. Defaults applied in `New`. `Predict` must copy `out.Data` into a fresh slice before returning the clone to the pool (the tensor belongs to the clone's next caller otherwise — subtle!). Actually `Forward` allocates a fresh output tensor per call (`nn/linear.go:59`), but copy anyway and note why: the contract is "caller owns the returned slice."
- [ ] **Step 4: Run** — `go test ./serve/ -race -v` → PASS.
- [ ] **Step 5: Commit** — `git add serve/ && git commit -m "feat(serve): lock-free hot-swap serving core"`

---

### Task 8: `serve` online learning with validation gate

**Files:**
- Create: `serve/online.go`
- Test: `serve/online_test.go`
- Modify: `serve/server.go` (add `/feedback` route; `feedback chan Sample` field)

**Interfaces:**
- Consumes: `Server.swapIn`, `nn.Unmarshal`, `train.New`, `train.SGD`, `train.Epochs`, `train.BatchSize`, `train.Shuffle`, `train.Seed`, `Trainer.Fit`, `Trainer.Evaluate` (returns `train.Metrics`; use its `.Loss`), `nn.NewTensorFromData`.
- Produces:

```go
func (s *Server) StartOnline(ctx context.Context) error // error if cfg.Holdout or cfg.Loss is nil; starts one goroutine
// internal:
// type ringBuffer — fixed capacity, Push overwrites oldest, Snapshot() []Sample (copy)
// func samplesToTensors(ss []Sample) (x, y *nn.Tensor, err error)  — shapes [n, dimX], [n, dimY]
// func (s *Server) retrain(samples []Sample)
// func (s *Server) holdoutLoss(doc []byte) float32 — unmarshal, Evaluate on cfg.Holdout, return Metrics.Loss
```

Trainer loop: receive from `s.feedback`; push to ring; every `RetrainEvery` received samples call `retrain(ring.Snapshot())`. `retrain`: candidate = `nn.Unmarshal(current.doc)`; `train.New(candidate, train.SGD(cfg.LearningRate), cfg.Loss)`; `Fit(x, y, train.Epochs(cfg.Epochs), train.BatchSize(min(32, len(samples))), train.Shuffle(true), train.Seed(int64(gen)))`; gate: candidate holdout loss `<=` current holdout loss `+ MaxValLossIncrease` → `swapIn(nn.Marshal(candidate))`, else `swapRejectedTotal.Add(1)`. `/feedback` handler: validate lengths (X vs InputDim; all Y same length as holdout Y), **non-blocking** channel send — on full channel drop the sample and bump `feedbackDropped` (never block a request), respond 202.

- [ ] **Step 1: Failing tests** (all deterministic, no HTTP except the feedback-endpoint one, no sleeps — call `retrain` directly where possible):

```go
func TestRingBufferOverwritesOldest(t *testing.T)  // cap 4, push 6, Snapshot has last 4 in order
func TestSamplesToTensors(t *testing.T)            // 3 samples of dims 2/1 → shapes [3,2],[3,1]
func TestRetrainSwapsWhenBetter(t *testing.T)
// Untrained model on XOR; Holdout = 4 XOR rows; Loss = train.BCELoss();
// call s.retrain(xorSamples) directly with Epochs: 300, LearningRate: 0.9.
// Assert: Generation() == 2 and holdout loss decreased.
func TestGateRejectsWorseCandidate(t *testing.T)
// Model pre-trained on XOR (fit until holdout loss < 0.2 using train.Trainer in the test);
// call s.retrain with INVERTED labels. Assert: Generation() still 1,
// swapRejectedTotal == 1, predictions unchanged.
func TestFeedbackEndpointAccepts(t *testing.T)     // POST /feedback → 202, feedbackTotal == 1
func TestStartOnlineRequiresConfig(t *testing.T)   // nil Loss or Holdout → error
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** per the interfaces block. `retrain` reads `s.current.Load()` once at entry (gen + doc) — swap-under-retrain is benign: the gate compares against the doc it started from.
- [ ] **Step 4: Run** — `go test ./serve/ -race -v` → PASS.
- [ ] **Step 5: Commit** — `git add serve/ && git commit -m "feat(serve): online learning with holdout validation gate"`

---

### Task 9: `serve` rollback + `/metrics` + example

**Files:**
- Modify: `serve/server.go` (`previous` field, `/admin/rollback`, `/metrics` routes)
- Test: extend `serve/server_test.go`
- Create: `examples/serve_xor/main.go`

- [ ] **Step 1: Failing tests** — `TestRollbackRestoresPreviousGeneration` (after one swap: rollback → 200, `Predict` returns the *old* outputs under a *new* generation number; `previous` and `current` exchange, so rollback-of-rollback returns to the newer model); `TestRollbackWithoutPrevious` (fresh server → 409); `TestMetricsEndpoint` (GET `/metrics` contains `neugo_predict_total` and `neugo_model_generation`).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** — `swapIn` stores the displaced version into `s.previous` (plain `atomic.Pointer[modelVersion]`); rollback exchanges current/previous under a small mutex (rollback is not a hot path; the predict path stays lock-free) and installs the restored doc as a **new generation number** (monotonic gen, never reuse), bumping `swapTotal`.
- [ ] **Step 4: Run** — `go test ./serve/ -race -v` → PASS.
- [ ] **Step 5: Write `examples/serve_xor/main.go`** — follows `examples/xor/main.go` conventions: build+train XOR with `nn`/`train`, `serve.New` with holdout+BCE config, `StartOnline(context.Background())`, print the curl walkthrough (predict → feedback ×N → healthz shows gen bump → rollback), `ListenAndServe(":8080")`.
- [ ] **Step 6: Verify example compiles** — `go build ./examples/serve_xor/` → OK.
- [ ] **Step 7: Commit** — `git add serve/ examples/serve_xor/ && git commit -m "feat(serve): rollback + metrics endpoint + serve_xor example"`

---

### Task 10: `tune` search space

**Files:**
- Create: `tune/space.go`
- Test: `tune/space_test.go`

**Interfaces:**
- Produces:

```go
func NewSpace() *Space
func (s *Space) Float(name string, min, max float64) *Space     // uniform
func (s *Space) LogFloat(name string, min, max float64) *Space  // log-uniform; panics if min <= 0 or max < min
func (s *Space) Int(name string, min, max int) *Space           // inclusive both ends
func (s *Space) Choice(name string, options ...string) *Space
func (s *Space) Sample(r *rand.Rand) Params

type Params map[string]any
func (p Params) Float(name string) float64  // panics with clear message on missing name / wrong kind
func (p Params) Int(name string) int
func (p Params) Choice(name string) string
```

- [ ] **Step 1: Failing test** — with `rand.New(rand.NewSource(1))`: 1000 samples of `Float("x",-2,2)` all in range; `LogFloat("lr",1e-4,1e-1)`: all in range AND ≥10% of samples < 1e-3 (uniform would give ~0.9% — this asserts log-uniformity); `Int("h",4,8)` hits both 4 and 8 across 1000 draws, never outside; `Choice` only returns listed options; two spaces sampled with same seed produce identical sequences (`reflect.DeepEqual` over 20 draws); `Params.Float("missing")` panics (use `defer/recover`).
- [ ] **Step 2: Run** — `go test ./tune/ -v` → FAIL.
- [ ] **Step 3: Implement** — `Space` holds an ordered `[]paramDef` slice (NOT a map — `Sample` iterates insertion order; determinism depends on it); LogFloat: `math.Exp(lo + r.Float64()*(hi-lo))` with `lo, hi := math.Log(min), math.Log(max)`; Int: `min + r.Intn(max-min+1)`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add tune/ && git commit -m "feat(tune): search space with deterministic sampling"`

---

### Task 11: `tune` worker-pool runner (random search)

**Files:**
- Create: `tune/tuner.go`, `tune/result.go`
- Test: `tune/tuner_test.go`

**Interfaces:**
- Consumes: `Space.Sample`, `Params` (Task 10).
- Produces (Task 12 extends `Trial`):

```go
type Trial struct {
	ID     int
	Params Params
	Seed   int64 // cfg.Seed + int64(ID); objectives use this for their own RNG
}
type Objective func(t *Trial) (float64, error)
type Config struct {
	Trials   int
	Workers  int   // <=0 → runtime.NumCPU()
	Seed     int64
	Maximize bool  // false = minimize
	ASHA     *ASHAConfig // Task 12; nil = no pruning
}
type TrialResult struct {
	ID       int
	Params   Params
	Value    float64
	Err      error
	Pruned   bool
	Duration time.Duration
}
type Results struct{ Trials []TrialResult } // sorted best-first; errored/pruned trials after all scored ones
func (r *Results) Best() TrialResult
func (r *Results) Top(k int) []TrialResult
func (r *Results) String() string // text/tabwriter table: rank, ID, value, params
func Run(ctx context.Context, space *Space, obj Objective, cfg Config) (*Results, error)
```

**Determinism rule (binding):** sample ALL trial params up front, sequentially, from one `rand.New(rand.NewSource(cfg.Seed))` — trial i's params never depend on goroutine scheduling. Then feed prebuilt `*Trial`s through a channel to `Workers` goroutines; each writes `results[trial.ID]` (disjoint indices — no mutex); `sync.WaitGroup` join; sort at the end.

- [ ] **Step 1: Failing tests**

```go
func TestRunFindsMinimum(t *testing.T)
// Space: Float("x", -10, 10); objective (x-3)^2; Trials: 500, Workers: 8, Seed: 42.
// Assert Best().Value < 0.05 and |Best().Params.Float("x") - 3| < 0.25.
func TestRunIsParallel(t *testing.T)
// Objective sleeps 20ms; 32 trials, 8 workers. Wall time < 320ms (half of serial 640ms).
func TestRunDeterministicParams(t *testing.T)
// Two Runs, same Seed: Best().Params identical (DeepEqual) and same set of trial params.
func TestObjectiveErrorRecorded(t *testing.T)
// Objective errors for x < 0: those TrialResults carry Err, run does not fail, Best has no Err.
func TestRunHonorsContext(t *testing.T)
// Cancel after ~50ms with slow objective: Run returns ctx.Err(), partial Results non-nil.
func TestMaximize(t *testing.T)
// Maximize: true with objective -(x-3)^2 → same optimum found.
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** per the determinism rule. Context: workers select on `ctx.Done()` between trials; `Run` returns `(partialResults, ctx.Err())` on cancellation.
- [ ] **Step 4: Run** — `go test ./tune/ -race -v` → PASS.
- [ ] **Step 5: Commit** — `git add tune/ && git commit -m "feat(tune): goroutine worker-pool random search"`

---

### Task 12: `tune` ASHA early stopping

**Files:**
- Create: `tune/asha.go`
- Modify: `tune/tuner.go` (`Trial` gains `Report`/`ShouldPrune` and internal wiring; `Run` builds the shared asha state when `cfg.ASHA != nil`)
- Test: `tune/asha_test.go`

**Interfaces:**
- Produces:

```go
type ASHAConfig struct {
	MinResource     int // e.g. 1 (epochs)
	MaxResource     int // e.g. 81
	ReductionFactor int // η; <=1 → default 3
}
func (t *Trial) Report(resource int, value float64) // record intermediate metric
func (t *Trial) ShouldPrune() bool                  // true if the last Report fell outside the top 1/η at its rung
```

Documented usage pattern (goes in `Report`'s doc comment):

```go
for epoch := 1; epoch <= maxEpochs; epoch++ {
	loss = trainOneEpoch()
	tr.Report(epoch, loss)
	if tr.ShouldPrune() {
		return loss, nil // tuner marks the TrialResult Pruned
	}
}
```

**Semantics:** rungs at resources `MinResource * η^k` for k = 0,1,2,… up to `MaxResource`. `Report(resource, value)`: if `resource` equals a rung's resource exactly, record the value in that rung (shared `asha` struct: `mu sync.Mutex; rungs map[int][]float64`) and decide promotion: promoted iff value is within the best `ceil(n/η)` of all values recorded at that rung so far (direction per `cfg.Maximize`; with n < η observations, promote by default — async ASHA). Reports at non-rung resources never prune. A trial that returns after `ShouldPrune()` is marked `Pruned: true` in its result (tuner-side: track that the trial's last decision was "prune"). With `cfg.ASHA == nil`, `Report` is a no-op and `ShouldPrune` always returns false.

- [ ] **Step 1: Failing tests**

```go
func TestRungPromotionMath(t *testing.T)
// Direct unit test on the asha struct: rung already holds {1..9}, η=3:
// decide(1.5) → promote (top third); decide(8.0) → prune. Minimize direction.
func TestNonRungResourceNeverPrunes(t *testing.T)
func TestNoASHAConfigNeverPrunes(t *testing.T)
func TestAshaPrunesBadTrialsAndSavesWork(t *testing.T)
// Space: Float("x", -1, 1). Objective simulates 81 epochs:
//   good (x>0): loss = 1/float64(epoch); bad (x<=0): loss = 10.
//   Reports every epoch, honors ShouldPrune (returns early).
// ASHA{1, 81, 3}, 100 trials, seed 7. Assert:
//   Best().Value < 0.1; ≥30% of bad-x trials Pruned;
//   total epochs executed (count via atomic in objective) < 100*81/2.
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** per semantics block.
- [ ] **Step 4: Run** — `go test ./tune/ -race -v` → PASS.
- [ ] **Step 5: Commit** — `git add tune/ && git commit -m "feat(tune): ASHA successive-halving pruning"`

---

### Task 13: `tune` real-model example + guide

**Files:**
- Create: `examples/tune_wine/main.go`
- Create: `docs/TUNE_GUIDE.md`

- [ ] **Step 1: Write the example** — follow `examples/wine_quality/main.go` for data loading (package `data`, dataset at `dataset/wine_quality/winequality-red.csv`) and model construction idioms. Space: `LogFloat("lr", 1e-4, 0.5)`, `Int("hidden", 4, 64)`, `Choice("act", "relu", "tanh")`. Objective: build `nn.Sequential` from params (seeded `nn.NewRNG(tr.Seed)`), `train.New(model, train.SGD(lr), train.BCELoss())`, train epoch-by-epoch (`train.Epochs(1)` per Fit call, or a per-epoch callback — whichever `train`'s API makes cleaner; check `train/callback.go`), `Report(epoch, valLoss)` each epoch, honor `ShouldPrune`. Config: `Trials: 60, Workers: runtime.NumCPU(), ASHA: &tune.ASHAConfig{MinResource: 2, MaxResource: 32, ReductionFactor: 4}`. Print `results.String()` top-10, total wall time, and total-epochs-executed vs. `Trials×MaxResource` to show ASHA's savings.
- [ ] **Step 2: Verify** — `go build ./examples/tune_wine/` then `go run ./examples/tune_wine` finishes (a few minutes max) and prints a best val loss below the untuned default from `examples/wine_quality/main.go`.
- [ ] **Step 3: Write `docs/TUNE_GUIDE.md`** — space→objective→Run→Results walkthrough; ASHA rung diagram; determinism guarantee (same Seed ⇒ same param sets regardless of scheduling); the one-process/all-cores/no-cluster pitch.
- [ ] **Step 4: Commit** — `git add examples/tune_wine/ docs/TUNE_GUIDE.md && git commit -m "feat(tune): wine-quality tuning example + guide"`

---

### Task 14: README + full-suite verification

- [ ] **Step 1:** `go vet ./... && go build ./... && go test ./... -race` — all green.
- [ ] **Step 2:** Update `README.md`: add Export / Serve / Tune to the feature list, each with a ≤10-line example (`neugo export` command; `serve.New` + `StartOnline` + `ListenAndServe`; `tune.Run` with ASHA), linking `docs/EXPORT_GUIDE.md` and `docs/TUNE_GUIDE.md`. Keep the existing README structure and tone.
- [ ] **Step 3:** Commit — `git add README.md && git commit -m "docs: document export/serve/tune features"`

---

## Self-review notes (v2)

- Every `nn`/`train` symbol referenced was verified against the flax-restructure source: `Marshal`/`Unmarshal`/`Clone` are created in Task 1 before use; `Sequential`, `Linear`, `ReLU/Sigmoid/Tanh/Softmax`, `NewTensorFromData`, `Context/Inference`, `NewRNG`, `Save`, `train.New/SGD/BCELoss/Epochs/BatchSize/Shuffle/Seed/Fit/Evaluate`, `Metrics.Loss` all exist today. Flagged in-task checks: `nn.Conv2D` signature (Task 2 test), per-epoch training idiom (Task 13).
- Concurrency claims verified: `LinearLayer.Forward` writes `l.input` (`nn/linear.go:57`) in all modes → clone-per-goroutine pooling is required, not optional.
- Exactness is achievable: generated code copies the engine's op order; hex-float literals round-trip; parity asserts `==`.
