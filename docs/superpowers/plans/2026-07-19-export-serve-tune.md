# Export / Serve / Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship neugo's three differentiating features: (1) `neugo export` — compile a trained model to a dependency-free Go source file that cross-compiles anywhere (native, WASM, TinyGo); (2) `serve` — in-process model serving with online learning and atomic hot weight swap; (3) `tune` — goroutine-parallel hyperparameter search with ASHA early stopping.

**Architecture:** All three features sit *on top of* the existing `Network` package and its `ModelConfig` JSON format — no changes to training internals. A small Part 0 adds the two missing primitives everything else needs: `Predict` (inference that returns outputs) and model cloning (`ForwardPass` mutates neuron state, so concurrency requires clones). Export is a code generator consuming `ModelConfig`; Serve wraps a `Network` behind `atomic.Pointer` + `sync.Pool` of inference clones; Tune is a worker pool that owns nothing of the model — it just calls a user objective.

**Tech Stack:** Go 1.25 stdlib only. No external dependencies anywhere in this plan (that is a feature headline: `go get neugo` pulls nothing else).

## Global Constraints

- Module path is `neugo`; existing package names (`Network`, `tensor`, `data`) stay as-is. New packages use standard lowercase names: `export`, `serve`, `tune`, `cmd/neugo`.
- Stdlib only. No third-party imports in any new file.
- All numerics are `float32` to match the engine (tune's search space uses `float64` for param math only).
- Generated code (export) must build with `GOOS=js GOARCH=wasm` and TinyGo: no reflection, no maps, no goroutines, imports at most `math`.
- Every task follows TDD: failing test → implement → pass → commit. Run tests with `go test ./<pkg>/ -run <Name> -v` from the repo root.
- Known engine quirk to respect: `Softmax` is applied element-wise by `Network.ForwardPass` (scalar `Apply`), which is not true vector softmax. Export parity tests therefore use Sigmoid/ReLU/Tanh/Linear models only; generated code emits *correct* vector softmax and documents the divergence.

## Independently shippable parts

Parts 1, 2, 3 are independent of each other (all depend only on Part 0). They can be executed in any order or by parallel workers after Part 0 lands. **Not in this plan:** the vectorized flat-slice core rewrite. It is a pure-performance optimization behind the same APIs and gets its own plan later; nothing here blocks on it.

---

# Part 0 — Core primitives (`Predict`, cloning)

### Task 0.1: `Predict` and `PredictBatch` on Network

**Files:**
- Create: `Network/predict.go`
- Test: `Network/predict_test.go`

**Interfaces:**
- Consumes: `Network.ForwardPass([]float32)`, `Layer.RegularSize()`, `layer.neurons[i].Activation()` (all exist).
- Produces: `func (network *Network) Predict(input []float32) []float32` and `func (network *Network) PredictBatch(inputs [][]float32) [][]float32`. Every later part calls these — exact names matter.

- [ ] **Step 1: Write the failing test**

```go
// Network/predict_test.go
package Network

import "testing"

func TestPredictReturnsOutputActivations(t *testing.T) {
	net := QuickBinary(2, 4)
	out := net.Predict([]float32{0.5, -0.5})
	if len(out) != 1 {
		t.Fatalf("want 1 output, got %d", len(out))
	}
	if out[0] < 0 || out[0] > 1 {
		t.Fatalf("sigmoid output out of range: %f", out[0])
	}
	// Predict must equal ForwardPass + reading the output layer.
	net.ForwardPass([]float32{0.5, -0.5})
	outLayer := net.layers[len(net.layers)-1]
	if got := outLayer.neurons[0].Activation(); got != out[0] {
		t.Fatalf("Predict %f != ForwardPass output %f", out[0], got)
	}
}

func TestPredictBatch(t *testing.T) {
	net := QuickBinary(2, 4)
	outs := net.PredictBatch([][]float32{{0, 0}, {1, 1}})
	if len(outs) != 2 || len(outs[0]) != 1 {
		t.Fatalf("unexpected shape: %v", outs)
	}
}
```

- [ ] **Step 2: Run to verify failure**

Run: `go test ./Network/ -run TestPredict -v`
Expected: FAIL — `net.Predict undefined`

- [ ] **Step 3: Implement**

```go
// Network/predict.go
package Network

// Predict runs a forward pass and returns a copy of the output layer's
// regular-neuron activations. NOTE: mutates internal neuron state; not safe
// for concurrent use on a shared Network — clone per goroutine (see clone.go).
func (network *Network) Predict(input []float32) []float32 {
	network.ForwardPass(input)
	outLayer := network.layers[len(network.layers)-1]
	out := make([]float32, outLayer.RegularSize())
	for i := range out {
		out[i] = outLayer.neurons[i].Activation()
	}
	return out
}

// PredictBatch runs Predict over each input sequentially.
func (network *Network) PredictBatch(inputs [][]float32) [][]float32 {
	outs := make([][]float32, len(inputs))
	for i, in := range inputs {
		outs[i] = network.Predict(in)
	}
	return outs
}
```

- [ ] **Step 4: Run to verify pass**

Run: `go test ./Network/ -run TestPredict -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add Network/predict.go Network/predict_test.go
git commit -m "feat(Network): add Predict and PredictBatch"
```

### Task 0.2: Cloning — shared-weight readers and deep copies

**Files:**
- Create: `Network/clone.go`
- Test: `Network/clone_test.go`

**Interfaces:**
- Consumes: `ToConfig()`, `NetworkFromConfig(ModelConfig)` (exist). Key facts: `NetworkFromConfig` *shares* the `Weights` slice with its config; `ForwardPass` writes only neuron activations, never weights; `BackPropagation`/training write weights.
- Produces:
  - `func (network *Network) CloneReader() Network` — private neuron state, **shared** weight storage. Safe for concurrent *inference* alongside other readers. Cheap (no weight copy).
  - `func (network *Network) CloneDeep() Network` — fully independent copy incl. weights. Required before any training on a copy.
  - `func DeepCopyConfig(c ModelConfig) ModelConfig` — deep-copies the weights tensor.

- [ ] **Step 1: Write the failing test**

```go
// Network/clone_test.go
package Network

import "testing"

func TestCloneReaderSharesWeightsPrivateNeurons(t *testing.T) {
	net := QuickBinary(2, 4)
	clone := net.CloneReader()
	if &net.weights[0][0][0] != &clone.weights[0][0][0] {
		t.Fatal("CloneReader must share weight storage")
	}
	// Same outputs, independent neuron state.
	a := net.Predict([]float32{1, 0})
	b := clone.Predict([]float32{1, 0})
	if a[0] != b[0] {
		t.Fatalf("reader clone diverged: %f vs %f", a[0], b[0])
	}
}

func TestCloneDeepIsIndependent(t *testing.T) {
	net := QuickBinary(2, 4)
	before := net.Predict([]float32{1, 0})[0]
	clone := net.CloneDeep()
	// Train the clone; original must not move.
	x := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float32{{0}, {1}, {1}, {0}}
	clone.QuickFit(x, y, 50, 0.5)
	if got := net.Predict([]float32{1, 0})[0]; got != before {
		t.Fatalf("training a deep clone mutated the original: %f -> %f", before, got)
	}
}
```

- [ ] **Step 2: Run to verify failure**

Run: `go test ./Network/ -run TestClone -v`
Expected: FAIL — `CloneReader undefined`

- [ ] **Step 3: Implement**

```go
// Network/clone.go
package Network

// DeepCopyConfig returns a config whose weight tensor is fully independent.
func DeepCopyConfig(c ModelConfig) ModelConfig {
	w := make([][][]float32, len(c.Weights))
	for l := range c.Weights {
		w[l] = make([][]float32, len(c.Weights[l]))
		for j := range c.Weights[l] {
			w[l][j] = append([]float32(nil), c.Weights[l][j]...)
		}
	}
	out := c
	out.Weights = w
	out.Layers = append([]LayerConfig(nil), c.Layers...)
	return out
}

// CloneReader returns a Network with its own neuron state but SHARED weight
// storage. Concurrent inference across readers is safe because ForwardPass
// only writes neuron activations. Never train a reader clone.
func (network *Network) CloneReader() Network {
	return NetworkFromConfig(network.ToConfig()) // ToConfig passes the live weights slice through
}

// CloneDeep returns a fully independent copy, safe to train.
func (network *Network) CloneDeep() Network {
	return NetworkFromConfig(DeepCopyConfig(network.ToConfig()))
}
```

Note: this leans on the existing behavior that `ToConfig` puts `network.weights` (the live slice) into the config and `NetworkFromConfig` assigns it without copying. Add a comment in `serialization.go` at `ToConfig` marking that behavior as load-bearing:

```go
// NOTE: Weights is the live slice, not a copy. CloneReader (clone.go) depends on this.
```

- [ ] **Step 4: Run to verify pass**

Run: `go test ./Network/ -run TestClone -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add Network/clone.go Network/clone_test.go Network/serialization.go
git commit -m "feat(Network): add CloneReader/CloneDeep/DeepCopyConfig"
```

---

# Part 1 — `export`: model → dependency-free Go source

Generated file layout (what `GenerateGo` emits — the contract for every task in this part):

```go
// Code generated by neugo export. DO NOT EDIT.
package <pkg>

import "math" // only if an emitted activation needs it

var w0 = []float32{0x1.5p-03, ...} // flat row-major, one var per layer gap
var w1 = []float32{...}

var (
	nnWeights = [][]float32{w0, w1}
	nnCols    = []int{8, 1}  // destination layer sizes
	nnAct     = []int{1, 0}  // ActivationType per destination layer
)

// <Fn> runs inference. Input length must be <inSize>.
func <Fn>(input []float32) []float32 { ...forward loop... }

func nnApplyAct(v []float32, kind int) { ...switch over used kinds only... }
```

Weight literals are emitted as **hex float literals** (`strconv.FormatFloat(float64(w), 'x', -1, 32)`) so the generated constants round-trip bit-exactly — parity with the source model is exact equality, not epsilon.

Bias handling (from `network.go`): `Weights[l]` has `len(Weights[l]) == Layers[l].Size` rows when `l == 0` (input layer, no bias) and `Layers[l].Size + 1` rows when `l >= 1`; the extra last row is the bias row. The forward loop detects this per layer as `len(W) == len(cur)+1` at generation time and emits the bias add.

### Task 1.1: `GenerateGo` codegen

**Files:**
- Create: `export/codegen.go`
- Test: `export/codegen_test.go`

**Interfaces:**
- Consumes: `Network.ModelConfig` (`Layers []LayerConfig{Size, ActivationType}`, `Weights [][][]float32`).
- Produces: `func GenerateGo(cfg Network.ModelConfig, opts Options) ([]byte, error)` and `type Options struct { Package string; FuncName string }` (defaults `"model"`, `"Predict"`). Output is `gofmt`-formatted source.

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

	"neugo/Network"
)

func testConfig() Network.ModelConfig {
	net := Network.QuickBinary(2, 3)
	return net.ToConfig()
}

func TestGenerateGoProducesValidSource(t *testing.T) {
	src, err := GenerateGo(testConfig(), Options{Package: "xormodel", FuncName: "Predict"})
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	if _, err := parser.ParseFile(fset, "gen.go", src, 0); err != nil {
		t.Fatalf("generated code does not parse: %v\n%s", err, src)
	}
	if _, err := format.Source(src); err != nil {
		t.Fatalf("generated code not gofmt-clean: %v", err)
	}
	for _, want := range []string{"package xormodel", "func Predict(input []float32) []float32", "0x"} {
		if !strings.Contains(string(src), want) {
			t.Fatalf("generated code missing %q", want)
		}
	}
}

func TestGenerateGoOmitsMathWhenUnused(t *testing.T) {
	// Linear-only model must not import math (TinyGo footprint).
	net := Network.MLP([]int{2, 2}, Network.Linear, Network.Linear)
	src, _ := GenerateGo(net.ToConfig(), Options{})
	if strings.Contains(string(src), `"math"`) {
		t.Fatal("math imported but no activation needs it")
	}
}
```

- [ ] **Step 2: Run to verify failure**

Run: `go test ./export/ -v`
Expected: FAIL — package export does not exist / `GenerateGo` undefined

- [ ] **Step 3: Implement**

```go
// export/codegen.go
package export

import (
	"bytes"
	"fmt"
	"go/format"
	"strconv"

	"neugo/Network"
)

type Options struct {
	Package  string // default "model"
	FuncName string // default "Predict"
}

// activations that require the math import
func needsMath(t Network.ActivationType) bool {
	switch t {
	case Network.Sigmoid, Network.Tanh, Network.Softmax:
		return true
	}
	return false
}

func GenerateGo(cfg Network.ModelConfig, opts Options) ([]byte, error) {
	if opts.Package == "" {
		opts.Package = "model"
	}
	if opts.FuncName == "" {
		opts.FuncName = "Predict"
	}
	if len(cfg.Layers) < 2 || len(cfg.Weights) != len(cfg.Layers)-1 {
		return nil, fmt.Errorf("export: config has %d layers and %d weight blocks; want blocks = layers-1 >= 1",
			len(cfg.Layers), len(cfg.Weights))
	}

	var b bytes.Buffer
	fmt.Fprintf(&b, "// Code generated by neugo export. DO NOT EDIT.\n")
	fmt.Fprintf(&b, "// Source model: %d layers, loss=%d, version=%s\n", len(cfg.Layers), cfg.LossType, cfg.Version)
	fmt.Fprintf(&b, "package %s\n\n", opts.Package)

	useMath := false
	usedKinds := map[int]bool{}
	for _, lc := range cfg.Layers[1:] {
		usedKinds[int(lc.ActivationType)] = true
		if needsMath(lc.ActivationType) {
			useMath = true
		}
	}
	if useMath {
		fmt.Fprintf(&b, "import \"math\"\n\n")
	}

	// One flat row-major weight var per layer gap; hex literals for exact round-trip.
	for l, W := range cfg.Weights {
		fmt.Fprintf(&b, "var w%d = []float32{", l)
		for j, row := range W {
			for k, v := range row {
				if (j*len(row)+k)%8 == 0 {
					fmt.Fprintf(&b, "\n\t")
				}
				fmt.Fprintf(&b, "%s, ", strconv.FormatFloat(float64(v), 'x', -1, 32))
			}
		}
		fmt.Fprintf(&b, "\n}\n\n")
	}

	fmt.Fprintf(&b, "var (\n\tnnWeights = [][]float32{")
	for l := range cfg.Weights {
		fmt.Fprintf(&b, "w%d, ", l)
	}
	fmt.Fprintf(&b, "}\n\tnnCols = []int{")
	for _, lc := range cfg.Layers[1:] {
		fmt.Fprintf(&b, "%d, ", lc.Size)
	}
	fmt.Fprintf(&b, "}\n\tnnAct = []int{")
	for _, lc := range cfg.Layers[1:] {
		fmt.Fprintf(&b, "%d, ", int(lc.ActivationType))
	}
	fmt.Fprintf(&b, "}\n)\n\n")

	fmt.Fprintf(&b, `// %s runs inference. len(input) must be %d; returns %d values.
func %s(input []float32) []float32 {
	cur := input
	for l := range nnWeights {
		cols := nnCols[l]
		next := make([]float32, cols)
		W := nnWeights[l]
		for j := 0; j < len(cur); j++ {
			xj := cur[j]
			row := W[j*cols : j*cols+cols]
			for k := range row {
				next[k] += xj * row[k]
			}
		}
		if len(W) == (len(cur)+1)*cols { // trailing bias row
			row := W[len(cur)*cols:]
			for k := range row {
				next[k] += row[k]
			}
		}
		nnApplyAct(next, nnAct[l])
		cur = next
	}
	return cur
}
`, opts.FuncName, cfg.Layers[0].Size, cfg.Layers[len(cfg.Layers)-1].Size, opts.FuncName)

	writeApplyAct(&b, usedKinds)
	return format.Source(b.Bytes())
}
```

And `writeApplyAct` in the same file — emit only the cases in `usedKinds`. **Copy the exact math from `Network/activation.go`** (same LeakyReLU slope constant, same Sigmoid formulation) so scalar activations match bit-for-bit:

```go
func writeApplyAct(b *bytes.Buffer, used map[int]bool) {
	fmt.Fprintf(b, "\nfunc nnApplyAct(v []float32, kind int) {\n\tswitch kind {\n")
	if used[int(Network.Sigmoid)] {
		fmt.Fprintf(b, "\tcase %d:\n\t\tfor i := range v {\n\t\t\tv[i] = float32(1.0 / (1.0 + math.Exp(float64(-v[i]))))\n\t\t}\n", int(Network.Sigmoid))
	}
	if used[int(Network.ReLU)] {
		fmt.Fprintf(b, "\tcase %d:\n\t\tfor i := range v {\n\t\t\tif v[i] < 0 {\n\t\t\t\tv[i] = 0\n\t\t\t}\n\t\t}\n", int(Network.ReLU))
	}
	if used[int(Network.Tanh)] {
		fmt.Fprintf(b, "\tcase %d:\n\t\tfor i := range v {\n\t\t\tv[i] = float32(math.Tanh(float64(v[i])))\n\t\t}\n", int(Network.Tanh))
	}
	// Linear: no case needed (identity) — but emit an empty case if used, for clarity.
	if used[int(Network.LeakyReLU)] {
		fmt.Fprintf(b, "\tcase %d:\n\t\tfor i := range v {\n\t\t\tif v[i] < 0 {\n\t\t\t\tv[i] *= 0.01 // keep in sync with Network/activation.go\n\t\t\t}\n\t\t}\n", int(Network.LeakyReLU))
	}
	if used[int(Network.Softmax)] {
		fmt.Fprintf(b, `	case %d: // true vector softmax; see plan note on engine divergence
		max := v[0]
		for _, x := range v[1:] {
			if x > max {
				max = x
			}
		}
		var sum float32
		for i := range v {
			v[i] = float32(math.Exp(float64(v[i] - max)))
			sum += v[i]
		}
		for i := range v {
			v[i] /= sum
		}
`, int(Network.Softmax))
	}
	fmt.Fprintf(b, "\t}\n}\n")
}
```

Before writing: open `Network/activation.go`, confirm the LeakyReLU slope (adjust `0.01` if it differs) and the exact Sigmoid expression.

- [ ] **Step 4: Run to verify pass**

Run: `go test ./export/ -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add export/codegen.go export/codegen_test.go
git commit -m "feat(export): generate dependency-free Go inference source from ModelConfig"
```

### Task 1.2: Compile-and-run parity test

**Files:**
- Create: `export/parity_test.go`

**Interfaces:**
- Consumes: `GenerateGo`, `Network.Predict` (Task 0.1).
- Produces: proof that generated code is **exactly** equal to the engine's outputs.

- [ ] **Step 1: Write the test (it should pass if 1.1 is correct — this is a verification task, failure means a codegen bug)**

```go
// export/parity_test.go
package export

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"neugo/Network"
)

// Compiles the generated package in a temp module with a tiny main that reads
// inputs as JSON on stdin and writes outputs as JSON on stdout, then compares
// against Network.Predict. Hex-float literals make this an exact comparison.
func TestGeneratedCodeMatchesEngineExactly(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles a subprocess")
	}
	net := Network.MLP([]int{3, 8, 5, 2}, Network.ReLU, Network.Sigmoid)
	src, err := GenerateGo(net.ToConfig(), Options{Package: "main", FuncName: "Predict"})
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
	cmd.Stdin = bytesReader(stdin)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go run failed: %v\n%s", err, out)
	}
	var got [][]float32
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("bad subprocess output: %v\n%s", err, out)
	}
	for i, in := range inputs {
		want := net.Predict(in)
		for k := range want {
			if got[i][k] != want[k] { // exact, not epsilon
				t.Fatalf("input %d output %d: generated %v != engine %v", i, k, got[i][k], want[k])
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

(`bytesReader` is `bytes.NewReader` — import `bytes`.)

- [ ] **Step 2: Run**

Run: `go test ./export/ -run TestGeneratedCodeMatchesEngineExactly -v`
Expected: PASS. If outputs differ, the bug is in codegen (bias-row detection or activation math) — fix `codegen.go`, not the test. Exact equality is achievable because both sides do the same float32 ops in the same order.

- [ ] **Step 3: Commit**

```bash
git add export/parity_test.go
git commit -m "test(export): exact parity between generated code and engine"
```

### Task 1.3: CLI — `neugo export`

**Files:**
- Create: `cmd/neugo/main.go`
- Test: `cmd/neugo/main_test.go`

**Interfaces:**
- Consumes: `Network.LoadFromFile`, `export.GenerateGo`.
- Produces: binary `neugo` with subcommand: `neugo export -model model.json -out model_gen.go -pkg model -fn Predict`. Internal seam for testing: `func run(args []string, stderr io.Writer) error`.

- [ ] **Step 1: Write the failing test**

```go
// cmd/neugo/main_test.go
package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"neugo/Network"
)

func TestExportSubcommand(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "m.json")
	outPath := filepath.Join(dir, "m_gen.go")
	net := Network.QuickBinary(2, 4)
	if err := net.SaveToFile(modelPath); err != nil {
		t.Fatal(err)
	}

	err := run([]string{"export", "-model", modelPath, "-out", outPath, "-pkg", "mymodel"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	src, _ := os.ReadFile(outPath)
	if !strings.Contains(string(src), "package mymodel") {
		t.Fatalf("output missing package clause:\n%s", src)
	}
}

func TestUnknownSubcommandErrors(t *testing.T) {
	if err := run([]string{"bogus"}, &bytes.Buffer{}); err == nil {
		t.Fatal("want error for unknown subcommand")
	}
}
```

- [ ] **Step 2: Run to verify failure**

Run: `go test ./cmd/neugo/ -v`
Expected: FAIL — `run` undefined

- [ ] **Step 3: Implement**

```go
// cmd/neugo/main.go
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"neugo/Network"
	"neugo/export"
)

func main() {
	if err := run(os.Args[1:], os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "neugo:", err)
		os.Exit(1)
	}
}

func run(args []string, stderr io.Writer) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: neugo export -model <model.json> -out <file.go> [-pkg model] [-fn Predict]")
	}
	switch args[0] {
	case "export":
		fs := flag.NewFlagSet("export", flag.ContinueOnError)
		fs.SetOutput(stderr)
		model := fs.String("model", "", "path to model JSON (required)")
		out := fs.String("out", "model_gen.go", "output .go file")
		pkg := fs.String("pkg", "model", "package name for generated file")
		fn := fs.String("fn", "Predict", "name of generated inference function")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if *model == "" {
			return fmt.Errorf("export: -model is required")
		}
		net, err := Network.LoadFromFile(*model)
		if err != nil {
			return fmt.Errorf("export: load %s: %w", *model, err)
		}
		src, err := export.GenerateGo(net.ToConfig(), export.Options{Package: *pkg, FuncName: *fn})
		if err != nil {
			return err
		}
		if err := os.WriteFile(*out, src, 0o644); err != nil {
			return err
		}
		fmt.Fprintf(stderr, "wrote %s (package %s, func %s)\n", *out, *pkg, *fn)
		return nil
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}
```

- [ ] **Step 4: Run to verify pass**

Run: `go test ./cmd/neugo/ -v` then smoke: `go run ./cmd/neugo export -model xor_final_model.json -out /tmp/xor_gen.go -pkg xor`
Expected: PASS; smoke prints `wrote /tmp/xor_gen.go ...`

- [ ] **Step 5: Commit**

```bash
git add cmd/neugo/
git commit -m "feat(cmd): neugo CLI with export subcommand"
```

### Task 1.4: Cross-target proof (WASM) + docs

**Files:**
- Create: `export/crosstarget_test.go`
- Create: `docs/EXPORT_GUIDE.md`

- [ ] **Step 1: Write the WASM build test**

```go
// export/crosstarget_test.go
package export

import (
	"os/exec"
	"path/filepath"
	"testing"

	"neugo/Network"
)

// Proves the headline: generated model code cross-compiles to WASM with stock Go.
func TestGeneratedCodeBuildsForWasm(t *testing.T) {
	if testing.Short() {
		t.Skip("invokes the go toolchain")
	}
	net := Network.QuickBinary(4, 8)
	src, err := GenerateGo(net.ToConfig(), Options{Package: "model"})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "model.go"), src)
	writeFile(t, filepath.Join(dir, "go.mod"), []byte("module wasmcheck\n\ngo 1.25\n"))
	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = dir
	cmd.Env = append(cmd.Environ(), "GOOS=js", "GOARCH=wasm")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("wasm build failed: %v\n%s", err, out)
	}
}
```

- [ ] **Step 2: Run**

Run: `go test ./export/ -run Wasm -v`
Expected: PASS

- [ ] **Step 3: Write `docs/EXPORT_GUIDE.md`** — short guide: train → `SaveToFile` → `neugo export` → drop the file in any repo; sections with exact commands for (a) native binary, (b) `GOOS=js GOARCH=wasm go build` browser demo, (c) TinyGo flash for a microcontroller (`tinygo build -target=arduino`), noting TinyGo is untested in CI and the generated code's only import is `math`.

- [ ] **Step 4: Commit**

```bash
git add export/crosstarget_test.go docs/EXPORT_GUIDE.md
git commit -m "test(export): WASM cross-compile proof; add export guide"
```

---

# Part 2 — `serve`: hot-swap serving with online learning

Design (contract for all tasks in this part):

- `modelVersion` = one immutable generation: `{ gen uint64; cfg Network.ModelConfig; pool sync.Pool }`. The pool hands out **reader clones** (shared weights, private neurons — Task 0.2), so concurrent `/predict` never contends on a lock for inference.
- `Server.current` is `atomic.Pointer[modelVersion]`. Hot swap = build a new `modelVersion` from a trained candidate's deep config, `current.Store(v)`. Old readers drain naturally; no request is ever interrupted.
- Online learning: `/feedback` pushes labeled samples into a bounded channel; one trainer goroutine accumulates them in a ring buffer; every `RetrainEvery` samples it deep-clones the current model, trains on the buffer, then a **validation gate** (holdout set) decides swap or reject. Previous version is retained for `/admin/rollback`.
- Metrics: hand-rolled counters/gauges exposed in Prometheus text format at `/metrics` (stdlib only).

HTTP API:

| Method/Path | Body → Response |
|---|---|
| POST `/predict` | `{"input":[...]}` → `{"output":[...],"model_gen":3}` |
| POST `/feedback` | `{"input":[...],"label":[...]}` → 202 |
| GET `/healthz` | → 200 `{"model_gen":3}` |
| GET `/metrics` | → Prometheus text |
| POST `/admin/rollback` | → 200 `{"model_gen":2}` (409 if no previous) |

### Task 2.1: metrics primitives

**Files:**
- Create: `serve/metrics.go`
- Test: `serve/metrics_test.go`

**Interfaces:**
- Produces: `type metrics struct` with `atomic.Int64` counters `predictTotal, feedbackTotal, swapTotal, swapRejectedTotal` and an `atomic.Uint64` gauge `modelGen`; method `writePrometheus(w io.Writer)` emitting lines like `neugo_predict_total 42`.

- [ ] **Step 1: Failing test** — construct `metrics`, bump counters, assert `writePrometheus` output contains `neugo_predict_total 2` and `neugo_model_generation 7`; assert every line matches `^[a-z_]+ \d+$` (valid exposition format).
- [ ] **Step 2: Run** `go test ./serve/ -v` — FAIL (package missing).
- [ ] **Step 3: Implement** — struct with the five atomics; `writePrometheus` uses `fmt.Fprintf` with a `# TYPE <name> counter|gauge` line before each metric.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(serve): stdlib prometheus-format metrics"`

### Task 2.2: Server core — `/predict`, `/healthz`, hot-swap plumbing

**Files:**
- Create: `serve/server.go`
- Test: `serve/server_test.go`

**Interfaces:**
- Consumes: `Network.CloneReader`, `Network.CloneDeep`, `Network.DeepCopyConfig`, `Network.NetworkFromConfig`, `Network.Predict`, `metrics` (2.1).
- Produces (used by 2.3/2.4 and by users):

```go
type Config struct {
	Holdout            []Sample // required for online learning; may be nil for serve-only
	BufferSize         int      // ring buffer capacity, default 1024
	RetrainEvery       int      // retrain after N feedback samples, default 256
	Epochs             int      // per retrain, default 5
	LearningRate       float32  // default 0.05
	MaxValLossIncrease float32  // gate slack, default 0 (candidate must be >= as good)
}
type Sample struct{ X, Y []float32 }

func New(net *Network.Network, cfg Config) *Server
func (s *Server) Handler() http.Handler          // full mux
func (s *Server) ListenAndServe(addr string) error
func (s *Server) Predict(x []float32) ([]float32, uint64)  // output + generation
func (s *Server) swapIn(cfg Network.ModelConfig)           // internal: install new generation
```

- [ ] **Step 1: Failing tests**

```go
// serve/server_test.go — core cases
func TestPredictHTTP(t *testing.T)        // POST /predict on XOR-trained model returns output + model_gen 1
func TestPredictConcurrent(t *testing.T)  // 100 goroutines x 50 requests; run with -race; all succeed, outputs consistent
func TestSwapBumpsGeneration(t *testing.T) // s.swapIn(newCfg) then Predict reports gen 2 and new weights' outputs
func TestHealthz(t *testing.T)
```

The concurrency test is the load-bearing one:

```go
func TestPredictConcurrent(t *testing.T) {
	net := trainXOR(t) // helper: QuickBinary(2,4) + QuickFit on XOR until loss < 0.1
	s := New(&net, Config{})
	want, _ := s.Predict([]float32{1, 0})
	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				got, _ := s.Predict([]float32{1, 0})
				if got[0] != want[0] {
					t.Error("racy prediction")
					return
				}
			}
		}()
	}
	wg.Wait()
}
```

- [ ] **Step 2: Run** `go test ./serve/ -race -v` — FAIL (`New` undefined).
- [ ] **Step 3: Implement**

```go
// serve/server.go (core shape — trainer wiring arrives in Task 2.3)
type modelVersion struct {
	gen  uint64
	cfg  Network.ModelConfig // deep-copied; the version owns it
	pool sync.Pool           // of *Network.Network reader clones
}

func newModelVersion(gen uint64, cfg Network.ModelConfig) *modelVersion {
	v := &modelVersion{gen: gen, cfg: cfg}
	v.pool.New = func() any {
		n := Network.NetworkFromConfig(v.cfg) // readers share v.cfg's weights
		return &n
	}
	return v
}

type Server struct {
	current  atomic.Pointer[modelVersion]
	previous atomic.Pointer[modelVersion]
	nextGen  atomic.Uint64
	feedback chan Sample
	met      metrics
	cfg      Config
}

func New(net *Network.Network, cfg Config) *Server {
	// defaults for BufferSize/RetrainEvery/Epochs/LearningRate as documented
	s := &Server{cfg: withDefaults(cfg), feedback: make(chan Sample, 4096)}
	s.nextGen.Store(2)
	s.current.Store(newModelVersion(1, Network.DeepCopyConfig(net.ToConfig())))
	s.met.modelGen.Store(1)
	return s
}

func (s *Server) Predict(x []float32) ([]float32, uint64) {
	v := s.current.Load()
	n := v.pool.Get().(*Network.Network)
	out := n.Predict(x)
	v.pool.Put(n)
	s.met.predictTotal.Add(1)
	return out, v.gen
}

func (s *Server) swapIn(cfg Network.ModelConfig) {
	old := s.current.Load()
	v := newModelVersion(s.nextGen.Add(1)-1, cfg)
	s.previous.Store(old)
	s.current.Store(v)
	s.met.modelGen.Store(v.gen)
	s.met.swapTotal.Add(1)
}
```

Subtle point the implementer must preserve: a pooled reader fetched from an **old** version's pool is still valid mid-request after a swap (weights it references are still alive via `old.cfg`); it is returned to the old pool and garbage-collected with it. That is why swap needs no locking.

`Handler()` builds a `http.ServeMux` with the routes from the table; JSON request/response structs live at the top of the file (`predictRequest{Input []float32}`, `predictResponse{Output []float32; ModelGen uint64}`, etc.). Input validation: wrong input length → 400 with `{"error": "..."}` (compare against `cfg` input layer size = `v.cfg.Layers[0].Size`).

- [ ] **Step 4: Run** `go test ./serve/ -race -v` — PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(serve): lock-free hot-swap serving core"`

### Task 2.3: Online trainer with validation gate

**Files:**
- Create: `serve/online.go`
- Test: `serve/online_test.go`
- Modify: `serve/server.go` (wire `/feedback`, start/stop trainer)

**Interfaces:**
- Consumes: `Server.swapIn`, `Network.CloneDeep`, `Network.ForwardPass`/`CalculateLoss`, `Network.TrainBatch`.
- Produces:

```go
func (s *Server) StartOnline(ctx context.Context)  // launches trainer goroutine; no-op if cfg.Holdout == nil
func valLoss(net *Network.Network, holdout []Sample) float32  // mean CalculateLoss over holdout
// internal: type ringBuffer struct{...} with Push(Sample) and Snapshot() []Sample
```

- [ ] **Step 1: Failing tests**

```go
func TestRingBufferOverwritesOldest(t *testing.T)
func TestValLoss(t *testing.T) // hand-computable: trained XOR net has low loss, fresh net higher

// The core behavior test — deterministic, no HTTP:
func TestOnlineLearningSwapsWhenBetter(t *testing.T) {
	// Start from a DELIBERATELY bad model (0 training epochs) on XOR.
	// Holdout = the 4 XOR rows. Feed the 4 XOR samples repeatedly
	// (RetrainEvery: 16, Epochs: 200, LearningRate: 0.5).
	// Poll up to 5s: expect model_gen > 1 AND valLoss(current) < valLoss(initial).
}

func TestGateRejectsWorseCandidate(t *testing.T) {
	// Feed WRONG labels (inverted XOR) but holdout stays correct:
	// candidate trains to be worse on holdout → gate rejects.
	// Expect: swapRejectedTotal > 0, model_gen stays 1, predictions unchanged.
}
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** trainer loop:

```go
func (s *Server) StartOnline(ctx context.Context) {
	if s.cfg.Holdout == nil {
		return
	}
	go func() {
		buf := newRingBuffer(s.cfg.BufferSize)
		sinceRetrain := 0
		for {
			select {
			case <-ctx.Done():
				return
			case sm := <-s.feedback:
				buf.Push(sm)
				sinceRetrain++
				if sinceRetrain < s.cfg.RetrainEvery {
					continue
				}
				sinceRetrain = 0
				s.retrain(buf.Snapshot())
			}
		}
	}()
}

func (s *Server) retrain(samples []Sample) {
	curV := s.current.Load()
	currentNet := Network.NetworkFromConfig(curV.cfg) // reader for scoring
	cand := currentNet.CloneDeep()
	x := make([][]float32, len(samples))
	y := make([][]float32, len(samples))
	for i, sm := range samples {
		x[i], y[i] = sm.X, sm.Y
	}
	for range s.cfg.Epochs {
		cand.TrainBatch(x, y, s.cfg.LearningRate)
	}
	candLoss := valLoss(&cand, s.cfg.Holdout)
	curLoss := valLoss(&currentNet, s.cfg.Holdout)
	if candLoss <= curLoss+s.cfg.MaxValLossIncrease {
		s.swapIn(Network.DeepCopyConfig(cand.ToConfig()))
	} else {
		s.met.swapRejectedTotal.Add(1)
	}
}
```

(`TrainBatch(inputs, labels [][]float32, learningRate float32) float32` — verified in `Network/batch.go:5`; the returned batch loss can be logged but the gate decision uses only holdout loss.) `/feedback` handler: decode, validate lengths, non-blocking send to `s.feedback` (drop + count if full — never block a request), respond 202.

- [ ] **Step 4: Run** `go test ./serve/ -race -v` (all Part 2 tests) — PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(serve): online learning with holdout validation gate"`

### Task 2.4: Rollback, metrics endpoint, example

**Files:**
- Modify: `serve/server.go` (rollback handler, `/metrics` route)
- Create: `examples/serve_xor.go`
- Test: extend `serve/server_test.go`

- [ ] **Step 1: Failing tests** — `TestRollbackRestoresPreviousGeneration` (swap then rollback → old gen's outputs; second rollback → 409), `TestMetricsEndpoint` (contains `neugo_predict_total`).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** rollback = `swapIn`-style store of `previous` into `current` (swap the pair, so rollback of rollback works); `/metrics` calls `met.writePrometheus(w)`.
- [ ] **Step 4: Run** `go test ./serve/ -race` — PASS.
- [ ] **Step 5: Write `examples/serve_xor.go`** (`//go:build ignore` tag like other examples if they use one — check `examples/train.go` first): trains XOR, `serve.New(...)`, `StartOnline`, `ListenAndServe(":8080")`, with a comment block of `curl` commands demonstrating predict → feedback → watch `model_gen` bump → rollback.
- [ ] **Step 6: Commit** `git commit -m "feat(serve): rollback + metrics endpoint + example"`

---

# Part 3 — `tune`: parallel hyperparameter search with ASHA

### Task 3.1: Search space and params

**Files:**
- Create: `tune/space.go`
- Test: `tune/space_test.go`

**Interfaces:**
- Produces:

```go
func NewSpace() *Space
func (s *Space) Float(name string, min, max float64) *Space     // uniform
func (s *Space) LogFloat(name string, min, max float64) *Space  // log-uniform, min > 0
func (s *Space) Int(name string, min, max int) *Space           // inclusive
func (s *Space) Choice(name string, options ...string) *Space
func (s *Space) Sample(r *rand.Rand) Params

type Params map[string]any
func (p Params) Float(name string) float64  // panics with a clear message on wrong name/type
func (p Params) Int(name string) int
func (p Params) Choice(name string) string
```

- [ ] **Step 1: Failing test** — sample 1000 times with a seeded `rand.New(rand.NewSource(1))`: Float stays in [min,max]; LogFloat(1e-4, 1e-1) produces ≥10% of samples below 1e-3 (log-uniformity smoke check — uniform would give ~0.9%); Int inclusive of both ends over many draws; Choice returns only listed options; same seed ⇒ identical sample sequence (determinism).
- [ ] **Step 2: Run** `go test ./tune/ -v` — FAIL.
- [ ] **Step 3: Implement** — `Space` holds `[]paramDef{name, kind, fmin, fmax, imin, imax, choices}`; `Sample` iterates in insertion order (determinism depends on it — do not use a map); LogFloat: `math.Exp(lo + r.Float64()*(hi-lo))` with `lo, hi := math.Log(min), math.Log(max)`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(tune): search space with deterministic sampling"`

### Task 3.2: Worker-pool runner (random search)

**Files:**
- Create: `tune/tuner.go`, `tune/result.go`
- Test: `tune/tuner_test.go`

**Interfaces:**
- Consumes: `Space.Sample`, `Params`.
- Produces:

```go
type Trial struct {
	ID     int
	Params Params
	Seed   int64 // cfg.Seed + ID; objective should use this for any RNG it needs
	// ASHA fields added in Task 3.3
}
type Objective func(t *Trial) (float64, error)
type Config struct {
	Trials   int
	Workers  int   // default runtime.NumCPU()
	Seed     int64
	Maximize bool  // default false = minimize
	ASHA     *ASHAConfig // nil = no pruning (Task 3.3)
}
type TrialResult struct {
	ID       int
	Params   Params
	Value    float64
	Err      error
	Pruned   bool
	Duration time.Duration
}
type Results struct{ Trials []TrialResult; /* sorted best-first */ }
func (r *Results) Best() TrialResult
func (r *Results) Top(k int) []TrialResult
func (r *Results) String() string // aligned table via text/tabwriter
func Run(ctx context.Context, space *Space, obj Objective, cfg Config) (*Results, error)
```

- [ ] **Step 1: Failing tests**

```go
// Deterministic objective: recover a known optimum.
func TestRunFindsMinimum(t *testing.T) {
	space := NewSpace().Float("x", -10, 10)
	obj := func(tr *Trial) (float64, error) {
		x := tr.Params.Float("x")
		return (x - 3) * (x - 3), nil
	}
	res, err := Run(context.Background(), space, obj, Config{Trials: 500, Workers: 8, Seed: 42})
	// Best x within 0.2 of 3.0; res.Best().Value < 0.05
}

// Parallelism actually happens: 32 trials x 20ms sleep on 8 workers
// must finish well under serial time (assert < 32*20ms/2 = 320ms elapsed).
func TestRunIsParallel(t *testing.T)

// Same seed twice ⇒ identical Best().Params (params determinism;
// scheduling order may differ, sampled values may not).
func TestRunDeterministicParams(t *testing.T)

// Objective error is captured on the TrialResult, not fatal to the run.
func TestObjectiveErrorRecorded(t *testing.T)

// Context cancellation stops early and returns partial results + ctx.Err().
func TestRunHonorsContext(t *testing.T)
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** — determinism rule: **sample all trial params up front, sequentially** from one `rand.New(rand.NewSource(cfg.Seed))` (trial i's params never depend on scheduling), then feed pre-built `*Trial`s through a `chan *Trial` to `Workers` goroutines; results into a slice indexed by ID (no mutex needed — disjoint writes); `sync.WaitGroup` to join; sort by Value (respecting `Maximize`, errors and pruned trials sort last).
- [ ] **Step 4: Run** `go test ./tune/ -race -v` — PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(tune): goroutine worker-pool random search"`

### Task 3.3: ASHA early stopping

**Files:**
- Create: `tune/asha.go`
- Modify: `tune/tuner.go` (Trial gains `Report`/`ShouldPrune`; Run wires the shared asha state)
- Test: `tune/asha_test.go`

**Interfaces:**
- Produces:

```go
type ASHAConfig struct {
	MinResource     int // e.g. 1 epoch
	MaxResource     int // e.g. 81 epochs
	ReductionFactor int // η, default 3
}
// On Trial:
func (t *Trial) Report(resource int, value float64) // record intermediate metric at a rung
func (t *Trial) ShouldPrune() bool                  // true if last Report landed outside top 1/η at its rung
```

Objective usage pattern (documented on `Report`):

```go
obj := func(tr *Trial) (float64, error) {
	net := buildModel(tr.Params)
	var loss float64
	for epoch := 1; epoch <= maxEpochs; epoch++ {
		loss = trainOneEpoch(net)
		tr.Report(epoch, loss)
		if tr.ShouldPrune() {
			return loss, nil // tuner marks TrialResult.Pruned = true
		}
	}
	return loss, nil
}
```

- [ ] **Step 1: Failing tests**

```go
// Unit: rung promotion math, single-threaded.
func TestAshaPromotesTopFraction(t *testing.T) {
	// rung with values {1,2,...,9} already reported, η=3:
	// a new value 1.5 (top third) → promote; value 8 → prune.
}

// Integration: objective where params with x>0 converge (loss→0 over 81 steps)
// and x<=0 plateau at loss 10. With ASHA{1,81,3}, 200 trials:
//   - Best().Value < 0.1
//   - at least 30% of x<=0 trials are Pruned
//   - total Reports across all trials is well below 200*81 (compute saved)
func TestAshaPrunesBadTrials(t *testing.T)
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** — shared `asha` struct: `mu sync.Mutex; rungs map[int][]float64` (key = rung index `r` where resource ≥ `MinResource*η^r`); `decide(rung int, value float64, maximize bool) bool` appends value, returns whether value is within the best `ceil(n/η)` of that rung's recorded values (async ASHA: with few observations, promote by default). `Trial.Report` maps resource → highest applicable rung, calls `decide`, stores result in `t.pruneNow`; resources between rungs are no-ops (never prune there).
- [ ] **Step 4: Run** `go test ./tune/ -race -v` — PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(tune): ASHA successive-halving pruning"`

### Task 3.4: Real-model example + doc

**Files:**
- Create: `examples/tune_wine.go` (match the build-tag/CLI convention of existing files in `examples/` — inspect `examples/wine_quality_clean.go` first)
- Create: `docs/TUNE_GUIDE.md`

- [ ] **Step 1: Write the example** — wine-quality dataset via existing `data` package; space: `LogFloat("lr", 1e-4, 0.5)`, `Int("hidden", 4, 64)`, `Choice("act", "relu", "tanh")`; objective builds `Network.QuickBinary`-style model from params, trains with per-epoch `Report(epoch, valLoss)`, honors `ShouldPrune`; run with `Workers: runtime.NumCPU()`, print `results.String()` top-10 and wall-clock vs. `Trials × mean-trial-time` to showcase the parallel speedup.
- [ ] **Step 2: Run it** — `go run examples/tune_wine.go` completes in minutes and prints a sane best config (val loss below the untuned baseline from `examples/wine_quality_clean.go`).
- [ ] **Step 3: Write `docs/TUNE_GUIDE.md`** — API walkthrough (space → objective → Run → Results), ASHA explanation with the resource/rung picture, determinism guarantees (same Seed ⇒ same params), and the "one process, all cores, no cluster" pitch.
- [ ] **Step 4: Commit** `git commit -m "feat(tune): wine-quality tuning example + guide"`

---

# Final integration task

### Task 4: README + full-suite verification

- [ ] **Step 1:** `go vet ./... && go test ./... -race` — all green.
- [ ] **Step 2:** Update `README.md`: add the three features to the feature list with 5-line examples each (`neugo export` one-liner, `serve.New(...).ListenAndServe`, `tune.Run`), linking `docs/EXPORT_GUIDE.md` and `docs/TUNE_GUIDE.md`; move them out of "Future Improvements".
- [ ] **Step 3:** Commit: `git commit -m "docs: document export/serve/tune features"`

---

## Self-review notes (already applied)

- **No invented APIs:** every `Network` symbol referenced (`ToConfig`, `NetworkFromConfig`, `QuickBinary`, `MLP`, `QuickFit`, `TrainBatch`, `LoadFromFile`, `SaveToFile`) exists today; the two that don't (`Predict`, clones) are created in Part 0 before anything uses them. One open check is flagged inline: the `examples/` build-tag convention (Tasks 2.4, 3.4).
- **Type consistency:** `Sample`, `Config`, `modelVersion`, `Params`, `Trial`, `TrialResult` are each defined once and referenced with identical shapes across tasks.
- **Known risks called out:** element-wise Softmax quirk (parity tests avoid it), `ForwardPass` mutation (drives the clone design), reader clones sharing weight storage (documented as load-bearing in Task 0.2).
