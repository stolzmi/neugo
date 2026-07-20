// export/parity_test.go
package export

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stolzmi/neugo/nn"
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
