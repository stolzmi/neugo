// export/crosstarget_test.go
package export

import (
	"os/exec"
	"path/filepath"
	"testing"

	"neugo/nn"
)

func TestWasmCrossCompile(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles a subprocess with WASM target")
	}
	rng := nn.NewRNG(42)
	m, err := nn.Sequential([]int{1, 2},
		nn.Linear(rng, 2, 3, nil), nn.Sigmoid(),
	)
	if err != nil {
		t.Fatal(err)
	}
	data, err := nn.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	src, err := GenerateGo(data, Options{Package: "model"})
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "model_gen.go"), src)
	writeFile(t, filepath.Join(dir, "go.mod"), []byte("module model\n\ngo 1.25\n"))

	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = dir
	cmd.Env = append(cmd.Environ(), "GOOS=js", "GOARCH=wasm")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build failed for WASM target: %v\n%s", err, out)
	}
}
