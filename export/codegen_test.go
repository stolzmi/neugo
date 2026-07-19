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
	m, _ := nn.Sequential([]int{1, 4, 4, 1},
		nn.Conv2D(rng, 1, 2, 3, nil),
	)
	_, err := GenerateGo(modelJSON(t, m), Options{})
	if err == nil || !strings.Contains(err.Error(), "conv2d") {
		t.Fatalf("want unsupported-type error naming conv2d, got %v", err)
	}
}
