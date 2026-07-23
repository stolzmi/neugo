package main

import (
	"bytes"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func TestDiffSubcommandIdenticalArchitectureReportsWeightDelta(t *testing.T) {
	tmpDir := t.TempDir()

	rng := nn.NewRNG(1)
	modelA, err := nn.Sequential([]int{1, 2},
		nn.Linear(rng, 2, 3, nn.XavierInit()), nn.ReLU(),
		nn.Linear(rng, 3, 1, nn.XavierInit()), nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	pathA := filepath.Join(tmpDir, "a.json")
	if err := nn.Save(modelA, pathA); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// modelB: same architecture, perturbed weights.
	modelB, err := nn.Clone(modelA)
	if err != nil {
		t.Fatalf("Clone: %v", err)
	}
	for _, p := range modelB.Params() {
		for i := range p.Value.Data {
			p.Value.Data[i] += 0.1
		}
	}
	pathB := filepath.Join(tmpDir, "b.json")
	if err := nn.Save(modelB, pathB); err != nil {
		t.Fatalf("Save: %v", err)
	}

	var stdout bytes.Buffer
	if err := run([]string{"diff", "-a", pathA, "-b", pathB}, &stdout, io.Discard); err != nil {
		t.Fatalf("run(diff): %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "architecture: identical") {
		t.Errorf("output missing \"architecture: identical\":\n%s", out)
	}
	if !strings.Contains(out, "total weight delta") {
		t.Errorf("output missing total weight delta summary:\n%s", out)
	}
	if strings.Contains(out, "|delta| = 0.000000") {
		t.Errorf("expected nonzero per-param deltas (weights were perturbed by 0.1 each), got:\n%s", out)
	}
}

func TestDiffSubcommandDifferentArchitectureReportsStructuralDiff(t *testing.T) {
	tmpDir := t.TempDir()

	rng := nn.NewRNG(1)
	modelA, err := nn.Sequential([]int{1, 2}, nn.Linear(rng, 2, 3, nn.XavierInit()), nn.ReLU())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	pathA := filepath.Join(tmpDir, "a.json")
	if err := nn.Save(modelA, pathA); err != nil {
		t.Fatalf("Save: %v", err)
	}

	modelB, err := nn.Sequential([]int{1, 2}, nn.Linear(rng, 2, 3, nn.XavierInit()), nn.Sigmoid())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	pathB := filepath.Join(tmpDir, "b.json")
	if err := nn.Save(modelB, pathB); err != nil {
		t.Fatalf("Save: %v", err)
	}

	var stdout bytes.Buffer
	if err := run([]string{"diff", "-a", pathA, "-b", pathB}, &stdout, io.Discard); err != nil {
		t.Fatalf("run(diff): %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "architecture: DIFFERS") {
		t.Errorf("output missing \"architecture: DIFFERS\":\n%s", out)
	}
	if !strings.Contains(out, "TYPE CHANGED") {
		t.Errorf("output missing a TYPE CHANGED line for the relu->sigmoid swap:\n%s", out)
	}
}

func TestDiffSubcommandMissingFlagsErrors(t *testing.T) {
	if err := run([]string{"diff", "-a", "x.json"}, io.Discard, io.Discard); err == nil {
		t.Fatal("expected error when -b is missing")
	}
}

func TestDiffSubcommandMissingFileErrors(t *testing.T) {
	tmpDir := t.TempDir()
	if err := run([]string{"diff", "-a", filepath.Join(tmpDir, "missing1.json"), "-b", filepath.Join(tmpDir, "missing2.json")}, io.Discard, io.Discard); err == nil {
		t.Fatal("expected error for nonexistent model files")
	}
}
