package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func TestExportSubcommand(t *testing.T) {
	// Build a small model using nn.Sequential + nn.Save
	rng := nn.NewRNG(1)
	m, err := nn.Sequential([]int{1, 2},
		nn.Linear(rng, 2, 3, nil), nn.ReLU(),
		nn.Linear(rng, 3, 1, nil), nn.Sigmoid(),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Save model to temp directory
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "model.json")
	if err := nn.Save(m, modelPath); err != nil {
		t.Fatal(err)
	}

	// Export to output file
	outPath := filepath.Join(tmpDir, "model.go")
	stderr := io.Discard

	// Call run with export subcommand
	err = run([]string{"export", "-model", modelPath, "-out", outPath, "-pkg", "mymodel"}, stderr)
	if err != nil {
		t.Fatalf("run() failed: %v", err)
	}

	// Verify output file was created
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("output file not created: %v", err)
	}

	// Verify output contains expected package name
	output := string(data)
	if !strings.Contains(output, "package mymodel") {
		t.Fatalf("output missing 'package mymodel', got:\n%s", output)
	}
}

func TestUnknownSubcommandErrors(t *testing.T) {
	stderr := &bytes.Buffer{}
	err := run([]string{"bogus"}, stderr)
	if err == nil {
		t.Fatal("run() should return error for unknown subcommand")
	}
}
