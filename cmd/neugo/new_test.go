package main

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewSubcommandUnknownArchErrors(t *testing.T) {
	if err := run([]string{"new", "-arch", "bogus", "-out", t.TempDir()}, io.Discard, io.Discard); err == nil {
		t.Fatal("expected error for an unknown -arch")
	}
}

func TestNewSubcommandMissingOutErrors(t *testing.T) {
	if err := run([]string{"new", "-arch", "mlp"}, io.Discard, io.Discard); err == nil {
		t.Fatal("expected error when -out is missing")
	}
}

func TestNewSubcommandWritesExpectedPackageName(t *testing.T) {
	outDir := t.TempDir()
	var stderr bytes.Buffer
	if err := run([]string{"new", "-arch", "mlp", "-out", outDir, "-pkg", "myapp"}, io.Discard, &stderr); err != nil {
		t.Fatalf("run(new): %v", err)
	}
	data, err := os.ReadFile(filepath.Join(outDir, "main.go"))
	if err != nil {
		t.Fatalf("reading generated file: %v", err)
	}
	if !strings.Contains(string(data), "package myapp") {
		t.Errorf("generated file missing \"package myapp\":\n%s", data)
	}
}

// TestNewSubcommandGeneratesBuildableProject writes each architecture's
// scaffold into a scratch directory *inside this module* (so its
// "github.com/stolzmi/neugo/..." imports resolve against the local
// module, not the network) and actually runs `go build` on it — proof
// the templates aren't just well-formatted strings but real, compiling
// Go code against the current API.
func TestNewSubcommandGeneratesBuildableProject(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping go-build verification in -short mode")
	}
	cwd, err := os.Getwd() // cmd/neugo
	if err != nil {
		t.Fatal(err)
	}
	moduleRoot := filepath.Join(cwd, "..", "..")

	for arch := range newTemplates {
		t.Run(arch, func(t *testing.T) {
			scratchName := "zz_scaffold_test_" + arch
			scratchDir := filepath.Join(moduleRoot, scratchName)
			if err := os.RemoveAll(scratchDir); err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(scratchDir)

			if err := run([]string{"new", "-arch", arch, "-out", scratchDir, "-pkg", "main"}, io.Discard, io.Discard); err != nil {
				t.Fatalf("run(new): %v", err)
			}

			cmd := exec.Command("go", "build", "-o", os.DevNull, "./"+scratchName)
			cmd.Dir = moduleRoot
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go build failed for arch %q:\n%s", arch, out)
			}
		})
	}
}
