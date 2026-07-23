package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFindModuleRootLocatesGoMod(t *testing.T) {
	root, err := findModuleRoot()
	if err != nil {
		t.Fatalf("findModuleRoot: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("findModuleRoot returned %s, but go.mod not found there: %v", root, err)
	}
}

// TestRunGeneratesNonTrivialConstructorList runs the real generator
// against this repo's actual nn package and sanity-checks the output —
// proof it's actually discovering the constructors (not silently
// producing an empty file, the bug this test would have caught before
// the Types[i].Funcs fix), and that a couple of well-known constructors
// with doc comments show up with their comment attached.
func TestRunGeneratesNonTrivialConstructorList(t *testing.T) {
	root, err := findModuleRoot()
	if err != nil {
		t.Fatalf("findModuleRoot: %v", err)
	}
	realOut := filepath.Join(root, outPathName)
	original, hadOriginal := readIfExists(t, realOut)
	defer func() {
		if hadOriginal {
			os.WriteFile(realOut, original, 0644)
		} else {
			os.Remove(realOut)
		}
	}()

	if err := run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	data, err := os.ReadFile(realOut)
	if err != nil {
		t.Fatalf("reading generated file: %v", err)
	}
	out := string(data)

	if !strings.Contains(out, "## `Linear`") {
		t.Error("generated doc missing the Linear constructor")
	}
	if !strings.Contains(out, "## `RNN`") {
		t.Error("generated doc missing the RNN constructor")
	}
	if strings.Count(out, "## `") < 30 {
		t.Errorf("generated doc only lists %d constructors, expected 30+ (regression toward the Package.Funcs-only bug)", strings.Count(out, "## `"))
	}
}

func readIfExists(t *testing.T, path string) ([]byte, bool) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false
		}
		t.Fatal(err)
	}
	return data, true
}
