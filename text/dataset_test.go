package text

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadLineDatasetSkipsEmptyLinesAndStripsCR(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	content := "first line\r\n\nsecond line\nthird line\r\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	lines, err := LoadLineDataset(path)
	if err != nil {
		t.Fatalf("LoadLineDataset: %v", err)
	}
	want := []string{"first line", "second line", "third line"}
	if len(lines) != len(want) {
		t.Fatalf("got %d lines, want %d: %v", len(lines), len(want), lines)
	}
	for i := range want {
		if lines[i] != want[i] {
			t.Errorf("lines[%d] = %q, want %q", i, lines[i], want[i])
		}
	}
}

func TestLoadLineDatasetMissingFileReturnsError(t *testing.T) {
	if _, err := LoadLineDataset(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("expected error for a nonexistent file, got nil")
	}
}
