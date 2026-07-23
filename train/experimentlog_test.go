// train/experimentlog_test.go
package train

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func readJSONLEntries(t *testing.T, path string) []experimentLogEntry {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	var entries []experimentLogEntry
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var e experimentLogEntry
		if err := json.Unmarshal(scanner.Bytes(), &e); err != nil {
			t.Fatalf("unmarshal line %q: %v", scanner.Text(), err)
		}
		entries = append(entries, e)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return entries
}

func TestExperimentLogWritesRunStartEpochsAndRunEnd(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runs.jsonl")
	model, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	logCb := ExperimentLog(path, map[string]string{"lr": "0.05", "arch": "xor-mlp"})
	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	if _, err := trainer.Fit(x, y, Epochs(3), BatchSize(4), Seed(1), Callbacks(logCb)); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if logCb.LastError != nil {
		t.Fatalf("LastError = %v, want nil", logCb.LastError)
	}

	entries := readJSONLEntries(t, path)
	if len(entries) != 5 { // run_start + 3 epochs + run_end
		t.Fatalf("got %d entries, want 5: %+v", len(entries), entries)
	}
	if entries[0].Type != "run_start" || entries[0].Meta["arch"] != "xor-mlp" {
		t.Errorf("entries[0] = %+v, want run_start with Meta[arch]=xor-mlp", entries[0])
	}
	for i := 0; i < 3; i++ {
		e := entries[1+i]
		if e.Type != "epoch" || e.Epoch != i {
			t.Errorf("entries[%d] = %+v, want epoch %d", 1+i, e, i)
		}
		if e.RunID != entries[0].RunID {
			t.Errorf("entries[%d].RunID = %q, want %q (same run)", 1+i, e.RunID, entries[0].RunID)
		}
	}
	if entries[4].Type != "run_end" {
		t.Errorf("entries[4] = %+v, want run_end", entries[4])
	}
}

func TestExperimentLogAppendsAcrossMultipleRuns(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sweep.jsonl")
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	for i := 0; i < 2; i++ {
		model, err := xorModel(nn.NewRNG(int64(i + 1)))
		if err != nil {
			t.Fatalf("Sequential: %v", err)
		}
		trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
		logCb := ExperimentLog(path, map[string]string{"trial": string(rune('a' + i))})
		if _, err := trainer.Fit(x, y, Epochs(2), BatchSize(4), Seed(1), Callbacks(logCb)); err != nil {
			t.Fatalf("Fit: %v", err)
		}
	}

	entries := readJSONLEntries(t, path)
	if len(entries) != 8 { // 2 runs * (run_start + 2 epochs + run_end)
		t.Fatalf("got %d entries across 2 runs, want 8", len(entries))
	}
	if entries[0].RunID == entries[4].RunID {
		t.Error("two separate Fit calls produced the same RunID, want distinct run_ids")
	}
}

func TestExperimentLogRecordsValidationMetrics(t *testing.T) {
	path := filepath.Join(t.TempDir(), "val.jsonl")
	model, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	logCb := ExperimentLog(path, nil)
	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	if _, err := trainer.Fit(x, y, Epochs(2), BatchSize(4), Seed(1), Validation(x, y), Callbacks(logCb)); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	entries := readJSONLEntries(t, path)
	found := false
	for _, e := range entries {
		if e.Type == "epoch" {
			found = true
			if e.ValLoss == 0 && e.ValAcc == 0 {
				t.Errorf("epoch entry %+v has zero-valued val metrics despite Validation() being set", e)
			}
		}
	}
	if !found {
		t.Fatal("no epoch entries found")
	}
}

func TestExperimentLogBadPathSetsLastErrorNotPanic(t *testing.T) {
	// A directory that doesn't exist can't be opened for writing.
	badPath := filepath.Join(t.TempDir(), "does", "not", "exist", "runs.jsonl")
	model, err := xorModel(nn.NewRNG(1))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	logCb := ExperimentLog(badPath, nil)
	trainer := New(model, Adam(0.05, 0.9, 0.999, 1e-8), BCELoss())
	if _, err := trainer.Fit(x, y, Epochs(2), BatchSize(4), Seed(1), Callbacks(logCb)); err != nil {
		t.Fatalf("Fit itself should not fail just because logging failed: %v", err)
	}
	if logCb.LastError == nil {
		t.Fatal("LastError = nil, want an error for an unopenable path")
	}
}
