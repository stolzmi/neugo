package nn

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadDenseModelRoundTrip(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 3},
		Linear(rng, 3, 4, XavierInit()),
		ReLU(),
		BatchNorm(4),
		Dropout(0.2),
		Linear(rng, 4, 2, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	// Run one Train-mode forward so BatchNorm accumulates non-trivial running stats.
	ctx := &Context{Mode: Train, RNG: NewRNG(2)}
	x := NewTensor([]int{2, 3})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	if _, err := model.Forward(ctx, x); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "model.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	infCtx := &Context{Mode: Inference}
	want, err := model.Forward(infCtx, x)
	if err != nil {
		t.Fatalf("Forward original: %v", err)
	}
	got, err := loaded.Forward(infCtx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v (loaded model diverges from original)", i, got.Data[i], want.Data[i])
		}
	}
}

func TestSaveLoadConvModelRoundTrip(t *testing.T) {
	rng := NewRNG(3)
	model, err := Sequential([]int{1, 6, 6, 1},
		Conv2D(rng, 1, 2, 3, HeInit()),
		ReLU(),
		MaxPool2D(2, 2),
		Flatten(),
		Linear(rng, 0, 1, XavierInit()),
		Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{1, 6, 6, 1})
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.2
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "cnn.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

// TestSaveLoadCoversRemainingModuleTypes exercises the module types not
// touched by the other round-trip tests: Tanh, LeakyReLU with a non-zero
// alpha, GELU, AvgPool2D, and a nested Sequential-within-Sequential model.
func TestSaveLoadCoversRemainingModuleTypes(t *testing.T) {
	rng := NewRNG(4)
	inner, err := Sequential([]int{1, 4}, Tanh(), LeakyReLU(0.15), GELU())
	if err != nil {
		t.Fatalf("Sequential (inner): %v", err)
	}
	model, err := Sequential([]int{1, 4, 4, 1},
		AvgPool2D(2, 2),
		Flatten(),
		inner,
		Linear(rng, 4, 2, XavierInit()),
		Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Sequential (outer): %v", err)
	}

	x := NewTensor([]int{1, 4, 4, 1})
	for i := range x.Data {
		x.Data[i] = float32(i)*0.1 - 0.5 // mix of negative/positive values to exercise LeakyReLU's alpha branch
	}
	ctx := &Context{Mode: Inference}
	want, err := model.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	path := filepath.Join(t.TempDir(), "remaining_types.json")
	if err := Save(model, path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := loaded.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward loaded: %v", err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

// TestLoadRejectsTruncatedParamData is the Finding-1 regression test: a
// param whose data array doesn't match the shape the module actually
// expects must fail Load with an error, never silently copy a short slice.
func TestLoadRejectsTruncatedParamData(t *testing.T) {
	// A "linear" module configured for 2x2 W (4 values) but the saved W
	// data only has 3 — truncated/corrupt file.
	doc := `{
		"type": "sequential",
		"modules": [
			{
				"type": "linear",
				"config": {"in_features": 2, "out_features": 2},
				"params": {
					"W": {"shape": [2, 2], "data": [1, 2, 3]},
					"B": {"shape": [2], "data": [0, 0]}
				}
			}
		]
	}`
	path := filepath.Join(t.TempDir(), "truncated.json")
	if err := os.WriteFile(path, []byte(doc), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading a truncated param array, got nil")
	}
}

func TestLoadMissingFileReturnsError(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "does-not-exist.json")); err == nil {
		t.Fatal("expected error loading a missing file, got nil")
	}
}

func TestLoadRejectsMalformedJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("not json"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading malformed JSON, got nil")
	}
}

// TestLoadRejectsNonPositiveLinearInFeatures is a regression test: a
// "linear" module with in_features<=0 (e.g. omitted or explicitly 0) used
// to reach Linear's lazy-build path, which leaves W/B nil, and then
// decodeModule's unconditional access to l.W.Value.Data would panic with a
// nil-pointer dereference. Load must reject it with a clean error instead.
func TestLoadRejectsNonPositiveLinearInFeatures(t *testing.T) {
	doc := `{
		"type": "sequential",
		"modules": [
			{
				"type": "linear",
				"config": {"in_features": 0, "out_features": 2},
				"params": {
					"W": {"shape": [0, 2], "data": []},
					"B": {"shape": [2], "data": [0, 0]}
				}
			}
		]
	}`
	path := filepath.Join(t.TempDir(), "zero_in_features.json")
	if err := os.WriteFile(path, []byte(doc), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading a linear module with in_features=0, got nil")
	}
}

// TestLoadRejectsNegativeConv2DChannels is a regression test: a "conv2d"
// module with a negative in_channels/out_channels/kernel_size used to reach
// newConv2D's unconditional tensor allocation, which panics with
// "makeslice: len out of range" for negative sizes. Load must reject it
// with a clean error instead.
func TestLoadRejectsNegativeConv2DChannels(t *testing.T) {
	doc := `{
		"type": "sequential",
		"modules": [
			{
				"type": "conv2d",
				"config": {"in_channels": -1, "out_channels": 2, "kernel_size": 3, "padding": 0},
				"params": {
					"W": {"shape": [2, -1, 3, 3], "data": []},
					"B": {"shape": [2], "data": [0, 0]}
				}
			}
		]
	}`
	path := filepath.Join(t.TempDir(), "negative_channels.json")
	if err := os.WriteFile(path, []byte(doc), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error loading a conv2d module with in_channels=-1, got nil")
	}
}

// TestSaveNilModelReturnsError is a regression test: Save(nil, path) used to
// panic inside encodeModule's *SequentialModel case when it dereferenced
// v.modules on a nil model pointer. Save must reject it with a clean error
// instead.
func TestSaveNilModelReturnsError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nil_model.json")
	if err := Save(nil, path); err == nil {
		t.Fatal("expected error saving a nil model, got nil")
	}
}
